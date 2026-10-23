"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb import cc_lin_mhd_5d_curlb
from struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb_mf import cc_lin_mhd_5d_curlb_mf
from struphy.pic.accumulation.kernels.marker_operator_rank1_dot import marker_operator_rank1_dot
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector, MarkerOperator
from struphy.pic.pushing.kernels.push_gc_cc_J1_H1vec import push_gc_cc_J1_H1vec
from struphy.pic.pushing.kernels.push_gc_cc_J1_Hdiv import push_gc_cc_J1_Hdiv
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurrentCoupling5DCurlb(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V\, \textnormal{d} \mathbf x = - \frac{A_\textnormal{h}}{A_b} \iint \frac{f^\text{vol}}{B^*_\parallel} v_\parallel^2 (\nabla \times \mathbf b_0)  \times \mathbf B \cdot \mathbf V \, \textnormal{d} \mathbf x \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
                \\
                &\frac{\partial f}{\partial t} = - \frac{1}{B^*_\parallel} v_\parallel (\nabla \times \mathbf b_0) \cdot (\mathbf B \times \tilde{\mathbf U}) \nabla_{v_\parallel}f \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ V_\parallel^{n+1} - V_\parallel^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} 
        \begin{bmatrix} 
            0 & - (\mathbb{M}^{2,n})^{-1} \left\{ \mathbb{L}^2 \frac{1}{\bar{\sqrt{g}}} \right\}\cdot_\text{vector} \left\{\bar{b}^{\nabla \times}_0 (\bar{B}^\times_f)^\top \bar{V}_\parallel \frac{1}{\bar{\sqrt{g}}}\right\} \frac{1}{\bar B^{*0}_\parallel})
            \\  
            \frac{1}{\bar B^{*0}_\parallel} \left\{\bar{b}^{\nabla \times}_0 (\bar{B}^\times_f)^\top \bar{V}_\parallel \frac{1}{\bar{\sqrt{g}}}\right\}\, \cdot_\text{vector} \left\{\frac{1}{\bar{\sqrt{g}}}(\mathbb{L}²)^\top\right\} (\mathbb{M}^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            (\mathbb{M}^{2,n})^{-1} (\mathbf u^{n+1} + \mathbf u^n)
            \\
            \frac{A_\textnormal{h}}{A_b} W (V_\parallel^{n+1} + V_\parallel^n)
        \end{bmatrix} \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    """

    class Variables:
        """Container for variables advanced by :class:`CurrentCoupling5DCurlb`.

        Attributes
        ----------
        u : FEECVariable
            Fluid-like FEEC variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        energetic_ions : PICVariable
            Energetic-ion particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._energetic_ions: PICVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def energetic_ions(self) -> PICVariable:
            return self._energetic_ions

        @energetic_ions.setter
        def energetic_ions(self, new):
            assert isinstance(new, PICVariable)
            assert "Particles5D" in new.space
            self._energetic_ions = new

    def __init__(self, b_tilde: FEECVariable = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the curl-B
            coupling term. If ``None``, only the equilibrium field is used.
        """
        self.variables = self.Variables()
        self.b_tilde = b_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`CurrentCoupling5DCurlb`.

        Parameters
        ----------
        ep_scale : float, default=None
            Factor of the energetic-particle contributions (accumulation): the mass ratio
            :math:`A_\textnormal{h}/A_\textnormal{b}` of the energetic ions and the MHD bulk if ``None``.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable. Only ``"Hdiv"`` and
            ``"H1vec"`` are supported.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used for the FEEC mass matrix block.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Filter of the accumulated (particle-to-grid) vectors; defaults to ``FilterParameters()`` (no filter).
            The three-point (binomial) filter ``FilterParameters(use_filter="three_point")`` reduces the noise,
            but the propagator then no longer conserves the energy exactly (the filter is applied to the
            accumulated vectors only, not to the fields seen by the particles).

        matrix_free : bool, default=True
            Apply the accumulated matrix :math:`\sum_p c_p (\vec \Lambda_p \cdot \mathbf e_p)
            (\vec \Lambda_p \cdot \mathbf e_p)^\top` (rank one per marker) in each iteration of the solver from data
            stored per marker (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`), instead of assembling it: one spline evaluation and
            one deposition per marker and iteration instead of the 6 blocks of size :math:`n_\textnormal{basis}^2`
            per marker (much cheaper for cubic splines). Same result up to the solver tolerance.
        """

        # propagator options
        ep_scale: float = None
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        matrix_free: bool = True

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            assert self.ep_scale is None or isinstance(self.ep_scale, float)
            assert isinstance(self.matrix_free, bool)

            # the accumulation kernel cc_lin_mhd_5d_curlb only has H1vec and Hdiv branches
            if self.u_space not in ("Hdiv", "H1vec"):
                raise ValueError(
                    f'{self.u_space = } is not supported by CurrentCoupling5DCurlb, choose from "Hdiv" or "H1vec".',
                )

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        # factor of the energetic-particle contributions (default: mass ratio A_h/A_b)
        self._ep_scale = self.options.ep_scale
        if self._ep_scale is None:
            self._ep_scale = self.variables.energetic_ions.species.mass_number / self.variables.u.species.mass_number
            logger.info(f"{self.__class__.__name__}: ep_scale = A_h/A_b = {self._ep_scale} (mass ratio, default).")

        if self.options.u_space == "H1vec":
            self._u_form_int = 0
        else:
            self._u_form_int = int(self.derham.space_to_form[self.options.u_space])

        # call operatros
        id_M = "M" + self.derham.space_to_form[self.options.u_space] + "n"
        _A = getattr(self.mass_ops, id_M)

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._b2 = self.projected_equil.b2

        # scaling factor
        epsilon = self.variables.energetic_ions.species.equation_params.epsilon

        # temporary vectors to avoid memory allocation
        self._b_full = self._b2.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()
        self._u_avg = self.variables.u.spline.vector.space.zeros()

        # define Accumulator and arguments
        particles = self.variables.energetic_ions.particles
        self._args_accum_kernel = (
            epsilon,
            self._ep_scale,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
            unit_b1[0]._data,
            unit_b1[1]._data,
            unit_b1[2]._data,
            curl_unit_b2[0]._data,
            curl_unit_b2[1]._data,
            curl_unit_b2[2]._data,
            self._u_form_int,
        )
        if self.options.matrix_free:
            # the vector, and the data of the matrix per marker (in the first free columns of the marker array)
            column = particles.args_markers.first_free_idx
            assert particles.n_cols_aux >= 4, "matrix_free needs 4 free (auxiliary) marker columns."
            self._ACC = AccumulatorVector(
                particles,
                self.options.u_space,
                cc_lin_mhd_5d_curlb_mf,
                self.mass_ops,
                self.domain.args_domain,
                filter_params=self.options.filter_params,
            )
            self._args_accum_kernel += (column,)
            self._A_particles = MarkerOperator(
                particles,
                self.options.u_space,
                marker_operator_rank1_dot,
                self.mass_ops,
                self.domain.args_domain,
                (self._u_form_int, column),
                symmetry="symm",
            )
        else:
            self._ACC = Accumulator(
                particles,
                self.options.u_space,
                cc_lin_mhd_5d_curlb,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry="symm",
                filter_params=self.options.filter_params,
            )
            self._A_particles = self._ACC.operators[0]

        # define Pusher
        if self.options.u_space == "Hdiv":
            pusher_kernel = push_gc_cc_J1_Hdiv
        elif self.options.u_space == "H1vec":
            pusher_kernel = push_gc_cc_J1_H1vec
        else:
            raise ValueError(
                f'{self.options.u_space  =} not valid, choose from "Hdiv" or "H1vec".',
            )

        args_pusher_kernel = (
            self.derham.args_derham,
            epsilon,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
            unit_b1[0]._data,
            unit_b1[1]._data,
            unit_b1[2]._data,
            curl_unit_b2[0]._data,
            curl_unit_b2[1]._data,
            curl_unit_b2[2]._data,
            self._u_avg[0]._data,
            self._u_avg[1]._data,
            self._u_avg[2]._data,
        )

        self._pusher = Pusher(
            self.variables.energetic_ions.particles,
            pusher_kernel,
            args_pusher_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

        _BC = -1 / 4 * self._A_particles

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b2.copy(out=self._b_full)

        if self.b_tilde is not None:
            b_full += self.b_tilde.spline.vector
        b_full.update_ghost_regions()

        self._ACC(
            *self._args_accum_kernel,
        )

        # solve
        un1, info = self._schur_solver(
            un,
            -self._ACC.vectors[0] / 2,
            dt,
            out=self._u_new,
        )

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg)
        _u += un1
        _u *= 0.5

        _u.update_ghost_regions()

        self._pusher(dt)

        # update u coefficients
        diffs = self.update_feec_variables(u=un1)

        # update_weights
        if self.variables.energetic_ions.species.weights_params.control_variate:
            self.variables.energetic_ions.particles.update_weights()

        if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for CurrentCoupling5DCurlb: {info['success']}")
            logger.info(f"Iterations for CurrentCoupling5DCurlb: {info['niter']}")
            logger.info(f"Maxdiff up for CurrentCoupling5DCurlb: {diffs['u']}")
            logger.info("")
