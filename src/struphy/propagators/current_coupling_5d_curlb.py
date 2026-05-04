"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.pic.pushing import pusher_kernels_gc
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
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
            assert new.space == "Particles5D"
            self._energetic_ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        """Configuration options for :class:`CurrentCoupling5DCurlb`.

        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Perturbed magnetic field variable used to build the total magnetic
            field in the coupling term.

        ep_scale : float, default=1.0
            Scaling factor applied to energetic-particle contributions in the
            accumulation kernel.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used for the FEEC mass matrix block.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters used by the accumulator.
            If ``None``, defaults to ``FilterParameters()``.
        """

        # propagator options
        b_tilde: FEECVariable = None
        ep_scale: float = 1.0
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            assert isinstance(self.b_tilde, FEECVariable)
            assert isinstance(self.ep_scale, float)

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

    @profile
    def allocate(self, verbose: bool = False):
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
        curl_unit_b1 = self.projected_equil.curl_unit_b1
        self._b2 = self.projected_equil.b2

        # magnetic field
        self._b_tilde = self.options.b_tilde.spline.vector

        # scaling factor
        epsilon = self.variables.energetic_ions.species.equation_params.epsilon

        # temporary vectors to avoid memory allocation
        self._b_full = self._b2.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()
        self._u_avg = self.variables.u.spline.vector.space.zeros()

        # define Accumulator and arguments
        self._ACC = Accumulator(
            self.variables.energetic_ions.particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels_gc.cc_lin_mhd_5d_curlb),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
            filter_params=self.options.filter_params,
        )

        self._args_accum_kernel = (
            epsilon,
            self.options.ep_scale,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
            unit_b1[0]._data,
            unit_b1[1]._data,
            unit_b1[2]._data,
            curl_unit_b1[0]._data,
            curl_unit_b1[1]._data,
            curl_unit_b1[2]._data,
            self._u_form_int,
        )

        # define Pusher
        if self.options.u_space == "Hcurl":
            pusher_kernel = Pyccelkernel(pusher_kernels_gc.push_gc_cc_J1_Hcurl)
        elif self.options.u_space == "Hdiv":
            pusher_kernel = Pyccelkernel(pusher_kernels_gc.push_gc_cc_J1_Hdiv)
        elif self.options.u_space == "H1vec":
            pusher_kernel = Pyccelkernel(pusher_kernels_gc.push_gc_cc_J1_H1vec)
        else:
            raise ValueError(
                f'{self.options.u_space  =} not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
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
            curl_unit_b1[0]._data,
            curl_unit_b1[1]._data,
            curl_unit_b1[2]._data,
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
        )

        _BC = -1 / 4 * self._ACC.operators[0]

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

        b_full += self._b_tilde
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
