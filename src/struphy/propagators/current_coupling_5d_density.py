import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurrentCoupling5DDensity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \iiint f^\text{vol} \left(1 - \frac{B_\parallel}{B^*_\parallel}\right) \tilde{\mathbf U} \times \mathbf B_f \cdot \mathbf V \, \textnormal{d} \mathbf{x} \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

    FE coefficients update:

    .. math::

        \mathbf u^{n+1} - \mathbf u^n = -\frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \mathbb{L}²{\mathbb{B}}^\times_f \mathbb{N}(1/g) \mathbb{W} \mathbb{N}\left(1- \frac{\hat B^0_\parallel}{\hat B^{*0} _\parallel}\right) (\mathbb{L}²)^\top \frac{\Delta t}{2} \cdot (\mathbf u^{n+1} + \mathbf u^n) \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    """

    class Variables:
        """Container for variables advanced by :class:`CurrentCoupling5DDensity`.

        Attributes
        ----------
        u : FEECVariable
            Fluid-like FEEC variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        """

        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

    def __init__(
        self,
        energetic_ions: PICVariable = None,
        b_tilde: FEECVariable = None,
    ):
        self.variables = self.Variables()
        self.energetic_ions = energetic_ions
        self.b_tilde = b_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`CurrentCoupling5DDensity`.

        Parameters
        ----------
        ep_scale : float, default=1.0
            Scaling factor applied in the energetic-particle accumulation
            kernel.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used for the linear system.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the FEEC mass matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters used by the accumulator.
            If ``None``, defaults to ``FilterParameters()``.
        """

        # propagator options
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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        if self.options.u_space == "H1vec":
            self._u_form_int = 0
        else:
            self._u_form_int = int(self.derham.space_to_form[self.options.u_space])

        # call operatros
        id_M = "M" + self.derham.space_to_form[self.options.u_space] + "n"
        self._A = getattr(self.mass_ops, id_M)

        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b1 = self.projected_equil.curl_unit_b1
        self._b2 = self.projected_equil.b2

        # scaling factor
        epsilon = self.energetic_ions.species.equation_params.epsilon

        # temporary vectors to avoid memory allocation
        self._b_full = self._b2.space.zeros()
        self._rhs_v = self.variables.u.spline.vector.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()

        # define Accumulator and arguments
        self._ACC = Accumulator(
            self.energetic_ions.particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels_gc.cc_lin_mhd_5d_D),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=False,
            symmetry="asym",
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

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # linear solver
        self._A_inv = inverse(
            self._A,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b2.copy(out=self._b_full)

        b_full += self.b_tilde.spline.vector
        b_full.update_ghost_regions()

        self._ACC(
            *self._args_accum_kernel,
        )

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = self._A - dt / 2 * self._ACC.operators[0]
        rhs = self._A + dt / 2 * self._ACC.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._A_inv.linop = lhs

        _u = self._A_inv.solve(rhs, out=self._u_new)
        info = self._A_inv._info

        diffs = self.update_feec_variables(u=_u)

        if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for CurrentCoupling5DDensity: {info['success']}")
            logger.info(f"Iterations for CurrentCoupling5DDensity: {info['niter']}")
            logger.info(f"Maxdiff up for CurrentCoupling5DDensity: {diffs['u']}")
            logger.info("")
