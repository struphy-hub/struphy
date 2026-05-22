import logging
from dataclasses import dataclass
from typing import Literal

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class MaxwellWeakAmpere(Propagator):
    r"""FEEC discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x - \int_{\Omega} \mathbf B \cdot \nabla \times \mathbf F \,\textrm d \mathbf x &= 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E &= 0\,.

    Time discretization:

    - implicit: Crank-Nicolson (implicit mid-point)
    - explicit: explicit RK methods from ButcherTableau

    System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    class Variables:
        """Container for Maxwell equation variables.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        b : FEECVariable
            Magnetic field in H(div) space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MaxwellWeakAmpere`.

        Parameters
        ----------
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping scheme to use for the Maxwell equations.

            - ``"implicit"``: Crank-Nicolson (implicit mid-point) scheme.
            - ``"explicit"``: explicit Runge-Kutta methods from ``ButcherTableau``.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Name of the preconditioner configuration.
            Currently used only for implicit time stepping.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        butcher : ButcherTableau, default=None
            Butcher tableau for explicit Runge-Kutta methods.
            Only used when ``algo="explicit"``.
            If ``None``, defaults to ``ButcherTableau()``.

        Notes
        -----
        System size reduction is performed via :class:`SchurSolver` for
        implicit time stepping to eliminate the magnetic field and reduce
        the system to the electric field equation.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.algo == "explicit" and self.butcher is None:
                self.butcher = ButcherTableau()

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
    def allocate(self, verbose: bool = False):
        # obtain needed matrices
        M1 = self.mass_ops.M1
        M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner for M1 + ...
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(M1)

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info
            # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            _A = M1

            # no dt
            self._B = -1 / 2 * curl.T @ M2
            self._C = 1 / 2 * curl

            # Instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _A,
                _BC,
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )

            # pre-allocate arrays
            self._byn = self._B.codomain.zeros()
        else:
            self._info = False

            # define vector field
            M1_inv = inverse(
                M1,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            weak_curl = M1_inv @ curl.T @ M2

            # allocate output of vector field
            out1 = self.variables.e.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                weak_curl.dot(y2, out=out)
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                curl.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.e.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

        # allocate place-holder vectors to avoid temporary array allocations in __call__
        self._e_tmp1 = self.variables.e.spline.vector.space.zeros()
        self._e_tmp2 = self.variables.e.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # solve for new e coeffs
            self._B.dot(bn, out=self._byn)

            en1, info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)

            # new b coeffs
            _e = en.copy(out=self._e_tmp2)
            _e += en1
            bn1 = self._C.dot(_e, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(e=en1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for Maxwell: {info['success']}")
                logger.info(f"Iterations for Maxwell: {info['niter']}")
                logger.info(f"Maxdiff e for Maxwell: {diffs['e']}")
                logger.info(f"Maxdiff b for Maxwell: {diffs['b']}")
                logger.info("")
