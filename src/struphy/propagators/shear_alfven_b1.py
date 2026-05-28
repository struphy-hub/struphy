import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenB1(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \tilde{\mathbf B } \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x - \int_\Omega \mathbf C \cdot\nabla\times   \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathcal {T^2}^\top \mathbb C \mathbb M_1^{-1}\\ - \mathbb M_1^{-1} \mathbb C^\top \mathcal {T^2} (\mathbb M^\rho_2)^{-1} & 0 \end{bmatrix}
        \begin{bmatrix} {\mathbb M^\rho_2}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_1(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2` is a weighted mass matrix in 2-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenB1`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"Hdiv"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hcurl"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hdiv")
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`ShearAlfvenB1`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver used for the Schur-reduced system.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner for ``M2n`` block.
        solver_params : SolverParameters, default=None
            Solver controls for the Schur solve.
        solver_M1 : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver used for inverting ``M1``.
        precond_M1 : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner for ``M1`` inversion.
        solver_params_M1 : SolverParameters, default=None
            Solver controls for ``M1`` inversion.
        """

        # propagator options
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        solver_M1: LiteralOptions.OptsSymmSolver = "pcg"
        precond_M1: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params_M1: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.solver_M1, LiteralOptions.OptsSymmSolver)
            check_option(self.precond_M1, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.solver_params_M1 is None:
                self.solver_params_M1 = SolverParameters()

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
        self._info = self.options.solver_params.info

        # define inverse of M1
        if self.options.precond_M1 is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond_M1)
            pc = pc_class(self.mass_ops.M1)

        M1_inv = inverse(
            self.mass_ops.M1,
            self.options.solver_M1,
            pc=pc,
            tol=self.options.solver_params_M1.tol,
            maxiter=self.options.solver_params_M1.maxiter,
            verbose=self.options.solver_params_M1.verbose,
        )

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M2n
        self._B = 1 / 2 * self.mass_ops.M2B @ self.derham.curl
        # I still have to invert M1
        self._C = 1 / 2 * M1_inv @ self.derham.curl.T @ self.mass_ops.M2B

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, "M2n"))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # allocate dummy vectors to avoid temporary array allocations
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector

        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @profile
    def __call__(self, dt):
        # current variables
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # solve for new u coeffs
        byn = self._B.dot(bn, out=self._byn)

        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_diffs = self.update_feec_variables(u=un1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for ShearAlfvenB1: {info['success']}")
            logger.info(f"Iterations for ShearAlfvenB1: {info['niter']}")
            logger.info(f"Maxdiff up for ShearAlfvenB1: {max_diffs['u']}")
            logger.info(f"Maxdiff b2 for ShearAlfvenB1: {max_diffs['b']}")
            logger.info("")
