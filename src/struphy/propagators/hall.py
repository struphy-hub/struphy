import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.species import Species
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class Hall(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x + \frac{1}{\varepsilon} \int_\Omega \nabla\times \mathbf C \cdot  \left( \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point):

    .. math::

        \mathbf b^{n+1} - \mathbf b^n
        = \frac{\Delta t}{2} \mathbb M_1^{-1} \mathbb C^\top  \mathbb M^{\mathcal{T},\rho}_2  \mathbb C  (\mathbf b^{n+1} + \mathbf b^n)  ,

    where :math:`\mathbb M^{\mathcal{T},\rho}_2` is a weighted mass matrix in 2-space, the weight being :math:`\frac{\mathcal{T}}{\rho_0}`,
    the MHD equilibirum density :math:`\rho_0` as a 0-form, and rotation matrix :math:`\mathcal{T} \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,`.
    The solution of the above system is based on the Pre-conditioned Biconjugate Gradient Stabilized algortihm (PBiConjugateGradientStab).
    """

    class Variables:
        """Container for variables advanced by :class:`Hall`.

        Attributes
        ----------
        b : FEECVariable
            Magnetic-field variable in ``"Hcurl"`` space.
        """

        def __init__(self):
            self._b: FEECVariable = None

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._b = new

    def __init__(self, epsilon_from: Species = None):
        """
        Parameters
        ----------
        epsilon_from : Species, default=None
            Species instance from which to read the Hall parameter ``epsilon``.
            If ``None``, ``epsilon`` defaults to ``1.0``.
        """
        self.variables = self.Variables()
        self.epsilon_from = epsilon_from

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`Hall`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            General iterative solver used for the linear Hall update.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the ``M1`` block.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """

        # propagator options
        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsGenSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

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
        if self.epsilon_from is None:
            epsilon = 1.0
        else:
            epsilon = self.epsilon_from.equation_params.epsilon

        self._info = self.options.solver_params.info
        self._tol = self.options.solver_params.tol
        self._maxiter = self.options.solver_params.maxiter
        self._verbose = self.options.solver_params.verbose

        # mass matrix in system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        id_M = "M1"
        id_M2Bn = "M2Bn"
        self._M = getattr(self.mass_ops, id_M)
        self._M2Bn = getattr(self.mass_ops, id_M2Bn)
        self._A = 1.0 / epsilon * self.derham.curl.T @ self._M2Bn @ self.derham.curl

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
            x0=self.variables.b.spline.vector,
            tol=self._tol,
            maxiter=self._maxiter,
            verbose=self._verbose,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_b = self._M.codomain.zeros()
        self._b_new = self.variables.b.spline.vector.space.zeros()

    def __call__(self, dt):
        # current variables
        bn = self.variables.b.spline.vector

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt / 2.0 * self._A
        rhs = self._M + dt / 2.0 * self._A

        # solve linear system for updated b coefficients (in-place)
        rhs = rhs.dot(bn, out=self._rhs_b)
        self._solver.linop = lhs

        bn1 = self._solver.solve(rhs, out=self._b_new)
        info = self._solver._info

        # write new coeffs into self.feec_vars
        max_db = self.update_feec_variables(b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for Hall: {info['success']}")
            logger.info(f"Iterations for Hall: {info['niter']}")
            logger.info(f"Maxdiff b1 for Hall: {max_db['b']}")
            logger.info("")
