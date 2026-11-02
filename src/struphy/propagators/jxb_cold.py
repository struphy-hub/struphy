import logging
from dataclasses import dataclass
from typing import get_args

from feectools.linalg.solvers import inverse
from line_profiler import profile
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class JxBCold(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf j \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x
        = \frac{1}{\varepsilon} \int_\Omega \frac{1}{n_0} (\mathbf j \times \mathbf B_0) \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), such that

    .. math::

        \mathbb M^1_{1/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right) = -\frac{\Delta t}{2} \frac{1}{\varepsilon} \mathbb M^1_{B_0/n_0} \left( \mathbf j^{n+1} + \mathbf j^n \right)\,,

    where :math:`\mathbb M^1_{1/n_0}` is the 1-form mass matrix weighted with :math:`1/n_0` (``M1ninv``) and
    :math:`\mathbb M^1_{B_0/n_0}` the skew-symmetric one of :math:`(\mathbf B_0 \times \mathbf j) \cdot \mathbf F / n_0`
    (``M1Bninv``), hence the minus sign.
    """

    class Variables:
        """Container for variables advanced by :class:`JxBCold`.

        Attributes
        ----------
        j : FEECVariable
            Current variable in ``"Hcurl"`` space.
        """

        def __init__(self):
            self._j: FEECVariable = None

        @property
        def j(self) -> FEECVariable:
            return self._j

        @j.setter
        def j(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._j = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`JxBCold`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            Iterative solver used for the linear current update. The system matrix
            :math:`\mathbb M^1_{1/n_0} + \frac{\Delta t}{2 \varepsilon} \mathbb M^1_{B_0/n_0}` is not symmetric
            (:math:`\mathbb M^1_{B_0/n_0}` is skew-symmetric), so a solver for general matrices is needed.
            The symmetric solvers of older parameter files (``"pcg"``, ...) are replaced by ``"pbicgstab"``,
            with a warning.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner built from the weighted mass matrix ``M1ninv``; only used by ``solver="pbicgstab"``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """

        # propagator options
        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # the system is not symmetric: symmetric solvers (the default up to v3.6) diverge for dt > epsilon
            if self.solver in get_args(LiteralOptions.OptsSymmSolver):
                logger.warning(
                    f"JxBCold: the system is not symmetric, replacing solver={self.solver!r} by 'pbicgstab'."
                )
                self.solver = "pbicgstab"

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
        self._info = self.options.solver_params.info

        epsilon = self.variables.j.species.equation_params.epsilon

        # mass matrix in system (M - dt/2 * A)*j^(n + 1) = (M + dt/2 * A)*j^n
        self._M = self.mass_ops.M1ninv
        self._A = -1 / epsilon * self.mass_ops.M1Bninv  # no dt

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate linear solver (only pbicgstab takes a preconditioner)
        pc_kwargs = {"pc": pc} if self.options.solver == "pbicgstab" else {}
        self._solver = inverse(
            self._M,
            self.options.solver,
            **pc_kwargs,
            x0=self.variables.j.spline.vector,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_j = self._M.codomain.zeros()
        self._j_new = self.variables.j.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current variables
        jn = self.variables.j.spline.vector

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt / 2.0 * self._A
        rhs = self._M + dt / 2.0 * self._A

        rhsv = rhs.dot(jn, out=self._rhs_j)

        self._solver.linop = lhs

        # solve linear system for updated j coefficients (in-place)
        with ProfileManager.profile_region(self._solve_region, functions=[self._solver.solve]):
            jn1 = self._solver.solve(rhsv, out=self._j_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_dj = self.update_feec_variables(j=jn1)

        if self._info:
            logger.info(f"Status     for FluidCold: {info['success']}")
            logger.info(f"Iterations for FluidCold: {info['niter']}")
            logger.info(f"Maxdiff j1 for FluidCold: {max_dj}")
            logger.info("")
