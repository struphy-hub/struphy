import logging
from dataclasses import dataclass
from feectools.linalg.solvers import inverse
from line_profiler import profile
from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
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

        \mathbb M_{1/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right) = \frac{\Delta t}{2} \frac{1}{\varepsilon} \mathbb M_{B_0/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right)\,.
    """

    class Variables:
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

    @dataclass
    class Options:
        # propagator options
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
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

    @profile
    def allocate(self, verbose: bool = False):
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

        # Instantiate linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
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
        jn1 = self._solver.solve(rhsv, out=self._j_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_dj = self.update_feec_variables(j=jn1)

        if self._info:
            logger.info(f"Status     for FluidCold: {info['success']}")
            logger.info(f"Iterations for FluidCold: {info['niter']}")
            logger.info(f"Maxdiff j1 for FluidCold: {max_dj}")
            logger.info("")
