import logging
from dataclasses import dataclass
from line_profiler import profile
from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.schur_solver import SchurSolver, SchurSolverFull
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class OhmCold(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf j \in H(\textnormal{curl})` and :math:`\mathbf E \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x &= \frac{1}{\varepsilon} \int_\Omega \mathbf E \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        -\frac{\partial \mathbf E}{\partial t} &= \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, such that

    .. math::

        \begin{bmatrix}
            \mathbf j^{n+1} - \mathbf j^n \\
            \mathbf e^{n+1} - \mathbf e^n
        \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix}
            0 & \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} \\
            - \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \alpha^2 \mathbb M_{1/n_0} (\mathbf j^{n+1} + \mathbf j^{n}) \\
            \mathbb M_1 (\mathbf e^{n+1} + \mathbf e^{n})
        \end{bmatrix} \,.
    """

    class Variables:
        """Container for variables advanced by :class:`OhmCold`.

        Attributes
        ----------
        j : FEECVariable
            Current variable in ``"Hcurl"`` space.
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        """
        def __init__(self):
            self._j: FEECVariable = None
            self._e: FEECVariable = None

        @property
        def j(self) -> FEECVariable:
            return self._j

        @j.setter
        def j(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._j = new

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        """Configuration options for :class:`OhmCold`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the weighted mass matrix ``M1ninv``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """
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

        self._alpha = self.variables.j.species.equation_params.alpha
        self._epsilon = self.variables.j.species.equation_params.epsilon

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1ninv

        self._B = -1 / 2 * 1 / self._epsilon * self.mass_ops.M1  # no dt

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate Schur solver (constant in this case)
        _BC = 1 / 2 * self._alpha**2 / self._epsilon * self._B

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        j = self.variables.j.spline.vector
        e = self.variables.e.spline.vector

        self._tmp_j1 = j.space.zeros()
        self._tmp_j2 = j.space.zeros()
        self._tmp_e1 = e.space.zeros()
        self._tmp_e2 = e.space.zeros()

    def __call__(self, dt):
        # current variables
        jn = self.variables.j.spline.vector
        en = self.variables.e.spline.vector

        # in-place solution (no tmps created here)
        Ben = self._B.dot(en, out=self._tmp_e1)

        jn1, info = self._schur_solver(jn, Ben, dt, out=self._tmp_j1)

        en1 = jn.copy(out=self._tmp_j2)
        en1 += jn1
        en1 *= 1 / 2 * self._alpha**2 / self._epsilon
        en1 *= -dt
        en1 += en

        # write new coeffs into Propagator.variables
        diffs = self.update_feec_variables(e=en1, j=jn1)

        if self._info:
            logger.info(f"Status     for OhmCold: {info['success']}")
            logger.info(f"Iterations for OhmCold: {info['niter']}")
            logger.info(f"Maxdiff e1 for OhmCold: {diffs['e']}")
            logger.info(f"Maxdiff j1 for OhmCold: {diffs['j']}")
            logger.info("")
