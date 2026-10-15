import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ScalarWaveLeapfrog(Propagator):
    r"""FEEC discretization of the scalar (0-form) wave equation
    :math:`\partial_t^2 \phi - c^2 \Delta \phi = 0` as the first-order system: find
    :math:`\phi, \psi \in H^1` such that

    .. math::

        \frac{\partial \phi}{\partial t} = \psi\,, \qquad
        \int_\Omega \frac{\partial \psi}{\partial t}\, \chi \,\textrm d \mathbf x
        + c^2 \int_\Omega \nabla \chi \cdot \nabla \phi \,\textrm d \mathbf x = 0 \qquad \forall \,\chi \in H^1\,.

    Time discretization: symplectic velocity Verlet (the staggered leapfrog of the polar wave studies of
    R. Remmerswaal), with :math:`\mathbb K = c^2\,\mathbb G^\top \mathbb M^1 \mathbb G`,

    .. math::

        \boldsymbol \psi^{n+1/2} = \boldsymbol \psi^n - \tfrac{\Delta t}{2} \mathbb M^{0,-1} \mathbb K \boldsymbol \phi^n\,,\quad
        \boldsymbol \phi^{n+1} = \boldsymbol \phi^n + \Delta t\, \boldsymbol \psi^{n+1/2}\,,\quad
        \boldsymbol \psi^{n+1} = \boldsymbol \psi^{n+1/2} - \tfrac{\Delta t}{2} \mathbb M^{0,-1} \mathbb K \boldsymbol \phi^{n+1}\,,

    which conserves a discrete energy and is stable for :math:`\Delta t < 2/\sqrt{\lambda_\textnormal{max}}`,
    :math:`\lambda_\textnormal{max}` the largest eigenvalue of :math:`\mathbb M^{0,-1}\mathbb K`.
    """

    class Variables:
        """Container for the wave variables.

        Attributes
        ----------
        phi : FEECVariable
            The wave field in H1.
        psi : FEECVariable
            Its time derivative in H1.
        """

        def __init__(self):
            self._phi: FEECVariable = None
            self._psi: FEECVariable = None

        @property
        def phi(self) -> FEECVariable:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._phi = new

        @property
        def psi(self) -> FEECVariable:
            return self._psi

        @psi.setter
        def psi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._psi = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Options of :class:`ScalarWaveLeapfrog`.

        Parameters
        ----------
        c : float
            Wave speed.
        solver : LiteralOptions.OptsSymmSolver
            Solver of the mass matrix.
        precond : LiteralOptions.OptsMassPrecond
            Preconditioner of the mass matrix.
        solver_params : SolverParameters
            Solver controls.
        """

        c: float = 1.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
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
        M0, M1 = self.mass_ops.M0, self.mass_ops.M1
        grad = self.derham.grad
        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(M0)
        M0_inv = inverse(
            M0,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )
        # psi' = -M0^{-1} K phi
        self._acceleration = -(self.options.c**2) * (M0_inv @ grad.T @ M1 @ grad)
        space = self.variables.phi.spline.vector.space
        self._psi_half = space.zeros()
        self._phi_new = space.zeros()
        self._psi_new = space.zeros()

    @profile
    def __call__(self, dt):
        phi = self.variables.phi.spline.vector
        psi = self.variables.psi.spline.vector

        psi_half = self._acceleration.dot(phi, out=self._psi_half)
        psi_half *= dt / 2
        psi_half += psi

        phi_new = psi_half.copy(out=self._phi_new)
        phi_new *= dt
        phi_new += phi

        psi_new = self._acceleration.dot(phi_new, out=self._psi_new)
        psi_new *= dt / 2
        psi_new += psi_half

        self.update_feec_variables(phi=phi_new, psi=psi_new)
