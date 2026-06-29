import logging
from dataclasses import dataclass
from typing import Callable, Literal

from feectools.linalg.stencil import StencilVector

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class PoissonSolve(ImplicitDiffusion):
    r"""
    Weak discretization of the (stabilized) Poisson equation: find :math:`\phi \in H^1` such that

    .. math::

        \epsilon \int_\Omega \psi\, \phi\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top \, \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`\epsilon \in \mathbb R` is a stabilization parameter. Boundary terms from integration by parts are assumed to vanish. The equation is discretized as

    .. math::

        \left( \epsilon\,\mathbb S + \mathbb G^\top \mathbb M^1 \mathbb G \right)\, \boldsymbol{\phi} = \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`\mathbb M^1` is the :math:`H(\textnormal{curl})`-mass matrix and :math:`\mathbb S` is a stabilization matrix.
    """

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`Poisson`.

        Parameters
        ----------
        stab_eps : float, default=0.0
            Stabilization weight used for the mass-like term in the Poisson
            operator.
            Internally mapped to ``sigma_1 = stab_eps`` in the parent
            :class:`ImplicitDiffusion` formulation.

        stab_mat : {"M0", "M0ad", "Id"}, default="Id"
            Stabilization matrix multiplied by ``stab_eps``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator.
            - ``"Id"``: identity operator.

        diffusion_mat : {"M1", "M1perp", "M1gyro"}, defaults="M1"
            Diffusion matrix.

            - ``"M1"``: standard weighted 1-form mass operator.
            - ``"M1perp"``: weighted 1-form mass operator perpendicular to magnetic field.
            - ``"M1para"``: weighted 1-form mass operator parallele to magnetic field.
            - ``"M1gyro"``: weighted 1-form mass operator used in gyrokinetic model.

        x0 : StencilVector, default=None
            Initial guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Name of the preconditioner configuration.
            Currently this class inherits the same behavior as
            :class:`ImplicitDiffusion`, where ``pc=None`` is used internally.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        Notes
        -----
        ``Poisson.Options`` reuses :class:`ImplicitDiffusion` internals by
        enforcing
        ``sigma_2 = 0.0``, ``sigma_3 = 1.0``, ``divide_by_dt = False`` and
        ``diffusion_mat = "M1"`` in ``__post_init__``.
        """

        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id"]
        OptsDiffusionMat = Literal["M1", "M1perp", "M1para", "M1gyro"]
        # propagator options
        stab_eps: float = 0.0
        stab_mat: OptsStabMat = "Id"
        diffusion_mat: OptsDiffusionMat = "M1"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.diffusion_mat, self.OptsDiffusionMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            # Poisson solve (-> set some params of parent class)
            self.sigma_1 = self.stab_eps
            self.sigma_2 = 0.0
            self.sigma_3 = 1.0
            self.divide_by_dt = False

    def __init__(
        self,
        rho: FEECVariable | Callable | list = None,
        rho_coeffs: float | list = None,
    ):
        super().__init__(rho=rho, rho_coeffs=rho_coeffs)

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
