import logging
from dataclasses import dataclass
from typing import Callable, Literal

from feectools.linalg.stencil import StencilVector

from struphy.feec.preconditioner import _parse_directions
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
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

        stab_mat : {"M0", "M0ad", "Id"}, default="M0"
            Stabilization matrix multiplied by ``stab_eps``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator.
            - ``"Id"``: identity operator.

        stab_average : str, default=None
            If not None, the stabilization becomes ``stab_mat @ (I - A)``, where ``A`` is the
            ``stab_mat``-weighted average over the given logical directions, e.g. ``"eta3"`` or
            ``"eta2 eta3"``, see :class:`ImplicitDiffusion`.

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

        precond : LiteralOptions.OptsDiffusionPrecond, default=None
            Name of the preconditioner, see :class:`ImplicitDiffusion`
            (``"MultiGrid"`` for geometric multigrid, ``"StiffnessPreconditioner"`` for the
            Kronecker approximation of the Laplacian, ``"MassMatrixPreconditioner"`` for the
            Kronecker approximation of ``M0``). Requires ``solver="pcg"``.

        precond_params : dict, default=None
            Keyword arguments passed to the constructor of the mass-matrix or stiffness
            preconditioner, see :class:`ImplicitDiffusion`.

        multigrid : MultiGridOptions, default=None
            Options of the multigrid preconditioner (if ``precond="MultiGrid"``).
            For periodic or Neumann boundary conditions with ``stab_eps = 0``, use
            ``MultiGridOptions(nullspace="constants")``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        filter_params : dict[PICVariable | SPHVariable, FilterParameters], default=None
            If not None, specifies a filter to the accumulation of a specific variable.

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
        stab_mat: OptsStabMat = "M0"
        stab_average: str = None
        diffusion_mat: OptsDiffusionMat = "M1"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsDiffusionPrecond = None
        precond_params: dict = None
        multigrid: MultiGridOptions = None
        solver_params: SolverParameters = None
        filter_params: dict[PICVariable | SPHVariable, FilterParameters] = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.diffusion_mat, self.OptsDiffusionMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsDiffusionPrecond)
            if self.precond is not None:
                assert self.solver == "pcg", f"precond={self.precond!r} requires solver='pcg'."
            _parse_directions(self.stab_average)
            if self.stab_average and self.precond == "MultiGrid":
                raise ValueError("precond='MultiGrid' does not support stab_average (no coarsening of the average).")

            # defaults
            if self.precond_params is None:
                self.precond_params = {}
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.multigrid is None:
                self.multigrid = MultiGridOptions()

            # Poisson solve (-> set some params of parent class)
            self.sigma_1 = self.stab_eps
            self.sigma_2 = 0.0
            self.sigma_3 = 1.0
            self.divide_by_dt = False

    def __init__(
        self,
        rho: FEECVariable | Callable | ParticlesToGrid | list = None,
        rho_coeffs: float | list = None,
    ):
        """
        Parameters
        ----------
        rho : FEECVariable or Callable or ParticlesToGrid or list, default=None
            Right-hand side source term(s) of the Poisson problem.
            Accepted entries are:

            - ``None``: zero source.
            - ``FEECVariable`` in ``H1``.
            - ``Callable`` to be projected to ``H1`` via ``L2Projector``.
            - :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`, describing a
              particle-to-grid (charge/current) deposition.
            - a ``list`` containing any mix of the entries above.

        rho_coeffs : float or list, default=None
            Multiplicative coefficient(s) applied to ``rho``.
            If ``None``, coefficients default to ``1.0`` for all sources.
        """
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
