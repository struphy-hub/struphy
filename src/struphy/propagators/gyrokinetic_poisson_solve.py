from dataclasses import dataclass
from typing import Callable, Literal

from feectools.linalg.stencil import StencilVector

from struphy.io.options import LiteralOptions
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.utils.utils import check_option


class GyrokineticPoissonSolve(ImplicitDiffusion):
    r"""
    Weak discretization of the Poisson equation, with adiabatic response of electrons.

    Find :math:`\phi \in H^1` such that

    .. math::

        \frac{1}{Z\epsilon^2} \int_\Omega \frac{n_0}{T_0} \psi\, (\phi - \langle \phi \rangle)\,\textrm d \mathbf x + \int_\Omega \frac{n_0}{|B_0|^2} \nabla \psi^\top \,\mathbb D \nabla \phi \,\textrm d \mathbf x = \frac{1}{\epsilon} \sum_i c_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`c_i` are the ``rho_coeffs``, :math:`\mathbb D` can be an anisotropic (gyrokinetic) diffusion operator
    (or the identity; the weight :math:`n_0/|B_0|^2` is that of ``diffusion_mat="M1gyro"``, ``"M1"`` and ``"M1perp"``
    are unweighted),
    :math:`\epsilon \in \mathbb R` is the gyrokinetic ratio defined in units, Z the charge number of ions and
    :math:`\langle \phi \rangle` the (weighted) average over :math:`\eta_3` or over :math:`\eta_2, \eta_3`
    (``which_geometry``): the electrons do not respond to the flux-surface averaged potential.
    For a full-f particle source (no control variate) the neutralizing background :math:`-n_0` is added to
    :math:`\rho_i` automatically (see :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion`).
    Boundary terms from integration by parts are assumed to vanish.

    The equation is discretized as

    .. math::

        \left( \frac{1}{Z\epsilon^2}\,\mathbb M^0_{\textnormal{ad},T} (\mathbb I - \mathbb A) + \mathbb G^\top \mathbb M^1_\textnormal{gyro} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \frac{1}{\epsilon} \sum_i c_i (\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`\mathbb M^0_{\textnormal{ad},T}` is the mass matrix weighted with :math:`n_0/T_0` (``stab_mat``,
    default ``"M0ad_withT"``), :math:`\mathbb A` the weighted average projection and
    :math:`\mathbb M^1_\textnormal{gyro}` the gyrokinetic mass matrix (``diffusion_mat``).

    Parameters
    ----------
    rho : FEECVariable or Callable or ParticlesToGrid or list
        (List of) right-hand side source term(s) of the Poisson problem (optional, can be set with
        a setter later). See :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion` for
        the accepted entries; particle sources are passed as
        :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`.

    rho_coeffs : float or list
        Multiplicative coefficient(s) applied to ``rho`` (default 1.0 for all sources).

    epsilon : float
        Gyrokinetic parameter.

    Z : int
        Charge number of the ions.

    diagnostic : FEECVariable, optional
        If given, the L2 projection of the right-hand side is stored in it.
    """

    def __init__(
        self,
        rho: FEECVariable | Callable | ParticlesToGrid | list = None,
        rho_coeffs: float | list = None,
        epsilon: float = 1.0,
        Z: int = 1,
        diagnostic: FEECVariable | None = None,
    ):
        """
        Parameters
        ----------
        see ImplicitDiffusion.__init__ docstring for other parameters.

        epsilon : float
            Gyrokinetic parameter, appears in gyrokinetic Poisson equation

        Z : int
            Charge number for ions, appears in gyrokinetic Poisson equation
        """
        super().__init__(rho=rho, rho_coeffs=rho_coeffs, diagnostic=diagnostic)
        self.epsilon = epsilon
        self.Z = Z

    @dataclass
    class Options:
        r"""Configuration options for :class:`GyrokineticPoissonSolve`.

        Parameters
        ----------
        stab_mat : {"M0", "M0ad", "Id", "M0ad_withT"}, default="M0ad_withT"
            Mass matrix of the adiabatic term, multiplied by ``1 / (Z * epsilon**2)``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator (weight :math:`n_0`).
            - ``"Id"``: identity operator.
            - ``"M0ad_withT"``: adiabatic-electron weighted 0-form mass operator (weight :math:`n_0/T_0`).

        which_geometry: {"cylindrical", "toroidal"}, default="cylindrical"
            Geometry of the problem, determines the meaning of `<\phi>` in the stabilization term:
            the average over ``eta3`` (cylindrical) or over ``eta2`` and ``eta3`` (toroidal), i.e.
            ``stab_average`` of :class:`ImplicitDiffusion` is ``"eta3"`` or ``"eta2 eta3"``. The average
            is weighted with ``stab_mat`` (for equilibrium density and temperature constant on the
            averaged surfaces, the flux-surface average with the volume element), which keeps the
            operator symmetric (:class:`~struphy.feec.mass.WeightedAverageProjection`).

        diffusion_mat : {"M1", "M1perp", "M1gyro"}, default="M1gyro"
            Diffusion matrix.

        x0 : StencilVector, default=None
            Initial guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsDiffusionPrecond, default=None
            Name of the preconditioner, see :class:`ImplicitDiffusion`. Requires ``solver="pcg"``.
            ``"StiffnessPreconditioner"`` approximates the operator with ``diffusion_mat``, ``stab_mat``
            and the average; ``"MultiGrid"`` is the geometric multigrid V-cycle, with the average
            re-built on every coarse level.

        precond_params : dict, default=None
            Keyword arguments passed to the constructor of the mass-matrix or stiffness
            preconditioner, see :class:`ImplicitDiffusion`.

        multigrid : MultiGridOptions, default=None
            Options of the multigrid preconditioner (if ``precond="MultiGrid"``).
            If ``None``, defaults to ``MultiGridOptions()``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        filter_params : dict[PICVariable | SPHVariable, FilterParameters], default=None
            If not None, specifies a filter to the accumulation of a specific variable.

        Notes
        -----
        ``GyrokineticPoissonSolve`` reuses :class:`ImplicitDiffusion` internals by
        enforcing
        ``sigma_1 = 1 / (Z * epsilon**2)``, ``sigma_2 = 0.0``, ``sigma_3 = 1 / epsilon`` and
        ``divide_by_dt = False`` in ``allocate``.
        """

        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id", "M0ad_withT"]
        OptsDiffusionMat = Literal["M1", "M1perp", "M1gyro"]
        OptsGeometry = Literal["cylindrical", "toroidal"]
        # propagator options
        stab_mat: OptsStabMat = "M0ad_withT"
        diffusion_mat: OptsDiffusionMat = "M1gyro"
        which_geometry: OptsGeometry = "cylindrical"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsDiffusionPrecond = None
        precond_params: dict = None
        multigrid: MultiGridOptions = None
        solver_params: SolverParameters = None
        filter_params: dict[PICVariable | SPHVariable, FilterParameters] = None
        # the adiabatic term is physical, not a stabilization: the rhs is not projected
        enforce_compatibility: bool = False

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsDiffusionPrecond)
            check_option(self.which_geometry, self.OptsGeometry)
            if self.precond is not None:
                assert self.solver == "pcg", f"precond={self.precond!r} requires solver='pcg'."

            # defaults
            if self.precond_params is None:
                self.precond_params = {}
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.multigrid is None:
                self.multigrid = MultiGridOptions()

        @property
        def stab_average(self) -> str:
            """Directions averaged in the stabilization, see ``which_geometry``."""
            return {"cylindrical": "eta3", "toroidal": "eta2 eta3"}[self.which_geometry]

    def allocate(self):
        epsilon = self.epsilon
        Z = self.Z
        self.options.sigma_1 = 1 / epsilon**2 / self.Z
        self.options.sigma_2 = 0.0
        self.options.sigma_3 = 1 / epsilon
        self.options.divide_by_dt = False
        # the stabilization stab_mat @ (I - A) (A from stab_average) is built in ImplicitDiffusion
        super().allocate()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
