from dataclasses import dataclass
from typing import Callable, Literal

from feectools.linalg.stencil import StencilVector

from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.utils.utils import check_option


class AdiabaticPoissonSolve(ImplicitDiffusion):
    r"""
    Weak discretization of the Poisson equation for full-orbit ions with adiabatic electrons.

    Find :math:`\phi \in H^1` such that

    .. math::

        \frac{1}{Z\varepsilon^2} \int_\Omega \frac{n_0}{T_0}\, \psi\, (\phi - \langle \phi \rangle)\,\textrm d \mathbf x
        + \frac{1}{\alpha^2} \int_\Omega \nabla \psi \cdot \nabla \phi \,\textrm d \mathbf x
        = \frac{1}{\varepsilon} \int_\Omega \psi\, (n_i - n_0)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`n_i = \int f \,\textrm d^3 \mathbf v` is the ion density, :math:`n_0` and :math:`T_0` are the
    equilibrium ion density and electron temperature, :math:`Z` is the ion charge number,
    :math:`\varepsilon = 1/(\hat \Omega_c \hat t)` and :math:`\alpha = \hat \omega_p / \hat \Omega_c`.
    The first term is the linearized Boltzmann response of the electrons. The optional flux-surface average
    :math:`\langle \phi \rangle` (``which_geometry``) is the zonal correction of
    :class:`~struphy.propagators.gyrokinetic_poisson_solve.GyrokineticPoissonSolve`: electrons do not respond
    to the averaged potential. The second term is the vacuum (Debye) term; it is dropped in the quasi-neutral
    limit :math:`\alpha \to \infty` (``quasi_neutral=True``), where the equation becomes a weighted :math:`L^2`
    projection. Boundary terms from integration by parts are assumed to vanish.

    With ``polarization=True`` (quasi-neutral only), the polarization density of guiding centers is added as in
    :class:`~struphy.propagators.gyrokinetic_poisson_solve.GyrokineticPoissonSolve`,

    .. math::

        \frac{1}{Z\varepsilon^2} \int_\Omega \frac{n_0}{T_0}\, \psi\, (\phi - \langle \phi \rangle)\,\textrm d \mathbf x
        + \int_\Omega \frac{n_0}{|\mathbf B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi \,\textrm d \mathbf x
        = \frac{1}{\varepsilon} \int_\Omega \psi\, (n_i - n_0)\,\textrm d \mathbf x\,,

    discretized with the gyrokinetic mass matrix :math:`\mathbb M^1_\textnormal{gyro}` in place of
    :math:`\alpha^{-2} \mathbb M^1`. This is needed when :math:`n_i` is the density of the marker positions and
    these are guiding-center positions whose density does not contain the polarization density
    (e.g. classical Pauli particles with the control variate, see
    :class:`~struphy.models.VlasovPoissonAdiabaticPauli`).

    In the quasi-neutral limit the ion-acoustic branch :math:`\omega = k c_s` has no Debye cutoff, so an explicit
    particle push needs :math:`\Delta t\, c_s k_\textnormal{max} \lesssim 2`, with :math:`k_\textnormal{max} \approx \pi / h`
    for the smallest cell size :math:`h`; refining the grid requires a smaller time step.

    The equation is discretized as (multiplied by :math:`\alpha^2` if ``quasi_neutral=False``)

    .. math::

        \left( \frac{1}{Z\varepsilon^2}\,\mathbb M^0_{\textnormal{ad},T} (\mathbb I - \mathbb A)
        + \frac{1}{\alpha^2} \mathbb G^\top \mathbb M^1 \mathbb G \right)\, \boldsymbol\phi
        = \frac{1}{\varepsilon} (\Lambda^0, n_i - n_0 )_{L^2}\,,

    where :math:`\mathbb M^0_{\textnormal{ad},T}` is the mass matrix weighted with :math:`n_0/T_0`
    (``stab_mat="M0ad_withT"``) and :math:`\mathbb A` is the weighted average projection
    (:math:`\mathbb A = 0` if ``which_geometry=None``).
    The particle density is deposited with the marker weights; the background :math:`n_0` is subtracted
    either by the control variate or, for full-f markers, by an explicit background source
    (see :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion`).
    """

    def __init__(
        self,
        rho: FEECVariable | Callable | ParticlesToGrid | list = None,
        rho_coeffs: float | list = None,
        epsilon: float = 1.0,
        alpha: float = 1.0,
        Z: int = 1,
        diagnostic: FEECVariable | None = None,
    ):
        """
        Parameters
        ----------
        see ImplicitDiffusion.__init__ docstring for other parameters.

        epsilon : float
            Inverse normalized cyclotron frequency :math:`1/(\\hat \\Omega_c \\hat t)`.

        alpha : float
            Ratio of plasma to cyclotron frequency :math:`\\hat \\omega_p / \\hat \\Omega_c`
            (only used if ``quasi_neutral=False``).

        Z : int
            Charge number of the ions.
        """
        super().__init__(rho=rho, rho_coeffs=rho_coeffs, diagnostic=diagnostic)
        self.epsilon = epsilon
        self.alpha = alpha
        self.Z = Z

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`AdiabaticPoissonSolve`.

        Parameters
        ----------
        quasi_neutral : bool, default=True
            If ``True``, drop the vacuum term :math:`\alpha^{-2}\Delta \phi` (limit :math:`\alpha \to \infty`);
            the system is then a weighted mass-matrix solve (unless ``polarization=True``).

        polarization : bool, default=False
            If ``True``, add the polarization term :math:`-\nabla \cdot (n_0 |\mathbf B_0|^{-2} \nabla_\perp \phi)`
            (diffusion matrix ``"M1gyro"``). Requires ``quasi_neutral=True``.

        which_geometry : {None, "cylindrical", "toroidal"}, default=None
            Zonal correction :math:`\phi - \langle \phi \rangle` in the adiabatic electron response:
            no average (``None``), average over ``eta3`` (``"cylindrical"``) or over ``eta2`` and ``eta3``
            (``"toroidal"``), as in
            :class:`~struphy.propagators.gyrokinetic_poisson_solve.GyrokineticPoissonSolve`.
            Requires ``quasi_neutral=False`` or ``polarization=True``: without a diffusion term the averaged potential is
            undetermined.

        stab_mat : {"M0", "M0ad", "M0ad_withT", "Id"}, default="M0ad_withT"
            Mass matrix of the adiabatic term (``"M0ad_withT"`` has the weight :math:`n_0/T_0`).

        x0 : StencilVector, default=None
            Initial guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver.

        precond : LiteralOptions.OptsDiffusionPrecond, default=None
            Name of the preconditioner, see :class:`ImplicitDiffusion`. ``"MultiGrid"`` and
            ``"StiffnessPreconditioner"`` need ``quasi_neutral=False``.

        precond_params : dict, default=None
            Keyword arguments passed to the constructor of the preconditioner, see :class:`ImplicitDiffusion`.

        multigrid : MultiGridOptions, default=None
            Options of the multigrid preconditioner (if ``precond="MultiGrid"``).

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.

        filter_params : dict[PICVariable | SPHVariable, FilterParameters], default=None
            If not None, specifies a filter to the accumulation of a specific variable.

        Notes
        -----
        ``AdiabaticPoissonSolve`` reuses :class:`ImplicitDiffusion` with ``sigma_2 = 0``, ``divide_by_dt = False``,
        ``sigma_1 = 1 / (Z * epsilon**2)``, ``sigma_3 = 1 / epsilon`` and ``diffusion_mat=None`` (quasi-neutral),
        or ``sigma_1 = alpha**2 / (Z * epsilon**2)``, ``sigma_3 = alpha**2 / epsilon`` and ``diffusion_mat="M1"``.
        """

        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "M0ad_withT", "Id"]
        OptsGeometry = Literal["cylindrical", "toroidal", None]
        # propagator options
        quasi_neutral: bool = True
        polarization: bool = False
        which_geometry: OptsGeometry = None
        stab_mat: OptsStabMat = "M0ad_withT"
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
            if self.polarization:
                assert self.quasi_neutral, "polarization=True needs quasi_neutral=True (one diffusion matrix)."
            if self.quasi_neutral and not self.polarization:
                assert self.which_geometry is None, (
                    "The zonal correction (which_geometry) needs quasi_neutral=False or polarization=True: "
                    "without the vacuum term the averaged potential is undetermined."
                )
                assert self.precond not in ("MultiGrid", "StiffnessPreconditioner"), (
                    f"precond={self.precond!r} needs quasi_neutral=False or polarization=True."
                )

            # defaults
            if self.precond_params is None:
                self.precond_params = {}
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.multigrid is None:
                self.multigrid = MultiGridOptions()

        @property
        def stab_average(self) -> str | None:
            """Directions averaged in the adiabatic term, see ``which_geometry``."""
            return {None: None, "cylindrical": "eta3", "toroidal": "eta2 eta3"}[self.which_geometry]

        @property
        def diffusion_mat(self) -> str | None:
            """Diffusion matrix of :class:`ImplicitDiffusion`: none in the quasi-neutral limit, ``"M1gyro"`` with
            the polarization term."""
            if self.polarization:
                return "M1gyro"
            return None if self.quasi_neutral else "M1"

    def allocate(self):
        scale = 1.0 if self.options.quasi_neutral else self.alpha**2
        self.options.sigma_1 = scale / (self.Z * self.epsilon**2)
        self.options.sigma_2 = 0.0
        self.options.sigma_3 = scale / self.epsilon
        self.options.divide_by_dt = False
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
