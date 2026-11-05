import logging
from dataclasses import dataclass

from line_profiler import profile

from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import PICVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator
from struphy.propagators.utils import Magnetization1form, TotalFieldMasses1form

logger = logging.getLogger("struphy")


class ShearAlfvenCurlCurlCurrentCoupling5D(ShearAlfvenCurlCurlPropagator):
    r"""Shear-Alfvén step of linear MHD with the magnetic field in :math:`H(\textnormal{curl})`, coupled to the
    magnetization current of drift-kinetic energetic ions: the :math:`H(\textnormal{curl})` counterpart of
    :class:`~struphy.propagators.shear_alfven_current_coupling_5d.ShearAlfvenCurrentCoupling5D`.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` and
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \left[\nabla \times (\tilde{\mathbf B} + \mathbf M) \times \mathbf B\right] \cdot \mathbf V \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x = - \int_\Omega \left[(\nabla \times \mathbf C) \times \mathbf B\right] \cdot \tilde{\mathbf{U}} \,\textrm d \mathbf x \qquad \forall \, \mathbf C \in H(\textrm{curl})\,,

    with the magnetization of the energetic ions (fixed in this step) along the equilibrium field,
    :math:`\mathbf M \in H(\textnormal{curl})`,

    .. math::

        \int_\Omega \mathbf M \cdot \mathbf C \,\textnormal d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}}
        \int_\Omega \iint f^\textnormal{vol} \mu \, \mathbf b_0 \cdot \mathbf C \,\textnormal d \mu \,\textnormal d v_\parallel \,\textnormal d \mathbf x
        \qquad \forall \, \mathbf C \in H(\textrm{curl})\,,

    and :math:`\mathbf B = \mathbf B_0` (``nonlinear=False``) or :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`
    (``nonlinear=True``, taken at the start of the step). Since :math:`\mathbf M` does not change in the step, the
    system is that of :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`
    for :math:`\tilde{\mathbf B} + \mathbf M`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u \in \mathbb R^{N_2}` of
    :math:`\tilde{\mathbf U}_h \in V^2_h`, :math:`\mathbf b \in \mathbb R^{N_1}` of :math:`\tilde{\mathbf B}_h \in V^1_h`
    and the markers :math:`(\boldsymbol \eta_k, \mu_k, w_k)`, :math:`k = 1, \dots, N_p`,

    .. math::

        \mathbb M^2_{\rho_0} \dot{\mathbf u} = - \mathbb M^2_{B} \mathbb C (\mathbf b + \mathbf m)\,, \qquad
        \mathbb M^1 \dot{\mathbf b} = - \mathbb C^\top \mathbb M^2_{B} \mathbf u\,, \qquad
        \mathbb M^1 \mathbf m = (\mathcal P^{b,1})^\top \mathbf a\,, \qquad
        \mathbf a = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{N_p} \sum_{k=1}^{N_p} w_k \mu_k \vec \Lambda^0(\boldsymbol \eta_k)\,,

    with the discrete curl :math:`\mathbb C`, the mass matrix :math:`\mathbb M^1`, the density-weighted mass matrix
    :math:`\mathbb M^2_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`), the antisymmetric mass matrix
    :math:`\mathbb M^2_{B}` of :math:`\int (\mathbf B \times \mathbf W) \cdot \mathbf V \,\textnormal d \mathbf x`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M2B` for ``nonlinear=False``; for ``nonlinear=True`` assembled in
    each step with the weight :math:`[G^{-1} (\hat{\mathbf B}^1_0 + \tilde{\mathbf b}^\top \vec \Lambda^1)]_\times`),
    the basis projection operator :math:`\mathcal P^{b,1}` of the parallel component of a 1-form
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB1form`) and the accumulated vector
    :math:`\mathbf a` (:class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` with the kernel
    ``gc_mag_density_0form``). There is no basis projection of the Lorentz force (no :math:`\hat \Pi^1` error).

    Since :math:`\mathbb M^2_B` is antisymmetric, the semi-discrete system conserves
    :math:`\frac 12 \mathbf u^\top \mathbb M^2_{\rho_0} \mathbf u + \frac 12 \mathbf b^\top \mathbb M^1 \mathbf b + \mathbf a^\top \mathcal P^{b,1} \mathbf b`,
    the perturbed fluid energy plus the energy :math:`\sum_k w_k \mu_k \tilde B_\parallel(\boldsymbol \eta_k) / N_p`
    of the magnetic moments in the parallel perturbation (``en_fB`` of
    :class:`~struphy.models.LinearMHDDriftkineticCC`, up to the constant equilibrium part), since
    :math:`\frac 12 (\mathbf b + \mathbf m)^\top \mathbb M^1 (\mathbf b + \mathbf m) = \frac 12 \mathbf b^\top \mathbb M^1 \mathbf b + \mathbf a^\top \mathcal P^{b,1} \mathbf b + \textnormal{const}`.

    **Time discretization.** The step of
    :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator` (``Options.algo``
    and the constructor argument ``projected``, with the same solvers and preconditioners) for the shifted magnetic
    field :math:`\mathbf b + \mathbf m`, with :math:`\mathbf m` computed once per step (inner ``M1`` solve,
    ``solver_M1`` with the mass matrix preconditioner): with ``algo="implicit"``, ``projected=True``, the conserved
    energy above is conserved exactly (up to the solver tolerances, and without filter of the accumulated vector).
    With ``nonlinear=True``, :math:`\mathbb M^2_B` (and :math:`\mathbb M^2_\perp` of the curl-curl term for
    ``projected=False``, with the weight :math:`(|\mathbf B|^2 G - \hat{\mathbf B}^1 (\hat{\mathbf B}^1)^\top) / (n_0 \sqrt g)`)
    are assembled with the field at the start of the step (the Hiptmair-Xu preconditioner keeps the equilibrium
    field).

    Parameters
    ----------
    energetic_ions : PICVariable, default=None
        Energetic-ion particle variable (``"Particles5D"`` space, not updated) providing the magnetization.
    projected : bool, default=False
        Whether the curl-curl term of the implicit step contains the projection of the Lorentz force (exactly energy
        conserving step), see :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`.
    """

    def __init__(self, energetic_ions: PICVariable = None, projected: bool = False):
        super().__init__(projected=projected)
        self.energetic_ions = energetic_ions

    @dataclass(repr=False)
    class Options(ShearAlfvenCurlCurlPropagator.Options):
        r"""Configuration options for :class:`ShearAlfvenCurlCurlCurrentCoupling5D`: those of
        :class:`ShearAlfvenCurlCurlPropagator.Options <struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator.Options>`
        and

        Parameters
        ----------
        ep_scale : float, default=None
            Factor of the energetic-particle contributions (accumulation): the mass ratio
            :math:`A_\textnormal{h}/A_\textnormal{b}` of the energetic ions and the MHD bulk if ``None``.
        filter_params : FilterParameters, default=None
            Filter of the accumulated vector; defaults to ``FilterParameters()`` (no filter). With a filter the energy
            is no longer conserved exactly (the filter acts on the accumulated vector only).
        nonlinear : bool, default=True
            If ``True``, the Lorentz force uses the total magnetic field :math:`\mathbf B_0 + \tilde{\mathbf B}` (mass
            matrices assembled in each step), else the equilibrium field :math:`\mathbf B_0`.
        solver_params_M1 : SolverParameters, default=None
            Solver controls of the inner ``M1`` solve for the magnetization (pcg with the mass matrix preconditioner);
            defaults to ``SolverParameters(tol=1e-14)``.
        """

        ep_scale: float = None
        filter_params: FilterParameters = None
        nonlinear: bool = True
        solver_params_M1: SolverParameters = None

        def __post_init__(self):
            super().__post_init__()
            assert self.ep_scale is None or isinstance(self.ep_scale, float)
            assert isinstance(self.nonlinear, bool)

            if self.filter_params is None:
                self.filter_params = FilterParameters()

            if self.solver_params_M1 is None:
                self.solver_params_M1 = SolverParameters(tol=1e-14)

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
        assert self.energetic_ions is not None, f"{self.__class__.__name__} needs the energetic ions."

        # factor of the energetic-particle contributions (default: mass ratio A_h/A_b)
        self._ep_scale = self.options.ep_scale
        if self._ep_scale is None:
            self._ep_scale = self.energetic_ions.species.mass_number / self.variables.u.species.mass_number
            logger.info(f"{self.__class__.__name__}: ep_scale = A_h/A_b = {self._ep_scale} (mass ratio, default).")

        # mass matrices of the total field (assembled in each step), before the Schur system of the parent is built
        if self.options.nonlinear:
            self._total_field = TotalFieldMasses1form(
                self.variables.b, self.derham, self.domain, self.mass_ops, self.basis_ops, gyro=not self._projected
            )

        super().allocate()

        # magnetization m = M1^{-1} PB1form^T a
        self._magnetization = Magnetization1form(
            self.energetic_ions.particles,
            self.derham,
            self.domain,
            self.mass_ops,
            self.basis_ops,
            self.options.filter_params,
            self.options.solver_params_M1,
        )
        self._ACC = self._magnetization.accumulator

    def _lorentz_mass(self):
        return self._total_field.M2B if self.options.nonlinear else self.mass_ops.M2B

    def _gyro_mass(self):
        return self._total_field.M2gyro if self.options.nonlinear else self.mass_ops.M2gyro

    @profile
    def __call__(self, dt):
        # mass matrices of the field at the start of the step
        if self.options.nonlinear:
            self._total_field.update()

        # magnetization of the (fixed) energetic ions: m = M1^{-1} PB1form^T a
        m = self._m = self._magnetization(self._ep_scale)

        # step of the uncoupled curl-curl propagator for the shifted field b + m (m is constant in the step)
        b = self.variables.b.spline.vector
        b += m
        b.update_ghost_regions()
        super().__call__(dt)
        b = self.variables.b.spline.vector
        b -= m
        b.update_ghost_regions()

        # update_weights
        if self.energetic_ions.species.weights_params.control_variate:
            self.energetic_ions.particles.update_weights()
