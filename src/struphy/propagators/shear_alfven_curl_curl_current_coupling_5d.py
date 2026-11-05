import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import PICVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.kernels.gc_mag_density_0form import gc_mag_density_0form
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator

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
    ``solver_params_M1`` with the mass matrix preconditioner): with ``algo="implicit"``, ``projected=True``, the conserved
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
        nonlinear : bool, default=False
            If ``True``, the Lorentz force uses the total magnetic field :math:`\mathbf B_0 + \tilde{\mathbf B}` (mass
            matrices assembled in each step, which costs about as much as the curl-curl solve itself), else the
            equilibrium field :math:`\mathbf B_0` (default; as the shear-Alfvén step of
            :class:`~struphy.models.LinearMHD`).
        solver_params_M1 : SolverParameters, default=None
            Solver controls of the inner ``M1`` solve for the magnetization (pcg with the mass matrix preconditioner);
            defaults to ``SolverParameters(tol=1e-14)``.
        """

        ep_scale: float = None
        filter_params: FilterParameters = None
        nonlinear: bool = False
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
            self._total_field = _TotalFieldMasses1form(
                self.variables.b, self.derham, self.domain, self.mass_ops, self.basis_ops, gyro=not self._projected
            )

        super().allocate()

        # magnetization m = M1^{-1} PB1form^T a
        self._magnetization = _Magnetization1form(
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
        m = self._magnetization(self._ep_scale)

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


class _Magnetization1form:
    r"""Magnetization of drift-kinetic energetic ions along the equilibrium field as a 1-form (in :math:`V^1_h`),

    .. math::

        \mathbb M^1 \mathbf m = (\mathcal P^{b,1})^\top \mathbf a\,, \qquad
        \mathbf a = s \, \frac{1}{N_p} \sum_{k=1}^{N_p} w_k \mu_k \vec \Lambda^0(\boldsymbol \eta_k)\,,

    with the basis projection operator :math:`\mathcal P^{b,1}`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.PB1form`) of the parallel component of a
    1-form and the accumulated vector :math:`\mathbf a` (kernel ``gc_mag_density_0form``, factor :math:`s`), i.e. the
    :math:`L^2` projection of :math:`\int f \mu \, \mathbf b_0 \,\textnormal d \mu\, \textnormal d v_\parallel` onto
    :math:`V^1_h` (pairing: :math:`\mathbf a^\top \mathcal P^{b,1} \mathbf b` is the energy of the magnetic moments in
    the parallel perturbation).

    Parameters
    ----------
    particles : Particles5D
        The energetic-ion markers.
    derham, domain, mass_ops, basis_ops
        Discrete de Rham sequence, mapping, weighted mass and basis projection operators.
    filter_params : FilterParameters
        Filter of the accumulated vector.
    solver_params : SolverParameters
        Controls of the ``M1`` solve (pcg with the mass matrix preconditioner).
    """

    def __init__(self, particles, derham, domain, mass_ops, basis_ops, filter_params, solver_params):
        self._acc = AccumulatorVector(
            particles, "H1", gc_mag_density_0form, mass_ops, domain.args_domain, filter_params=filter_params
        )
        self._PB1T = basis_ops.PB1form.T
        M1 = mass_ops.M1
        if derham.polar_splines:
            pc = preconditioner.PolarMassMatrixPreconditioner(M1)
        else:
            pc = preconditioner.MassMatrixPreconditioner(M1)
        self._M1_inv = inverse(
            M1,
            "pcg",
            pc=pc,
            tol=solver_params.tol,
            maxiter=solver_params.maxiter,
            verbose=solver_params.verbose,
            recycle=True,
        )
        self._rhs = M1.codomain.zeros()
        self._m = M1.domain.zeros()

    @property
    def accumulator(self):
        """The :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` of :math:`\\mathbf a`."""
        return self._acc

    def __call__(self, scale: float):
        """Accumulate :math:`\\mathbf a` with the factor ``scale`` and return :math:`\\mathbf m` (the same vector in
        each call; collective)."""
        self._acc(scale)
        self._PB1T.dot(self._acc.vectors[0], out=self._rhs)
        return self._M1_inv.dot(self._rhs, out=self._m)


class _TotalFieldMasses1form:
    r"""Mass matrices of the Lorentz force with the total magnetic field
    :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` (1-form components
    :math:`\hat{\mathbf B}^1 = \hat{\mathbf B}^1_0 + \tilde{\mathbf b}^\top \vec \Lambda^1`, the equilibrium evaluated
    exactly), assembled by :meth:`update`:

    * :attr:`M2B`, as :attr:`~struphy.feec.mass.WeightedMassOperators.M2B` of :math:`\int (\mathbf B \times \mathbf W)
      \cdot \mathbf V \,\textnormal d \mathbf x` on 2-forms, with the weight
      :math:`\textnormal{sign}(\det DF) \, [G^{-1} \hat{\mathbf B}^1]_\times` (:math:`\hat{\mathbf B}^2 / |\det DF|`);
    * :attr:`M2gyro` (if ``gyro``), as :attr:`~struphy.feec.mass.WeightedMassOperators.M2gyro` of
      :math:`\int \frac{1}{n_0} (\mathbf B \times \mathbf W) \cdot (\mathbf B \times \mathbf V)
      \,\textnormal d \mathbf x`,
      with the weight :math:`(|\mathbf B|^2 G - \hat{\mathbf B}^1 (\hat{\mathbf B}^1)^\top) / (n_0 \sqrt g)`.

    Parameters
    ----------
    b : FEECVariable
        The magnetic field perturbation (``"Hcurl"`` space).
    derham, domain, mass_ops, basis_ops
        Discrete de Rham sequence, mapping, weighted mass and basis projection operators (with the MHD equilibrium).
    gyro : bool
        Whether to assemble :attr:`M2gyro` too.
    """

    def __init__(self, b, derham, domain, mass_ops, basis_ops, gyro: bool = False):
        assert b.space == "Hcurl", "The magnetic field must be in Hcurl."
        self._b = b
        pts = [points.flatten() for points in derham.spline_attributes["Hdiv"].quad_grid_pts[0]]
        self._spans, bn, bd = derham.prepare_eval_tp_fixed(pts)
        self._bases_1form = [[bd[0], bn[1], bn[2]], [bn[0], bd[1], bn[2]], [bn[0], bn[1], bd[2]]]

        eq = basis_ops.weights["eq_mhd"]
        PTS = xp.meshgrid(*pts, indexing="ij")
        self._b0_1 = [fun(*PTS) for fun in (eq.b1_1, eq.b1_2, eq.b1_3)]
        g_inv = domain.metric_inv(*pts)
        g = domain.metric(*pts)
        self._g_inv = [[g_inv[a, c] for c in range(3)] for a in range(3)]
        self._g = [[g[a, c] for c in range(3)] for a in range(3)]
        det_df = domain.jacobian_det(*pts)
        # M2B has the weight R(B^2)/|det DF| = sign(det DF) R(G^{-1} B^1) (1/sqrt_g is 1/|det DF|)
        self._sign_det = xp.sign(det_df)
        self._gyro_scale = 1.0 / (eq.n0(*PTS) * abs(det_df))

        self._bf = derham.create_spline_function("b_tot", "Hcurl")
        self.M2B = mass_ops.create_weighted_mass("Hdiv", "Hdiv", name="M2B_tot")
        self.M2gyro = mass_ops.create_weighted_mass("Hdiv", "Hdiv", name="M2gyro_tot") if gyro else None
        self.update()

    def update(self):
        """Assemble the mass matrices with the current magnetic field (collective)."""
        self._b.spline.vector.copy(out=self._bf.vector)
        b_v = self._bf.eval_tp_fixed_loc(self._spans, self._bases_1form)
        b1 = [self._b0_1[a] + b_v[a] for a in range(3)]
        c = [sum(self._g_inv[a][e] * b1[e] for e in range(3)) for a in range(3)]
        b_sq = sum(b1[a] * c[a] for a in range(3))
        c = [self._sign_det * ca for ca in c]
        self.M2B.assemble([[None, -c[2], c[1]], [c[2], None, -c[0]], [-c[1], c[0], None]])
        if self.M2gyro is not None:
            self.M2gyro.assemble(
                [[(b_sq * self._g[a][e] - b1[a] * b1[e]) * self._gyro_scale for e in range(3)] for a in range(3)]
            )
