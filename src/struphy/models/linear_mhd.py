from struphy.io.options import BaseUnits
from struphy.models.linear_euler import LinearEuler
from struphy.models.linear_euler_mixins import MagneticFieldMixin, add_magnetic_propagators
from struphy.models.variables import FEECVariable


class LinearMHD(MagneticFieldMixin, LinearEuler):
    """Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

    Extends :class:`~struphy.models.LinearEuler` by the magnetic field: the Lorentz force and the induction equation.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    b_space: {"Hdiv", "Hcurl"}
        Space of the magnetic field (default: "Hdiv"). ``"Hdiv"``: strong induction equation, implicit shear-Alfvén
        solve for the velocity (:class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`).
        ``"Hcurl"``: weak induction equation, implicit curl-curl solve for the magnetic field
        (:class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`).
    projected: bool
        Only with ``b_space="Hcurl"``: the curl-curl term of the implicit shear-Alfvén step contains the projection of
        the Lorentz force, which makes this step exactly energy conserving at the price of inner mass matrix solves
        (default: False).
    """

    ## additional propagators

    class Propagators(LinearEuler.Propagators):
        """Propagators of :class:`~struphy.models.LinearEuler` plus ``shear_alf`` (shear-Alfvén waves) and
        ``bg_current`` (force of the background current).

        Parameters
        ----------
        b_field : FEECVariable, optional
            Magnetic field perturbation, used by ``bg_current`` (default: None).
        b_space : str
            Space of the magnetic field, ``"Hdiv"`` or ``"Hcurl"`` (default: ``"Hdiv"``).
        projected : bool
            Projected curl-curl shear-Alfvén step, only with ``b_space="Hcurl"`` (default: False).
        **kwargs
            Passed to :class:`LinearEuler.Propagators <struphy.models.LinearEuler.Propagators>`.
        """

        splitting_order = ("shear_alf", "bg_current", "bg_pressure", "sound")
        # the background steps are exact in time but do not commute with the sound wave step: symmetric splitting
        # of the three keeps the compressional part second order in time (as the former unsplit Magnetosonic step)
        strang_groups = (("bg_current", "bg_pressure", "sound"),)

        def __init__(
            self,
            b_field: FEECVariable = None,
            b_space: str = "Hdiv",
            projected: bool = False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            add_magnetic_propagators(self, b_field=b_field, b_space=b_space, projected=projected)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: MagneticFieldMixin.EMFields
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        b_space: MagneticFieldMixin.OptsBSpace = "Hdiv",
        projected: bool = False,
    ):
        # 0. store input parameters
        self._store_params(locals())
        # for the hooks (see LinearEuler)
        self._init_magnetic_field(b_space, projected)

        # 1.-5. see LinearEuler
        super().__init__(base_units, mass_number)

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum (Lorentz force):

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Energy (adiabatic process):

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde p`, :math:`\tilde{\mathbf B}` are the perturbed
        density, velocity, pressure and magnetic field, and :math:`\rho_0`, :math:`p_0`, :math:`\mathbf B_0` those of a
        static MHD equilibrium; :math:`\gamma = 5/3`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries):

        Mass perturbation and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Perturbed energy, for an equilibrium with :math:`\nabla \times \mathbf B_0 = 0` and :math:`\nabla p_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2 + \frac 12 |\tilde{\mathbf B}|^2
            + \frac{\tilde p^2}{2 \gamma p_0} \right) \textnormal d \mathbf x = 0

        For general equilibria and perturbations generated by a plasma displacement :math:`\boldsymbol \xi`
        (:math:`\partial_t \boldsymbol \xi = \tilde{\mathbf U}`,
        :math:`\tilde{\mathbf B} = \nabla \times (\boldsymbol \xi \times \mathbf B_0)`,
        :math:`\tilde p = - \boldsymbol \xi \cdot \nabla p_0 - \gamma p_0 \nabla \cdot \boldsymbol \xi`, e.g. for a pure
        velocity perturbation at :math:`t = 0`), the conserved energy is the sum of the kinetic energy and the potential
        energy :math:`\delta W(\boldsymbol \xi)` (energy principle of ideal MHD).
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
        - Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
        - Mean pressure perturbation: :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (``en_p``). This is a linear diagnostic, not an energy.
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{1}{2}\int \frac{\tilde{p}^2}{\gamma p_0} \, \mathrm{d}V` (``en_thermal``), with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_{\mathrm{thermal}}` (``en_tot``). Its
          change by the shear-Alfvén step is zero (up to the solver tolerances) with ``b_space="Hdiv"`` and with
          ``b_space="Hcurl", projected=True``; with ``b_space="Hcurl"``, ``projected=False`` this step damps it
          numerically (the damping vanishes with the resolution along the background field). Its change by the sound
          wave step is zero too (up to the solver tolerances); the background current and background pressure steps
          change it only for :math:`\nabla \times \mathbf B_0 \neq 0` or :math:`\nabla p_0 \neq 0`.
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V`
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`"""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations in a magnetized plasma with a static
        equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. The model solves the linearized
        ideal magnetohydrodynamic equations, which couple fluid dynamics (density, velocity, pressure)
        with magnetic field evolution.

        The linear MHD system supports three wave families:

        - **Alfvén waves:** incompressible shear waves propagating along :math:`\mathbf{B}_0`
        - **Fast magnetosonic waves:** compressible waves with phase velocity above the Alfvén speed
        - **Slow magnetosonic waves:** compressible waves with phase velocity below the Alfvén speed

        All evolved quantities are perturbations around a stationary equilibrium:

        - :math:`\tilde{\rho}` - density perturbation
        - :math:`\tilde{\mathbf{U}}` - velocity perturbation
        - :math:`\tilde{p}` - pressure perturbation
        - :math:`\tilde{\mathbf{B}}` - magnetic field perturbation

        The corresponding equilibrium quantities are:

        - :math:`\rho_0` - background density
        - :math:`p_0` - background pressure
        - :math:`\mathbf{B}_0` - background magnetic field

        Two discretizations of the magnetic field are available through the flag ``b_space``. With ``"Hdiv"``
        (default), :math:`\tilde{\mathbf B}` is a 2-form and the induction equation holds strongly. With ``"Hcurl"``,
        :math:`\tilde{\mathbf B}` is a 1-form, the induction equation is weak and the implicit shear-Alfvén step solves a
        field-aligned curl-curl system for the magnetic field, preconditioned by Hiptmair-Xu, which allows large time
        steps; the force :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}` of the background current step then
        enters through the mixed mass matrix :attr:`~struphy.feec.mass.WeightedMassOperators.M21J`. This shear-Alfvén
        step damps the waves numerically, with a damping that vanishes quickly with the resolution along the
        background field; ``projected=True`` gives the exactly energy conserving version, at the price of an inner mass
        matrix solve in every application of the Schur operator.

        Boundary conditions with ``b_space="Hcurl"``: essential boundary conditions in the de Rham sequence set the
        tangential components of the magnetic field to zero on that boundary, which is unphysical at a wall with
        :math:`\mathbf B_0 \cdot \mathbf n = 0`. Free boundary conditions are therefore recommended: they impose
        :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` weakly in the shear-Alfvén step and :math:`\tilde p = 0` weakly in
        the sound wave step (natural boundary condition of the pressure term).

        The species, the sound wave step and the background pressure step are inherited from
        :class:`~struphy.models.LinearEuler`. The compressional part of the time step thus consists of three steps:
        the sound wave step, which
        conserves the energy :math:`\frac 12 \int \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{1}{2\gamma} \int \tilde p^2 / p_0 \,\textnormal d \mathbf x` exactly (Crank-Nicolson with the
        :math:`1/p_0`-weighted pressure equation), and the steps of the two terms that change the perturbed energy for
        general equilibria, :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}` in the momentum equation
        and :math:`- \tilde{\mathbf U} \cdot \nabla p_0` in the pressure equation, which are solved exactly in time.
        The three steps do not commute; they form one block of the splitting with symmetric (Strang) splitting, the
        sound wave step in the middle, so that the compressional part is second order in time as the former unsplit
        magnetosonic step (with Lie-Trotter sequencing of the three steps the frequencies of modes coupled to the
        compressional dynamics, e.g. TAEs in a torus, are wrong for large time steps).
        For an equilibrium with :math:`\nabla p_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`, the scheme is thus
        energy conserving up to the solver tolerances (with ``b_space="Hdiv"`` or
        ``projected=True``)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear MHD model:

        .. code-block:: python

            from struphy.models import LinearMHD

            model = LinearMHD()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_p"]

            # magnetic field in H(curl), exactly energy conserving shear-Alfvén step
            model_cc = LinearMHD(b_space="Hcurl", projected=True)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear wave propagation in a static magnetized plasma,
        including Alfvén and magnetosonic dynamics around a prescribed equilibrium.

        Typical use cases include:

        - verification of linear MHD wave dispersion and mode structure
        - perturbative studies around static equilibria
        - benchmark problems for ideal-MHD field-fluid coupling
        - large time steps with the curl-curl discretization (``b_space="Hcurl"``)"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description
        - polar splines with the Hiptmair-Xu preconditioner (``b_space="Hcurl"``)"""
