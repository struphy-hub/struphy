from struphy.io.options import BaseUnits
from struphy.models.linear_euler_mixins import MagneticFieldMixin, add_magnetic_propagators
from struphy.models.linear_euler_with_q import LinearEulerWithQ
from struphy.models.variables import FEECVariable


class LinearMHDWithQ(MagneticFieldMixin, LinearEulerWithQ):
    """Linear ideal MHD with zero-flow equilibrium, with the square root of the pressure, :math:`q = \\sqrt p`, as
    thermodynamic variable.

    Extends :class:`~struphy.models.LinearEulerWithQ` by the magnetic field: the Lorentz force and the induction
    equation. The magnetic field (species ``em_fields``, propagators ``shear_alf`` and ``bg_current``, scalars) is the
    one of :class:`~struphy.models.LinearMHD`.

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

    class Propagators(LinearEulerWithQ.Propagators):
        """Propagators of :class:`~struphy.models.LinearEulerWithQ` plus ``shear_alf`` (shear-Alfvén waves) and
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
            Passed to :class:`LinearEulerWithQ.Propagators <struphy.models.LinearEulerWithQ.Propagators>`.
        """

        splitting_order = ("shear_alf", "bg_current", "bg_q", "sound")
        # the background steps are exact in time but do not commute with the sound wave step: symmetric splitting
        # of the three keeps the compressional part second order in time (as in LinearMHD)
        strang_groups = (("bg_current", "bg_q", "sound"),)

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
        # for the hooks (see LinearEulerWithQ)
        self._init_magnetic_field(b_space, projected)

        # 1.-5. see LinearEulerWithQ
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

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla (2 q_0 \tilde q) = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla q_0
            + \frac{\gamma}{2} q_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q`, :math:`\tilde{\mathbf B}` are the perturbed
        density, velocity, square root of the pressure and magnetic field, and :math:`\rho_0`,
        :math:`q_0 = \sqrt{p_0}`, :math:`\mathbf B_0` those of a static MHD equilibrium; :math:`\gamma = 5/3`. The
        pressure perturbation is :math:`\tilde p = 2 q_0 \tilde q`, and the equations are those of
        :class:`~struphy.models.LinearMHD` written in :math:`\tilde q`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries):

        Mass perturbation and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Perturbed energy, for an equilibrium with :math:`\nabla \times \mathbf B_0 = 0` and :math:`\nabla q_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2 + \frac 12 |\tilde{\mathbf B}|^2
            + \frac{2}{\gamma} \tilde q^2 \right) \textnormal d \mathbf x = 0

        For general equilibria and perturbations generated by a plasma displacement :math:`\boldsymbol \xi`, the
        conserved energy is the sum of the kinetic energy and the potential energy :math:`\delta W(\boldsymbol \xi)`
        (energy principle of ideal MHD), as for :class:`~struphy.models.LinearMHD`.
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}\,, \qquad
            \hat{q} = \sqrt{\hat p} = \frac{\hat B_0}{\sqrt{\mu_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V` (``en_U``)
        - Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V` (``en_B``)
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{2}{\gamma}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_thermal``), with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_{\mathrm{thermal}}` (``en_tot``). Its
          change by the shear-Alfvén step is zero (up to the solver tolerances) with ``b_space="Hdiv"`` and with
          ``b_space="Hcurl", projected=True``; with ``b_space="Hcurl"``, ``projected=False`` this step damps it
          numerically (the damping vanishes with the resolution along the background field). Its change by the sound
          wave step is zero too (up to the solver tolerances); the background current and background :math:`q`
          steps change it only for :math:`\nabla \times \mathbf B_0 \neq 0` or :math:`\nabla q_0 \neq 0`.
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V` (``en_B_eq``)
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V` (``en_p_eq``)
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V` (``en_B_tot``)"""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations in a magnetized plasma with a static
        equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. It solves the linearized ideal
        MHD equations of :class:`~struphy.models.LinearMHD` with the square root of the pressure,
        :math:`q = \sqrt p`, instead of the pressure; it is the linear limit of the perturbative MHD model in
        :math:`q`, and supports the same waves (shear Alfvén, fast and slow magnetosonic).

        The magnetic field and its two discretizations (flag ``b_space``: ``"Hdiv"`` with the strong induction
        equation, ``"Hcurl"`` with the weak induction equation and the field-aligned curl-curl solve, exactly energy
        conserving with ``projected=True``) are those of :class:`~struphy.models.LinearMHD`, shared through
        :class:`~struphy.models.linear_euler_mixins.MagneticFieldMixin`. The species, the sound wave step and the
        background :math:`q` step are inherited from :class:`~struphy.models.LinearEulerWithQ`.

        The compressional part of the time step consists of three steps: the sound wave step, which conserves the
        energy :math:`\frac 12 \int \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{2}{\gamma} \int \tilde q^2 \,\textnormal d \mathbf x` exactly (Crank-Nicolson), and the steps of the
        two terms that change the perturbed energy for general equilibria,
        :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}` in the momentum equation and
        :math:`- \tilde{\mathbf U} \cdot \nabla q_0` in the equation of :math:`\tilde q`, which are solved exactly in
        time. The three steps form one block of the splitting with symmetric (Strang) splitting, the sound wave step in
        the middle, so that the compressional part is second order in time. For an equilibrium with
        :math:`\nabla p_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`, the scheme is thus energy conserving up to
        the solver tolerances (with ``b_space="Hdiv"`` or ``projected=True``).

        Boundary conditions with ``b_space="Hcurl"``: free boundary conditions are recommended, as for
        :class:`~struphy.models.LinearMHD`."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear MHD model in :math:`q`:

        .. code-block:: python

            from struphy.models import LinearMHDWithQ

            model = LinearMHDWithQ()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_thermal"]

            # magnetic field in H(curl), exactly energy conserving shear-Alfvén step
            model_cc = LinearMHDWithQ(b_space="Hcurl", projected=True)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear wave propagation in a static magnetized plasma with the
        variable :math:`q = \sqrt p`, including Alfvén and magnetosonic dynamics around a prescribed equilibrium.

        Typical use cases include:

        - verification of linear MHD wave dispersion and mode structure
        - the linear limit of the perturbative MHD model in :math:`q`
        - comparison with :class:`~struphy.models.LinearMHD` (same equations in :math:`\tilde p`)"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description
        - polar splines with the Hiptmair-Xu preconditioner (``b_space="Hcurl"``)"""
