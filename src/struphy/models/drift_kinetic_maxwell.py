import copy
import logging

from struphy import BaseUnits
from struphy.models.guiding_center import GuidingCenter
from struphy.models.scalars import FunctionScalarFEEC, Scalars
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.drift_kinetic_maxwell_coupling import DriftKineticMaxwellCoupling
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

logger = logging.getLogger("struphy")


class DriftKineticMaxwell(GuidingCenter):
    """Electromagnetic drift-kinetic equation for one guiding-center species coupled to Maxwell's equations, in the
    thermal normalization, with an implicit, energy conserving field-particle coupling that also holds in the
    quasi-neutral limit.

    The electric field enters the guiding-center equations through the gauge-invariant
    :math:`\\mathbf E^* = \\mathbf E - \\varepsilon \\mu \\nabla |B|`. Ampère's law contains the guiding-center
    current and the (linearized) polarization current; the displacement current has the coefficient
    :math:`\\lambda^2 \\varepsilon^2 = \\hat v_\\textrm{th}^2/c^2`, option ``lam`` of the coupling propagator
    :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`; ``lam=0.0`` is the
    quasi-neutral limit. Extends :class:`~struphy.models.GuidingCenter` by the electromagnetic fields.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``; the thermal energy defines the velocity unit).
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass
        numbers.
    lam: float, optional
        Cyclotron frequency × length unit / speed of light. If None, computed from units and charge/mass numbers.
    kappa: float, optional
        Plasma frequency × length unit / speed of light. If None, computed from units and charge/mass numbers.
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class KineticIons(GuidingCenter.KineticIons):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
            lam: float = None,
            kappa: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
                lam=lam,
                kappa=kappa,
            )

    ## propagators

    class Propagators(GuidingCenter.Propagators):
        splitting_order = ("coupling", "push_parallel", "push_bxe")

        def __init__(
            self,
            e_field: FEECVariable = None,
            b_field: FEECVariable = None,
        ):
            super().__init__()
            # the pushers of GuidingCenter with the total magnetic field B0 + b_field; the parallel electric force is
            # applied by the coupling step only: push_parallel gets no phi and no e_field
            self.push_bxe = PushGuidingCenterBxEstar(b_tilde=b_field, e_field=e_field)
            self.push_parallel = PushGuidingCenterParallel(b_tilde=b_field)
            self.coupling = DriftKineticMaxwellCoupling()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
        lam: float = None,
        kappa: float = None,
    ):
        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(charge_number, mass_number, epsilon, lam, kappa)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(e_field=self.em_fields.e_field, b_field=self.em_fields.b_field)

        # 4. assign variables to propagators
        self._link_propagator_variables()

        # 5. define scalars to be tracked during simulation: the parts of the energy conserved by the coupling step
        # (with the coefficients of the propagator, known after its allocation)
        coupling = self.propagators.coupling
        electric_energy = FunctionScalarFEEC(lambda: coupling.energies()[0])
        magnetic_energy = FunctionScalarFEEC(lambda: coupling.energies()[1])
        particle_energy = FunctionScalarFEEC(lambda: coupling.energies()[2])
        self.scalars = Scalars(
            en_E=electric_energy,
            en_B=magnetic_energy,
            en_f=particle_energy,
            en_tot=electric_energy + magnetic_energy + particle_energy,
        )

    def _link_propagator_variables(self):
        """Assign the species variables to the propagators.

        :meta private:
        """
        super()._link_propagator_variables()
        self.propagators.coupling.variables.e = self.em_fields.e_field
        self.propagators.coupling.variables.b = self.em_fields.b_field
        self.propagators.coupling.variables.ions = self.kinetic_ions.var

    @property
    def velocity_scale(self):
        return "thermal"

    def post_allocate(self):
        """Check that the parallel electric force is applied only once (by the coupling step), then no initial field
        solve: the electric field is initialized from the parameter file.

        :meta private:
        """
        push_parallel = self.propagators.push_parallel
        assert push_parallel.e_field is None and push_parallel._allocate_phi, (
            "DriftKineticMaxwell: the parallel electric force b* . E is applied by the coupling step; "
            "push_parallel must not get an electric field or a potential (it would be applied twice)."
        )

    ## default parameters

    def generate_default_parameter_file(self, path=None, prompt=True):
        """The thermal velocity scale needs the thermal energy unit: ``BaseUnits(kBT=1.0)``.

        :meta private:
        """
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        with open(params_path, "r") as f:
            lines = f.readlines()
        with open(params_path, "w") as f:
            for line in lines:
                f.write("base_units = BaseUnits(kBT=1.0)\n" if "BaseUnits(" in line else line)
        return params_path

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Drift-kinetic equation for :math:`f(\mathbf X, v_\parallel, \mu, t)`:

        .. math::

            \frac{\partial f}{\partial t} + \frac{v_\parallel \mathbf B^* + \mathbf E^* \times \mathbf b_0}{B^*_\parallel}
            \cdot \nabla f + \frac{1}{\varepsilon} \frac{\mathbf B^* \cdot \mathbf E^*}{B^*_\parallel}
            \frac{\partial f}{\partial v_\parallel} = 0\,,

        with

        .. math::

            \mathbf E^* = \mathbf E - \varepsilon \mu \nabla |\mathbf B|\,,\qquad
            \mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
            B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*\,,

        where :math:`|\mathbf B| = \mathbf b_0 \cdot (\mathbf B_0 + \tilde{\mathbf B})`: the total magnetic field in
        the first term of :math:`\mathbf B^*` (parallel streaming along the perturbed field lines), the curvature
        term with the equilibrium unit vector :math:`\mathbf b_0`.

        Ampère's law, with the displacement current and the (linearized) polarization current of the guiding centers:

        .. math::

            \varepsilon^2 \lambda^2 \frac{\partial \mathbf E}{\partial t}
            + \varepsilon^2 \kappa^2 \frac{n_0}{|\mathbf B_0|^2} \frac{\partial \mathbf E_\perp}{\partial t}
            = \nabla \times \tilde{\mathbf B} - \varepsilon \kappa^2 \iint f B^*_\parallel
            \left( v_\parallel \frac{\mathbf B^*}{B^*_\parallel} + \frac{\mathbf E \times \mathbf b_0}{B^*_\parallel}
            \right) \textrm d v_\parallel \textrm d \mu\,,

        Faraday's law:

        .. math::

            \frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla \times \mathbf E = 0\,.

        :math:`\mathbf B_0` is a static equilibrium field, :math:`n_0` the equilibrium density, and
        :math:`\mathbf E_\perp = \mathbf E - (\mathbf b_0 \cdot \mathbf E) \mathbf b_0`. The current in Ampère's law is
        the guiding-center current without the magnetic (grad-:math:`B`) drift, whose work on :math:`\mathbf E` is
        therefore missing where :math:`\nabla |B| \neq 0`. The coefficients :math:`\varepsilon\lambda` (option ``lam``,
        0 is the quasi-neutral limit), :math:`(\varepsilon\kappa)^2` (option ``chi``) and :math:`\varepsilon\kappa^2`
        (option ``c2``) can be changed in
        :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`. Without the
        polarization current the quasi-neutral limit would be singular for flute modes (:math:`k_\parallel = 0`).
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the thermal velocity of the species as reference velocity:

        .. math::

            \hat v = \hat v_\textrm{th} = \sqrt{\frac{k_\textrm{B} \hat T}{m}}\,,\qquad \hat t = \frac{\hat x}{\hat v}\,,
            \qquad \hat E = \hat v \hat B\,,\qquad \hat \mu = \frac{m \hat v^2}{\hat B}\,.

        The species parameters (see :ref:`equation_params`) are

        .. math::

            \varepsilon = \frac{1}{\hat\Omega_c \hat t} = \frac{\hat v}{\hat\Omega_c \hat x}\,,\qquad
            \lambda = \frac{\hat\Omega_c \hat x}{c}\,,\qquad \kappa = \frac{\hat\Omega_p \hat x}{c}\,.

        The drift-kinetic equation and Faraday's law then have only the coefficient :math:`\varepsilon`
        (:math:`\frac{m \hat v}{q \hat B \hat x} = \varepsilon` in :math:`\mathbf B^*` and :math:`\mathbf E^*`,
        :math:`\frac{q \hat B \hat t}{m} = \frac{1}{\varepsilon}` in the parallel force). Ampère's law
        :math:`\frac{1}{c^2} \partial_t \mathbf E + \mu_0 \frac{n m}{|B|^2} \partial_t \mathbf E_\perp = \nabla \times
        \mathbf B - \mu_0 q \mathbf J`, multiplied by :math:`\hat x / \hat B`, has the coefficients

        .. math::

            \frac{\hat v^2}{c^2} = \varepsilon^2 \lambda^2\,,\qquad
            \frac{\mu_0 \hat n m \hat v^2}{\hat B^2} = \frac{\hat v^2}{\hat v_A^2} = \varepsilon^2 \kappa^2\,,\qquad
            \frac{\mu_0 q \hat n \hat v \hat x}{\hat B} = \varepsilon \kappa^2\,.

        Their ratio :math:`\kappa^2/\lambda^2 = \hat\Omega_p^2/\hat\Omega_c^2 = c^2/\hat v_A^2` compares the polarization
        with the vacuum dielectric, and :math:`1/\kappa` is the inertial length :math:`c/\hat\Omega_p` in units of
        :math:`\hat x`. For example, deuterium with :math:`\hat x = 1` m, :math:`\hat B = 5` T,
        :math:`\hat n = 10^{20}` m\ :sup:`-3`, :math:`k_\textrm{B} \hat T = 10` keV gives :math:`\varepsilon = 2.9 \cdot
        10^{-3}`, :math:`\lambda = 0.80`, :math:`\kappa = 31`, i.e. :math:`\varepsilon^2 \lambda^2 = 5.3 \cdot 10^{-6}`,
        :math:`\varepsilon^2 \kappa^2 = 8.0 \cdot 10^{-3}` and :math:`\varepsilon \kappa^2 = 2.8`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation** (the parts of the energy conserved by the coupling
        step, with the coefficients of
        :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`, here the defaults):

        - Electric field energy (displacement and polarization)
          :math:`\frac{\varepsilon^2 \lambda^2}{2} \| \mathbf E \|^2 + \frac{\varepsilon^2 \kappa^2}{2}
          \| \mathbf E_\perp \|^2_{n_0/|B_0|^2}`: ``en_E``
        - Magnetic field energy :math:`\frac{1}{2} \| \tilde{\mathbf B} \|^2`: ``en_B``
        - Particle energy :math:`\varepsilon^2 \kappa^2 \sum_p w_p (v_{\parallel,p}^2/2 + \mu_p |B_0|)`: ``en_f``
        - Total energy: ``en_tot`` (conserved by the coupling step up to the solver tolerance)"""

    @classmethod
    def doc_discretization(cls):
        r"""Time integration is performed by the following propagators (in sequence).
        Each term of the drift-kinetic equation is applied by exactly one propagator; the parallel electric
        force only by the coupling step (the model passes no potential and no electric field to ``push_parallel``, and
        asserts this):

        - :math:`\dot v_\parallel \ni \frac{1}{\varepsilon} \mathbf b^* \cdot \mathbf E`: coupling step
          (``push_vpar_with_efield``);
        - :math:`\dot v_\parallel \ni -\mu\, \mathbf b^* \cdot \nabla |B|` (mirror force): ``push_parallel``;
        - :math:`\dot{\mathbf X} \ni v_\parallel \mathbf b^*` (streaming, curvature drift): ``push_parallel``; its
          current enters Ampère's law in the coupling step;
        - :math:`\dot{\mathbf X} \ni \mathbf E \times \mathbf b_0 / B^*_\parallel`: ``push_bxe``; its current
          enters Ampère's law in the coupling step (option ``exb_current``);
        - :math:`\dot{\mathbf X} \ni -\varepsilon \mu \nabla |B| \times \mathbf b_0 / B^*_\parallel`
          (grad-:math:`B` drift): ``push_bxe``; its current is not included."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""DriftKineticMaxwell couples one drift-kinetic (guiding-center) species to Maxwell's equations in the
        gauge-invariant :math:`v_\parallel` formulation (the electric field, not a potential, enters
        :math:`\mathbf E^*`), normalized with the thermal velocity of the species. Maxwell's equations and the parallel
        acceleration are solved together in one implicit, energy conserving step, which allows time steps above the
        plasma period and has no cancellation problem. Since the drift-kinetic ordering removes the polarization drift,
        its (linearized) polarization current is added to Ampère's law; with it, the quasi-neutral limit without
        displacement current is well-posed and gives the inertial Alfvén wave. The guiding centers then stream along
        the field and drift with :math:`\mathbf E^* \times \mathbf b_0`. The total energy is conserved exactly by the
        splitting if :math:`\nabla |B_0| = 0` and either :math:`\mu = 0` or :math:`\mathbf b_0 \cdot \tilde{\mathbf B}
        = 0`."""

    @classmethod
    def doc_examples(cls):
        r"""Create a drift-kinetic Maxwell model in the quasi-neutral limit:

        .. code-block:: python

            from struphy.models import DriftKineticMaxwell

            model = DriftKineticMaxwell()
            prop = model.propagators.coupling
            prop.options = prop.Options(lam=0.0)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic drift-kinetic simulations with time steps above the plasma period
        - shear (inertial) Alfvén waves with kinetic parallel dynamics
        - the quasi-neutral limit (``lam=0.0``) with full-f weights"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - polar splines (not supported yet by the coupling propagator)
        - the quasi-neutral limit with control-variate (delta-f) weights or with vacuum regions
        - finite Larmor radius effects beyond the linearized long-wavelength polarization
        - exact energy conservation in inhomogeneous :math:`|B_0|` (the grad-:math:`B` drift current is missing)"""
