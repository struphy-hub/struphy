import copy

from struphy import BaseUnits
from struphy.models.linear_vlasov_ampere_one_species import LinearVlasovAmpereOneSpecies
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarPIC, Scalars
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere


class LinearVlasovMaxwellOneSpecies(LinearVlasovAmpereOneSpecies):
    """Linearized Vlasov-Maxwell equations for one kinetic species around a Maxwellian background.

    Extends :class:`~struphy.models.LinearVlasovAmpereOneSpecies` by the magnetic field perturbation (Faraday's law
    and the curl of the magnetic field in Ampère's law).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    alpha: float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass numbers.
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators(LinearVlasovAmpereOneSpecies.Propagators):
        def __init__(self):
            super().__init__()
            self.maxwell = MaxwellWeakAmpere()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            alpha,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self._link_propagator_variables()

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        particle_energy = FunctionScalarPIC(
            self._compute_en_w,
            self.kinetic_ions.var,
        )
        self.scalars = Scalars(
            en_E=electric_energy,
            en_B=magnetic_energy,
            en_w=particle_energy,
            en_tot=electric_energy + magnetic_energy + particle_energy,
        )

        # initial Poisson (not a propagator used in time stepping);
        # the source is the delta-f charge density, see LinearVlasovAmpereOneSpecies
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon
        self._create_initial_poisson(rho_coeffs=alpha**2 / epsilon, delta_f_source=False)

    def _link_propagator_variables(self):
        """Assign the species variables to the propagators.

        :meta private:
        """
        super()._link_propagator_variables()
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Linearized Ampère's law:

        .. math::

            \frac{\partial \tilde{\mathbf{E}}}{\partial t} = \nabla \times \tilde{\mathbf{B}} - \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} \tilde{f} \, \textrm{d}^3 \mathbf{v}

        Linearized Faraday's law:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} = -\nabla \times \tilde{\mathbf{E}}

        Linearized Vlasov equation:

        .. math::

            \frac{\partial \tilde{f}}{\partial t} + \mathbf{v} \cdot \nabla \tilde{f} + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial \tilde{f}}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \tilde{\mathbf{E}} \cdot \mathbf{v} f_0

        where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

        .. math::

            f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3} \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right)

        and the background electric field has to verify the following compatibility condition between with background density

        .. math::

            \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0

        At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

        .. math::

            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \, \textrm{d} \mathbf{x}
            &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi \, \tilde{f} \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x}
            \qquad \forall \ \psi \in H^1
            \\[2mm]
            \tilde{\mathbf{E}}(t=0) &= -\nabla \phi(t=0)

        Moreover, it is assumed that

        .. math::

            \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The normalization matches the linear Vlasov-Ampère model:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat\phi = \hat E \hat x.

        The model retains the same :math:`\alpha` and :math:`\varepsilon`
        parameters while adding magnetic perturbation dynamics."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Magnetic field energy: ``en_B``
        - Perturbation particle energy: ``en_w``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_long_description(cls):
        r"""LinearVlasovMaxwellOneSpecies is the electromagnetic extension of the
        linear delta-f Vlasov model. It is intended for weakly perturbed kinetic
        electromagnetic waves and instabilities around a prescribed equilibrium."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear Vlasov-Maxwell model:

        .. code-block:: python

            from struphy.models import LinearVlasovMaxwellOneSpecies

            model = LinearVlasovMaxwellOneSpecies()
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear electromagnetic kinetic wave studies
        - delta-f verification of linear Vlasov-Maxwell coupling
        - weakly nonlinear regimes where equilibrium perturbations stay small"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - strongly nonlinear fully kinetic plasma dynamics
        - multi-species electromagnetic coupling
        - problems without a well-defined linearization background
        - collisional kinetic closures"""
