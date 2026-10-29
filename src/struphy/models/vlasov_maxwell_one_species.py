import logging

from struphy import BaseUnits
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.variables import FEECVariable
from struphy.models.vlasov_ampere_one_species import VlasovAmpereOneSpecies
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere

logger = logging.getLogger("struphy")


class VlasovMaxwellOneSpecies(VlasovAmpereOneSpecies):
    """Vlasov-Maxwell equations for one kinetic species.

    Extends :class:`~struphy.models.VlasovAmpereOneSpecies` by the magnetic field (Faraday's law and the curl of
    the magnetic field in Ampère's law).

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
    measure_gauss_law: bool
        Whether to track the Gauss-law error as a scalar quantity (default: False)
    """

    ## additional species variables

    class EMFields(VlasovAmpereOneSpecies.EMFields):
        def __init__(self):
            # declared before the parent's variables, whose init_variables() then includes it
            self.b_field = FEECVariable(space="Hdiv")
            super().__init__()

    ## additional propagators

    class Propagators(VlasovAmpereOneSpecies.Propagators):
        """Propagators of :class:`~struphy.models.VlasovAmpereOneSpecies` plus ``maxwell`` (Faraday's law and the
        curl of the magnetic field in Ampère's law).

        Parameters
        ----------
        **kwargs
            Passed to :class:`VlasovAmpereOneSpecies.Propagators <struphy.models.VlasovAmpereOneSpecies.Propagators>`
            (and from there to :class:`Vlasov.Propagators <struphy.models.Vlasov.Propagators>`).
        """

        splitting_order = ("maxwell", "push_eta", "push_vxb", "coupling_va")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.maxwell = MaxwellWeakAmpere()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        measure_gauss_law: bool = False,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # property to measure violation of gauss law from control variate
        self.measure_gauss_law = measure_gauss_law

        # 1.-5. and initial Poisson, see VlasovAmpereOneSpecies
        super().__init__(base_units, charge_number, mass_number, alpha, epsilon)

    def _propagator_args(self):
        """:meta private:"""
        # the Lorentz force of the total magnetic field B0 + B
        return {"b2_var": self.em_fields.b_field}

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        scalars.pop("en_tot")
        scalars["en_B"] = BilinearEnergyFEEC(self.em_fields.b_field)
        scalars["en_tot"] = scalars["en_E"] + scalars["en_B"] + scalars["en_f"]
        if self.measure_gauss_law:
            # the MPI reduction (max over ranks) is done in calculate_gauss_error, not by summing the scalar
            scalars["gauss_error"] = FunctionScalarFEEC(self.calculate_gauss_error)
        return scalars

    def post_allocate(self):
        """Solve initial Poisson equation and prepare the Gauss-law diagnostic.

        :meta private:
        """
        super().post_allocate()
        if self.measure_gauss_law:
            self._setup_gauss_law()

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Ampère's law:

        .. math::

            -\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        where :math:`Z=-1` and :math:`A=1/1836` for electrons.

        At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

        .. math::

            \int_{\Omega} \nabla \psi^{\top} \cdot \nabla \phi \, \textrm{d} \mathbf{x} = \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} \psi \, (f - f_0) \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
            \\[2mm]
            \mathbf{E}(t=0) = -\nabla \phi(t=0)

        Moreover, it is assumed that

        .. math::

            \nabla \times \mathbf{B}_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}

        where :math:`\mathbf{B}_0` is the static equilibrium magnetic field.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the light speed as reference velocity:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat\phi = \hat E \hat x.

        The species parameters are :math:`\alpha=\hat\Omega_p/\hat\Omega_c` and
        :math:`\varepsilon=1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Magnetic field energy: ``en_B``
        - Particle kinetic energy: ``en_f``
        - Total energy: ``en_tot``
        - Optional Gauss-law diagnostic: ``gauss_error``"""

    @classmethod
    def doc_long_description(cls):
        r"""VlasovMaxwellOneSpecies is the fully electromagnetic one-species PIC
        model in Struphy. It evolves particles and fields self-consistently and
        supports an optional control-variate formulation for the field coupling."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Vlasov-Maxwell model:

        .. code-block:: python

            from struphy.models import VlasovMaxwellOneSpecies

            model = VlasovMaxwellOneSpecies()
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - self-consistent electromagnetic kinetic simulations
        - one-species PIC benchmarks
        - wave-particle interaction studies with evolving magnetic fields
        - verification of the full Vlasov-Maxwell splitting"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - multi-species plasma dynamics without extension
        - collisional kinetic closures
        - reduced electrostatic-only models where magnetic evolution is unnecessary
        - linearized delta-f studies that should use the dedicated linear models"""
