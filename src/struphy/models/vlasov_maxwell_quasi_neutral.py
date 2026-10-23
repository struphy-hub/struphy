from struphy import BaseUnits
from struphy.models.scalars import FunctionScalarFEEC, Scalars
from struphy.models.vlasov_maxwell_one_species import VlasovMaxwellOneSpecies
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_maxwell_coupling import VlasovMaxwellCoupling


class VlasovMaxwellQuasiNeutral(VlasovMaxwellOneSpecies):
    """Vlasov-Maxwell equations for one kinetic species with an implicit, energy conserving field-particle coupling
    that also holds in the quasi-neutral limit (no displacement current).

    Same as :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`, but Ampère's law, Faraday's
    law and the electric force on the particles are solved together in one Crank-Nicolson step
    (:class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`). Its option ``kappa`` multiplies the
    displacement current; ``kappa=0.0`` is the quasi-neutral limit.

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

    class Propagators:
        def __init__(
            self,
            b2_var=None,
        ):
            self.coupling = VlasovMaxwellCoupling()
            self.push_eta = PushEta()
            self.push_vxb = PushVxB(b2_var=b2_var)

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        measure_gauss_law: bool = False,
    ):
        super().__init__(
            base_units=base_units,
            charge_number=charge_number,
            mass_number=mass_number,
            alpha=alpha,
            epsilon=epsilon,
            measure_gauss_law=measure_gauss_law,
        )

        # the energy conserved by the coupling step (with its coefficients kappa, c1, c2, c3)
        self.scalars = Scalars(**self.scalars.dct, en_H=FunctionScalarFEEC(self.propagators.coupling.energy))

    def _link_propagator_variables(self):
        """Assign the species variables to the propagators.

        :meta private:
        """
        self.propagators.coupling.variables.e = self.em_fields.e_field
        self.propagators.coupling.variables.b = self.em_fields.b_field
        self.propagators.coupling.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Ampère's law, with the coefficient :math:`\kappa \geq 0` of the displacement current (option ``kappa`` of
        :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`, not the species parameter):

        .. math::

            -\kappa \frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        For :math:`\kappa = 1` these are the equations of
        :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`. For :math:`\kappa = 0`
        (quasi-neutral limit) Ampère's law becomes :math:`\nabla \times \mathbf B = \frac{\alpha^2}{\varepsilon}
        \mathbf J`, which implies :math:`\nabla \cdot \mathbf J = 0`; the electric field is then determined by this
        constraint, not by its own evolution. The coefficients of the three coupling terms can be changed by the
        options ``c1``, ``c2``, ``c3`` of the propagator.

        At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law, as in
        :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies` (for :math:`\kappa = 0` the
        initial electric field does not enter the time stepping).
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Magnetic field energy: ``en_B``
        - Particle kinetic energy: ``en_f``
        - Total energy: ``en_tot``
        - Energy conserved by the coupling step,
          :math:`\frac{\kappa}{2} \| \mathbf E \|^2 + \frac{c_1}{2} \| \mathbf B \|^2 + \frac{c_2}{2 c_3} \sum_p w_p
          |\mathbf v_p|^2`: ``en_H`` (equal to ``en_tot`` for the default coefficients and :math:`\kappa = 1`)
        - Optional Gauss-law diagnostic: ``gauss_error``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`
        2. :class:`~struphy.propagators.push_eta.PushEta`
        3. :class:`~struphy.propagators.push_vxb.PushVxB`
        """
        doc = rf"""**1. VlasovMaxwellCoupling:**

{VlasovMaxwellCoupling.__doc__}

**2. PushEta:**

{PushEta.__doc__}

**3. PushVxB:**

{PushVxB.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""VlasovMaxwellQuasiNeutral solves the Vlasov-Maxwell system of
        :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies` with one implicit step for
        Maxwell's equations and the electric force, which conserves the energy exactly and allows time steps
        larger than the inverse plasma frequency and the light-wave CFL limit. Setting the coefficient of the
        displacement current to zero gives the quasi-neutral (Darwin-type) limit, in which the electric field is the
        Lagrange multiplier of a divergence-free current."""

    @classmethod
    def doc_examples(cls):
        r"""Create a quasi-neutral Vlasov-Maxwell model:

        .. code-block:: python

            from struphy.models import VlasovMaxwellQuasiNeutral

            model = VlasovMaxwellQuasiNeutral()
            prop = model.propagators.coupling
            prop.options = prop.Options(kappa=0.0)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic kinetic simulations with time steps above the plasma period
        - the quasi-neutral limit of the Vlasov-Maxwell system (``kappa=0.0``)
        - comparison with :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - the quasi-neutral limit with control-variate (delta-f) weights or with vacuum regions
        - multi-species plasma dynamics without extension
        - collisional kinetic closures"""
