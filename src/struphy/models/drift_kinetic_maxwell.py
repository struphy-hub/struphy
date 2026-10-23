import copy
import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, KineticEnergyPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.drift_kinetic_maxwell_coupling import DriftKineticMaxwellCoupling
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

logger = logging.getLogger("struphy")


class DriftKineticMaxwell(StruphyModel):
    """Electromagnetic drift-kinetic equation for one guiding-center species coupled to Maxwell's equations, with an
    implicit, energy conserving field-particle coupling that also holds in the quasi-neutral limit.

    The electric field enters the guiding-center equations through the gauge-invariant
    :math:`\\mathbf E^* = \\mathbf E - \\varepsilon \\mu \\nabla |B_0|`. Ampère's law contains the guiding-center
    current and the (linearized) polarization current; option ``kappa`` of the coupling propagator
    :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling` multiplies the
    displacement current, ``kappa=0.0`` is the quasi-neutral limit.

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

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(
            self,
            e_field: FEECVariable = None,
            b_field: FEECVariable = None,
        ):
            self.coupling = DriftKineticMaxwellCoupling()
            self.push_para = PushGuidingCenterParallel(b_tilde=b_field)
            self.push_bxe = PushGuidingCenterBxEstar(b_tilde=b_field, e_field=e_field)

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
        self.propagators = self.Propagators(e_field=self.em_fields.e_field, b_field=self.em_fields.b_field)

        # 4. assign variables to propagators
        self.propagators.coupling.variables.e = self.em_fields.e_field
        self.propagators.coupling.variables.b = self.em_fields.b_field
        self.propagators.coupling.variables.ions = self.kinetic_ions.var
        self.propagators.push_para.variables.ions = self.kinetic_ions.var
        self.propagators.push_bxe.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        self.scalars = Scalars(
            en_E=BilinearEnergyFEEC(self.em_fields.e_field),
            en_B=BilinearEnergyFEEC(self.em_fields.b_field),
            en_f=KineticEnergyPIC(
                self.kinetic_ions.var,
                normalization=self.kinetic_ions.equation_params.alpha**2,
            ),
            en_tot=FunctionScalarFEEC(self.propagators.coupling.energy),
        )

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        """No initial field solve: the electric field is initialized from the parameter file.

        :meta private:
        """

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
            \mathbf B^* = \mathbf B_0 + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
            B^*_\parallel = \mathbf b_0 \cdot (\mathbf B_0 + \tilde{\mathbf B}) + \varepsilon v_\parallel
            (\nabla \times \mathbf b_0) \cdot \mathbf b_0\,,

        where :math:`|\mathbf B| = \mathbf b_0 \cdot (\mathbf B_0 + \tilde{\mathbf B})`.

        Ampère's law with the displacement current (coefficient :math:`\kappa`) and the polarization current
        (coefficient :math:`\chi`, default :math:`\alpha^2`), options of
        :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`:

        .. math::

            \kappa \frac{\partial \mathbf E}{\partial t}
            + \chi \frac{n_0}{|B_0|^2} \frac{\partial \mathbf E_\perp}{\partial t}
            = \nabla \times \tilde{\mathbf B} - \frac{\alpha^2}{\varepsilon} \iint f B^*_\parallel
            \left( v_\parallel \frac{\mathbf B^*}{B^*_\parallel} + \frac{\mathbf E \times \mathbf b_0}{B^*_\parallel}
            \right) \textrm d v_\parallel \textrm d \mu\,,

        Faraday's law:

        .. math::

            \frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla \times \mathbf E = 0\,.

        :math:`\mathbf B_0` is a static equilibrium field, :math:`n_0` the equilibrium density. The current in
        Ampère's law is the guiding-center current without the magnetic (grad-:math:`B`) drift, whose work on
        :math:`\mathbf E` is therefore missing where :math:`\nabla |B| \neq 0`. For :math:`\kappa = 0` (quasi-neutral
        limit) the parallel electric field is determined by the parallel current (Ohm's law with inertia), the
        perpendicular one by the polarization current.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the light speed as reference velocity:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v\,.

        The species parameters are :math:`\alpha=\hat\Omega_p/\hat\Omega_c` and
        :math:`\varepsilon=1/(\hat\Omega_c\hat t)`. The polarization coefficient :math:`\chi = \alpha^2` times
        :math:`n_0/|B_0|^2` is :math:`c^2/v_A^2` of the species."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy :math:`\frac 12 \|\mathbf E\|^2`: ``en_E``
        - Magnetic field energy :math:`\frac 12 \|\tilde{\mathbf B}\|^2`: ``en_B``
        - Parallel kinetic energy :math:`\alpha^2 \sum_p w_p v_{\parallel,p}^2/2`: ``en_f``
        - Total energy conserved by the coupling step,
          :math:`\frac{\kappa}{2} \| \mathbf E \|^2 + \frac{\chi}{2} \| \mathbf E_\perp \|^2_{n_0/|B_0|^2}
          + \frac{1}{2} \| \tilde{\mathbf B} \|^2 + \alpha^2 \sum_p w_p (v_{\parallel,p}^2/2 + \mu_p |B_0|)`:
          ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`
        2. :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`
        3. :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`
        """
        doc = rf"""**1. DriftKineticMaxwellCoupling:**

{DriftKineticMaxwellCoupling.__doc__}

**2. PushGuidingCenterParallel:**

{PushGuidingCenterParallel.__doc__}

**3. PushGuidingCenterBxEstar:**

{PushGuidingCenterBxEstar.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""DriftKineticMaxwell couples one drift-kinetic (guiding-center) species to Maxwell's equations in the
        gauge-invariant :math:`v_\parallel` formulation (the electric field, not a potential, enters
        :math:`\mathbf E^*`). Maxwell's equations and the parallel acceleration are solved together in one implicit,
        energy conserving step, which allows time steps above the plasma period and has no cancellation problem.
        Since the drift-kinetic ordering removes the polarization drift, its (linearized) polarization current is
        added to Ampère's law; with it, the quasi-neutral limit without displacement current is well-posed and gives
        the inertial Alfvén wave. The guiding centers then stream along the field and drift with
        :math:`\mathbf E^* \times \mathbf b_0`. The total energy is conserved exactly by the splitting if
        :math:`\nabla |B_0| = 0` and either :math:`\mu = 0` or :math:`\mathbf b_0 \cdot \tilde{\mathbf B} = 0`."""

    @classmethod
    def doc_examples(cls):
        r"""Create a drift-kinetic Maxwell model in the quasi-neutral limit:

        .. code-block:: python

            from struphy.models import DriftKineticMaxwell

            model = DriftKineticMaxwell()
            prop = model.propagators.coupling
            prop.options = prop.Options(kappa=0.0)
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic drift-kinetic simulations with time steps above the plasma period
        - shear (inertial) Alfvén waves with kinetic parallel dynamics
        - the quasi-neutral limit (``kappa=0.0``) with full-f weights"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - polar splines (not supported yet by the coupling propagator)
        - the quasi-neutral limit with control-variate (delta-f) weights or with vacuum regions
        - finite Larmor radius effects beyond the linearized long-wavelength polarization
        - exact energy conservation in inhomogeneous :math:`|B_0|` (the grad-:math:`B` drift current is missing)"""
