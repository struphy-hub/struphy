"""Electrostatic ion-optics test-particle model.

This is the first building block for self-consistent ion optics.  It advances
an injected ion population in a prescribed electrostatic field; space-charge
deposition and the nonlinear Poisson iteration are intentionally not part of
this first milestone.
"""

import copy

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergyPIC, Scalars
from struphy.models.species import FieldSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_v_in_force_field import PushVinForceField


class IonOpticsElectrostatic(StruphyModel):
    r"""Ions moving in a prescribed electrostatic field.

    The model advances

    .. math::

        \dot{\mathbf x} = \mathbf v, \qquad
        \dot{\mathbf v} = \mathbf E(\mathbf x)/\varepsilon,

    with :math:`\mathbf E=-\nabla\phi`.  Initialize ``phi`` through
    Struphy's normal initial-condition path; the model constructs the FEEC
    electric field from it after allocation.

    This intentionally contains no field evolution and no particle charge
    deposition.  It is the verification baseline for the subsequent
    self-consistent Poisson/space-charge model.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class Ions(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    class Propagators:
        def __init__(self, phi: FEECVariable):
            self.push_v = PushVinForceField(potential=phi)
            self.push_eta = PushEta()

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):
        self.params = copy.deepcopy(locals())
        self.em_fields = self.EMFields()
        self.ions = self.Ions(
            charge_number=charge_number,
            mass_number=mass_number,
            alpha=alpha,
            epsilon=epsilon,
        )
        self.setup_equation_params(base_units=base_units)

        self.propagators = self.Propagators(phi=self.em_fields.phi)
        self.propagators.push_v.variables.var = self.ions.var
        self.propagators.push_eta.variables.var = self.ions.var

        self.scalars = Scalars(kinetic_energy=KineticEnergyPIC(self.ions.var))

    @property
    def bulk_species(self):
        return self.ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def post_allocate(self):
        Propagator.derham.grad.dot(
            -self.em_fields.phi.spline.vector,
            out=self.em_fields.e_field.spline.vector,
        )

    @classmethod
    def doc_pde(cls):
        return cls.__doc__

    @classmethod
    def doc_normalization(cls):
        return """The model uses the ion cyclotron normalization associated with ``BaseUnits``."""

    @classmethod
    def doc_scalar_quantities(cls):
        return """**The following scalars are tracked:**

        - Particle kinetic energy."""

    @classmethod
    def doc_discretization(cls):
        return """Time integration applies ``PushVinForceField`` followed by ``PushEta``."""

    @classmethod
    def doc_long_description(cls):
        return """A prescribed-field ion-optics baseline. Space charge is deliberately deferred to a later model milestone."""

    @classmethod
    def doc_examples(cls):
        return """Create the model with ``IonOpticsElectrostatic()`` and initialize ``em_fields.phi`` and ``ions.var``."""

    @classmethod
    def doc_use_cases(cls):
        return """Prescribed-field extraction, acceleration and focusing tests."""

    @classmethod
    def doc_cannot_be_used_for(cls):
        return """Self-consistent space charge, electrode solves and particle–wall interactions are not yet included."""
