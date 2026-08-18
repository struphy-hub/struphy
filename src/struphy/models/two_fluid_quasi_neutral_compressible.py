import copy

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.two_fluid_quasi_neutral_compressible import TwoFluidQuasiNeutralCompressible


class TwoFluidQuasiNeutral(StruphyModel):

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    class EMfields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class Ions(FluidSpecies):
        def __init__(self, charge_number=1, mass_number=1.0, epsilon=None):
            self.u = FEECVariable(space="Hcurl")
            self.init_variables(charge_number=charge_number, mass_number=mass_number, epsilon=epsilon)

    class Electrons(FluidSpecies):
        def __init__(self, charge_number=1, mass_number=1.0, epsilon=None):
            self.u = FEECVariable(space="Hcurl")
            self.init_variables(charge_number=charge_number, mass_number=mass_number, epsilon=epsilon)

    class Propagators:
        def __init__(self):
            self.qn_comp = TwoFluidQuasiNeutralCompressible()

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        ion_charge_number: int = 1,
        ion_mass_number: float = 1.0,
        ion_epsilon: float = None,
        electron_charge_number: int = 1,
        electron_mass_number: float = 1.0,
        electron_epsilon: float = None,
    ):
        self.params = copy.deepcopy(locals())

        self.em_fields = self.EMfields()
        self.ions = self.Ions(charge_number=ion_charge_number, mass_number=ion_mass_number, epsilon=ion_epsilon)
        self.electrons = self.Electrons(charge_number=electron_charge_number, mass_number=electron_mass_number, epsilon=electron_epsilon)

        self.setup_equation_params(base_units=base_units)

        self.propagators = self.Propagators()

        self.propagators.qn_comp.variables.u = self.ions.u
        self.propagators.qn_comp.variables.ue = self.electrons.u
        self.propagators.qn_comp.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.ions

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        pass