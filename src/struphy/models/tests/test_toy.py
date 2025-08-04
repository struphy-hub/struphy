from struphy.models.tests.base_test import ModelTest
from dataclasses import dataclass

from struphy.models.toy import Maxwell
from struphy.io.options import EnvironmentOptions, Units, Time, DerhamOptions
from struphy.geometry.base import Domain
from struphy.fields_background.base import FluidEquilibrium
from struphy.topology.grids import TensorProductGrid
from struphy.geometry.domains import Cuboid
from struphy.fields_background.equils import HomogenSlab
from struphy.initial import perturbations


class LightWaveDispersion(ModelTest):
    """Simulate light wave dispersion relation in Cartesian coordinates (Cuboid mapping)."""
    
    def __init__(self, model: Maxwell):
        self.model: Maxwell = model
        self.units: Units = Units()
        self.time_opts: Time = Time(dt=0.05, Tend=100.0 - 1e-6)
        self.domain: Domain = Cuboid(r3=20.0)
        self.equil: FluidEquilibrium = None
        self.grid: TensorProductGrid = TensorProductGrid(Nel=(1, 1, 128))
        self.derham_opts: DerhamOptions = DerhamOptions(p=(1, 1, 3))
        
    def set_propagator_opts(self):
        self.model.propagators.maxwell.set_options()
        
    def set_variables_inits(self):
        self.model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=0))
        self.model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=1))
    
    def verification_script(self):
        assert -1 < 0