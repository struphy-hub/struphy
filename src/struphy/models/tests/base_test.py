from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from struphy.io.options import EnvironmentOptions, Units, Time, DerhamOptions
from struphy.topology.grids import TensorProductGrid
from struphy.geometry.base import Domain
from struphy.geometry.domains import Cuboid
from struphy.fields_background.equils import HomogenSlab
from struphy.pic import particles
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.kinetic_background import maxwellians
from struphy.fields_background.base import FluidEquilibrium
from struphy.models.base import StruphyModel


@dataclass
class ModelTest(metaclass=ABCMeta):
    """Base class for model verification tests."""
    model: StruphyModel = None
    units: Units = None
    time_opts: Time = None
    domain: Domain = None
    equil: FluidEquilibrium = None
    grid: TensorProductGrid = None
    derham_opts: DerhamOptions = None
    
    @abstractmethod
    def set_propagator_opts(self):
        """Set propagator options for test."""
        
    @abstractmethod
    def set_variables_inits(self):
        """Set initial conditions for test."""
        
    @abstractmethod
    def verification_script(self):
        """Contains assert statements (and plots) that verify the test."""