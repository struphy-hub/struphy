from abc import ABCMeta, abstractmethod

class Simulation(metaclass=ABCMeta):
    """Abstract base class for simulations."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the simulation."""
        pass
    
    @abstractmethod
    def allocate(self, verbose: bool = False):
        """Allocate the simulation variables in memory."""
        pass

    @abstractmethod
    def save_geometry_and_equil_vtk(self, verbose: bool = False):
        """Save geometry and equilibrium in VTK format."""
        pass
    
    @abstractmethod
    def initialize_data(self, verbose: bool = False):
        """Initialize the simulation data storage."""
        pass

    @abstractmethod
    def run(self, verbose: bool = False):
        """Run the simulation."""
        pass