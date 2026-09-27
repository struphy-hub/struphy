import json
from abc import ABCMeta, abstractmethod

from struphy.utils.utils import dict_to_yaml


class SimulationBase(metaclass=ABCMeta):
    """Abstract base class for simulations."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the simulation."""
        pass

    @abstractmethod
    def allocate(self):
        """Allocate the simulation variables in memory."""
        pass

    @abstractmethod
    def save_geometry_and_equil_vtk(self):
        """Save geometry and equilibrium in VTK format."""
        pass

    @abstractmethod
    def initialize_data_storage(self):
        """Initialize the simulation data storage."""
        pass

    @abstractmethod
    def run(self):
        """Run the simulation."""
        pass

    @abstractmethod
    def pproc(self):
        """Post-process the simulation results."""
        pass

    @abstractmethod
    def load_plotting_data(self):
        """Load post-processed data for visualization."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the simulation configuration to a dictionary."""
        pass

    @abstractmethod
    def from_dict(cls, dct: dict):
        """Deserialize a simulation configuration from a dictionary."""
        pass

    @abstractmethod
    def from_file(cls, file_path: str):
        """Deserialize a simulation configuration from a file."""
        pass

    def export(self, file_path: str):
        """Export a simulation configuration to a YAML or JSON file based on the file extension."""
        dct = self.to_dict()
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            dict_to_yaml(dct, file_path)
        elif file_path.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(dct, f, indent=4)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml or .json.")
