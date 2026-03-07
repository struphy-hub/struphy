import json
from abc import ABCMeta, abstractmethod

import yaml


class SimulationBase(metaclass=ABCMeta):
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
    def initialize_data_storage(self, verbose: bool = False):
        """Initialize the simulation data storage."""
        pass

    @abstractmethod
    def run(self, verbose: bool = False):
        """Run the simulation."""
        pass

    @abstractmethod
    def pproc(self, verbose: bool = False):
        """Post-process the simulation results."""
        pass

    @abstractmethod
    def load_plotting_data(self, verbose: bool = False):
        """Load post-processed data for visualization."""
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

    @classmethod
    def from_file(cls, file_path: str) -> "Simulation":
        """Deserialize a simulation configuration from a file based on the file extension."""
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            with open(file_path, "r") as f:
                dct = yaml.safe_load(f)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                dct = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml or .json.")
        return cls.from_dict(dct)
