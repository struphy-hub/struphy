"""
profiling.py

This module provides a centralized profiling configuration and management system
using LIKWID markers. It includes:
- A singleton class for managing profiling configuration.
- A context manager for profiling specific code regions.
- Initialization and cleanup functions for LIKWID markers.
- Convenience functions for setting and getting the profiling configuration.

LIKWID is imported only when profiling is enabled to avoid unnecessary overhead.
"""

# Import the profiling configuration class and context manager
from functools import lru_cache

@lru_cache(maxsize=None)  # Cache the import result to avoid repeated imports
def _import_pylikwid():
    import pylikwid
    return pylikwid

class ProfilingConfig:
    """
    Singleton class for managing global profiling configuration.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.likwid = False  # Default value (profiling disabled)
            cls._instance.simulation_label = ''
        return cls._instance

    def set_likwid(self, value, simulation_label = ''):
        """
        Set the profiling flag to enable or disable profiling.
        
        Parameters
        ----------
        value: bool
            True to enable profiling, False to disable.
        """
        self.likwid = value
    
    def set_simulation_label(self, value):
        """
        Set the label for the simulation. When profiling a region,
        the region_name will be appended to the label. 

        Parameters
        ----------
        value: str
            Label name
        """
        
        self.simulation_label = value

    def get_likwid(self):
        """
        Get the current profiling configuration.

        Returns:
            bool: True if profiling is enabled, False otherwise.
        """
        return self.likwid

class ProfileRegion:
    """
    Context manager for profiling specific code regions using LIKWID markers.

    Attributes:
    ----------
    region_name: str
        Name of the profiling region.
    
    config: ProfilingConfig
        Instance of ProfilingConfig for accessing profiling settings.
    """

    def __init__(self, region_name):
        """
        Initialize the ProfileRegion context manager.

        Parameters
        ----------
        region_name: str
            Name of the profiling region.
        """
        
        self.config = ProfilingConfig()
        # By default, self.config.simulation_label = ''
        # --> self.region_name = region_name
        self.region_name = self.config.simulation_label + region_name
        
    def __enter__(self):
        """
        Enter the profiling context, starting the LIKWID marker if profiling is enabled.

        Returns:
            ProfileRegion: The current instance of ProfileRegion.
        """
        if self.config.get_likwid():
            self._pylikwid().markerstartregion(self.region_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the profiling context, stopping the LIKWID marker if profiling is enabled.

        Args:
            exc_type (type): The exception type, if any.
            exc_value (Exception): The exception value, if any.
            traceback (traceback): The traceback object, if any.
        """
        if self.config.get_likwid():
            self._pylikwid().markerstopregion(self.region_name)

    def _pylikwid(self):
        """
        Import and return the pylikwid module, caching the result to avoid repeated imports.

        Returns:
            module: The pylikwid module.
        """
        return _import_pylikwid()

def pylikwid_markerinit():
    """
    Initialize LIKWID profiling markers.
    
    This function imports pylikwid only if profiling is enabled.
    """
    if ProfilingConfig().get_likwid():
        _import_pylikwid().markerinit()

def pylikwid_markerclose():
    """
    Close LIKWID profiling markers.
    
    This function imports pylikwid only if profiling is enabled.
    """
    if ProfilingConfig().get_likwid():
        _import_pylikwid().markerclose()

def set_likwid(value):
    """
    Set the global profiling configuration.

    Parameters
    ----------
    value: bool
        True to enable profiling, False to disable.
    """
    ProfilingConfig().set_likwid(value)

def set_simulation_label(value):
    """
    Set the simulation label

    Parameters
    ----------
    value: str
        Simulation label
    """
    # This allows  for running multiple simulations with different labels but where the regions have the same name.
    ProfilingConfig().set_simulation_label(value)

def get_likwid():
    """
    Get the current global profiling configuration.

    Returns:
        bool: True if profiling is enabled, False otherwise.
    """
    return ProfilingConfig().get_likwid()
