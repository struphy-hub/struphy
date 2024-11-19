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

import os
import pickle

# Import the profiling configuration class and context manager
from functools import lru_cache

import numpy as np
from mpi4py import MPI


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
            cls._instance.sample_duration = 0
            cls._instance.sample_interval = 0
        return cls._instance

    def set_likwid(self, value, simulation_label=''):
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

    def set_sample_duration(self, value):
        self.sample_duration = value

    def get_sample_duration(self):
        return self.sample_duration

    def set_sample_interval(self, value):
        self.sample_interval = value

    def get_sample_interval(self):
        return self.sample_interval

        set_sample_duration


class ProfileManager:
    """
    Singleton class to manage and track all ProfileRegion instances.
    """

    _regions = {}

    @classmethod
    def profile_region(cls, region_name):
        """
        Get an existing ProfileRegion by name, or create a new one if it doesn't exist.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion: The ProfileRegion instance.
        """
        if region_name in cls._regions:
            return cls._regions[region_name]
        else:
            # Create and register a new ProfileRegion
            new_region = ProfileRegion(region_name)
            cls._regions[region_name] = new_region
            return new_region

    @classmethod
    def get_region(cls, region_name):
        """
        Get a registered ProfileRegion by name.

        Parameters
        ----------
        region_name: str
            The name of the profiling region.

        Returns
        -------
        ProfileRegion or None: The registered ProfileRegion instance or None if not found.
        """
        return cls._regions.get(region_name)

    @classmethod
    def get_all_regions(cls):
        """
        Get all registered ProfileRegion instances.

        Returns
        -------
        dict: Dictionary of all registered ProfileRegion instances.
        """
        return cls._regions

    @classmethod
    def save_to_pickle(cls, file_path):
        """
        Save profiling data to a single file using pickle and NumPy arrays in parallel.

        Parameters
        ----------
        file_path: str
            Path to the file where data will be saved.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Prepare the data to be gathered
        local_data = {}
        for name, region in cls._regions.items():
            local_data[name] = {
                "ncalls": region.ncalls,
                "durations": np.array(region.durations, dtype=np.float64),
                "start_times": np.array(region.start_times, dtype=np.float64),
                "end_times": np.array(region.end_times, dtype=np.float64),
                "config": {
                    "likwid": region.config.likwid,
                    "simulation_label": region.config.simulation_label,
                    "sample_duration": region.config.sample_duration,
                    "sample_interval": region.config.sample_interval,
                },
            }

        # Gather all data at the root process (rank 0)
        all_data = comm.gather(local_data, root=0)

        if rank == 0:
            # Combine the data from all processes
            combined_data = {f"rank_{i}": data for i, data in enumerate(all_data)}

            # Convert the file path to an absolute path
            absolute_path = os.path.abspath(file_path)

            # Save the combined data using pickle
            with open(absolute_path, "wb") as file:
                pickle.dump(combined_data, file)

            print(f"Data saved to {absolute_path}")

    @classmethod
    def print_summary(cls):
        """
        Print a summary of the profiling data for all regions.
        """

        print("Profiling Summary:")
        print("=" * 40)
        for name, region in cls._regions.items():
            if region.ncalls > 0:
                total_duration = sum(region.durations)
                average_duration = total_duration / region.ncalls
                min_duration = min(region.durations)
                max_duration = max(region.durations)
                std_duration = np.std(region.durations)
            else:
                total_duration = average_duration = min_duration = max_duration = std_duration = 0

            print(f"Region: {name}")
            print(f"  Number of Calls: {region.ncalls}")
            print(f"  Total Duration: {total_duration:.6f} seconds")
            print(f"  Average Duration: {average_duration:.6f} seconds")
            print(f"  Min Duration: {min_duration:.6f} seconds")
            print(f"  Max Duration: {max_duration:.6f} seconds")
            print(f"  Std Deviation: {std_duration:.6f} seconds")
            print("-" * 40)


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

        if hasattr(self, '_initialized') and self._initialized:
            return  # Skip re-initialization

        self.config = ProfilingConfig()
        # By default, self.config.simulation_label = ''
        # --> self.region_name = region_name
        self.region_name = self.config.simulation_label + region_name
        self._ncalls = 0
        self._start_times = []
        self._end_times = []
        self._durations = []
        self.started = False

    def __enter__(self):
        """
        Enter the profiling context, starting the LIKWID marker if profiling is enabled.

        Returns:
            ProfileRegion: The current instance of ProfileRegion.
        """
        if self.config.get_likwid():
            self._pylikwid().markerstartregion(self.region_name)

        self._ncalls += 1
        self._start_time = MPI.Wtime()
        if self._start_time % self.config.sample_interval < self.config.sample_duration or self._ncalls == 1:
            self._start_times.append(self._start_time)
            self.started = True
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
        if self.started:
            end_time = MPI.Wtime()
            self._end_times.append(end_time)
            self._durations.append(end_time - self._start_time)
            self.started = False

    def _pylikwid(self):
        """
        Import and return the pylikwid module, caching the result to avoid repeated imports.

        Returns:
            module: The pylikwid module.
        """
        return _import_pylikwid()

    @property
    def ncalls(self):
        """
        Get the number of times the region has been entered.

        Returns:
            int: Number of calls to this profiling region.
        """
        return self._ncalls

    @property
    def durations(self):
        """
        Get the list of durations for each call to the profiling region.

        Returns:
            list: Durations of each call in seconds.
        """
        return self._durations

    @property
    def start_times(self):
        """
        Get the list of start times for each call to the profiling region.

        Returns:
            list: Start times of each call in seconds.
        """
        return self._start_times

    @property
    def end_times(self):
        """
        Get the list of end times for each call to the profiling region.

        Returns:
            list: End times of each call in seconds.
        """
        return self._end_times


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


def set_sample_duration(value):
    return ProfilingConfig().set_sample_duration(value)


def get_sample_duration():
    return ProfilingConfig().get_sample_duration()


def set_sample_interval(value):
    return ProfilingConfig().set_sample_interval(value)


def get_sample_interval():
    return ProfilingConfig().get_sample_interval()
