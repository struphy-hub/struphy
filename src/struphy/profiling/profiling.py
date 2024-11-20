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
    """Singleton class for managing global profiling configuration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.likwid = False  # Default value (profiling disabled)
            cls._instance.simulation_label = ''
            cls._instance.sample_duration = 0.1
            cls._instance.sample_interval = 1.0
        return cls._instance

    @property
    def likwid(self):
        return self._likwid

    @likwid.setter
    def likwid(self, value):
        self._likwid = value

    @property
    def simulation_label(self):
        return self._simulation_label

    @simulation_label.setter
    def simulation_label(self, value):
        self._simulation_label = value

    @property
    def sample_duration(self):
        return self._sample_duration

    @sample_duration.setter
    def sample_duration(self, value):
        self._sample_duration = value

    @property
    def sample_interval(self):
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, value):
        self._sample_interval = value


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

        # Save the likwid configuration data
        likwid_data = {}
        if ProfilingConfig().likwid:
            pylikwid = _import_pylikwid()

            # Gather LIKWID-specific information
            pylikwid.inittopology()
            likwid_data["cpu_info"] = pylikwid.getcpuinfo()
            likwid_data["cpu_topology"] = pylikwid.getcputopology()
            pylikwid.finalizetopology()

            likwid_data["numa_info"] = pylikwid.initnuma()
            pylikwid.finalizenuma()

            likwid_data["affinity_info"] = pylikwid.initaffinity()
            pylikwid.finalizeaffinity()

            pylikwid.initconfiguration()
            likwid_data["configuration"] = pylikwid.getconfiguration()
            pylikwid.destroyconfiguration()

            likwid_data["groups"] = pylikwid.getgroups()


        if rank == 0:
            # Combine the data from all processes
            combined_data = {f"rank_{i}": data for i, data in enumerate(all_data)}

            # Add the likwid data
            if likwid_data:
                combined_data["likwid_data"] = likwid_data

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
    """Context manager for profiling specific code regions using LIKWID markers."""

    def __init__(self, region_name):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.config = ProfilingConfig()
        self.region_name = self.config.simulation_label + region_name
        self._ncalls = 0
        self._start_times = []
        self._end_times = []
        self._durations = []
        self.started = False

    def __enter__(self):
        if self.config.likwid:
            self._pylikwid().markerstartregion(self.region_name)

        self._ncalls += 1
        self._start_time = MPI.Wtime()
        if self._start_time % self.config.sample_interval < self.config.sample_duration or self._ncalls == 1:
            self._start_times.append(self._start_time)
            self.started = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.config.likwid:
            self._pylikwid().markerstopregion(self.region_name)
        if self.started:
            end_time = MPI.Wtime()
            self._end_times.append(end_time)
            self._durations.append(end_time - self._start_time)
            self.started = False

    def _pylikwid(self):
        return _import_pylikwid()

    @property
    def ncalls(self):
        return self._ncalls

    @property
    def durations(self):
        return self._durations

    @property
    def start_times(self):
        return self._start_times

    @property
    def end_times(self):
        return self._end_times


def pylikwid_markerinit():
    """Initialize LIKWID profiling markers."""
    if ProfilingConfig().likwid:
        _import_pylikwid().markerinit()

def pylikwid_markerclose():
    """Close LIKWID profiling markers."""
    if ProfilingConfig().likwid:
        _import_pylikwid().markerclose()