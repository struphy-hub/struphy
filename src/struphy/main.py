import copy
import datetime
import glob
import os
import pickle
import shutil
import sysconfig
import time
from typing import Optional, TypedDict

import cunumpy as xp
import h5py
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile
from pyevtk.hl import gridToVTK
from scope_profiler import ProfileManager

from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB
from struphy.fields_background.equils import HomogenSlab
from struphy.geometry import domains
from struphy.geometry.base import Domain
from struphy.io.options import BaseUnits, DerhamOptions, EnvironmentOptions, Time
from struphy.io.output_handling import DataContainer
from struphy.io.setup import import_parameters_py
from struphy.models.base import StruphyModel
from struphy.models.species import Species
from struphy.models.variables import FEECVariable
from struphy.physics.physics import Units
from struphy.pic.base import Particles
from struphy.post_processing.orbits import orbits_tools
from struphy.simulation.sim import Simulation
from struphy.topology import grids
from struphy.topology.grids import TensorProductGrid
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml


@profile
def run(
    model: StruphyModel,
    *,
    params_path: str = None,
    env: EnvironmentOptions = EnvironmentOptions(),
    base_units: BaseUnits = BaseUnits(),
    time_opts: Time = Time(),
    domain: Domain = domains.Cuboid(),
    equil: FluidEquilibrium = HomogenSlab(),
    grid: TensorProductGrid = None,
    derham_opts: DerhamOptions = None,
    verbose: bool = False,
):
    """
    Run a Struphy model.

    Parameters
    ----------
    model : StruphyModel
        The model to run. Check https://struphy-hub.github.io/struphy/sections/models.html for available models.

    params_path : str
        Absolute path to .py parameter file.
    """

    sim = Simulation(
        model=model,
        params_path=params_path,
        env=env,
        base_units=base_units,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        verbose=verbose,
    )

    sim.run(verbose=verbose)


class SimData:
    """Holds post-processed Struphy data as attributes.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.
    """

    def __init__(self, path: str):
        self.path = path
        self._orbits = {}
        self._f = {}
        self._spline_values = {}
        self._n_sph = {}
        self.grids_log: list[xp.ndarray] = None
        self.grids_phy: list[xp.ndarray] = None
        self.t_grid: xp.ndarray = None

    @property
    def orbits(self) -> dict[str, xp.ndarray]:
        """Keys: species name. Values: 3d arrays indexed by (n, p, a), where 'n' is the time index, 'p' the particle index and 'a' the attribute index."""
        return self._orbits

    @property
    def f(self) -> dict[str, dict[str, dict[str, xp.ndarray]]]:
        """Keys: species name. Values: dicts of slice names ('e1_v1' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._f

    @property
    def spline_values(self) -> dict[str, dict[str, xp.ndarray]]:
        """Keys: species name. Values: dicts of variable names with values being 3d arrays on the grid."""
        return self._spline_values

    @property
    def n_sph(self) -> dict[str, dict[str, dict[str, xp.ndarray]]]:
        """Keys: species name. Values: dicts of view names ('view_0' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._n_sph

    @property
    def Nt(self) -> dict[str, int]:
        """Number of available time points (snap shots) for each species."""
        if not hasattr(self, "_Nt"):
            self._Nt = {}
            for spec, orbs in self.orbits.items():
                self._Nt[spec] = orbs.shape[0]
        return self._Nt

    @property
    def Np(self) -> dict[str, int]:
        """Number of particle orbits for each species."""
        if not hasattr(self, "_Np"):
            self._Np = {}
            for spec, orbs in self.orbits.items():
                self._Np[spec] = orbs.shape[1]
        return self._Np

    @property
    def Nattr(self) -> dict[str, int]:
        """Number of particle attributes for each species."""
        if not hasattr(self, "_Nattr"):
            self._Nattr = {}
            for spec, orbs in self.orbits.items():
                self._Nattr[spec] = orbs.shape[2]
        return self._Nattr


def load_data(path: str) -> SimData:
    """Load data generated during post-processing.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.
    """

    path_pproc = os.path.join(path, "post_processing")
    assert os.path.exists(path_pproc), f"Path {path_pproc} does not exist, run 'pproc' first?"
    print("\n*** Loading post-processed simulation data:")
    print(f"{path =}")

    simdata = SimData(path)

    # load time grid
    simdata.t_grid = xp.load(os.path.join(path_pproc, "t_grid.npy"))

    # data paths
    path_fields = os.path.join(path_pproc, "fields_data")
    path_kinetic = os.path.join(path_pproc, "kinetic_data")

    # load point data
    if os.path.exists(path_fields):
        # grids
        with open(os.path.join(path_fields, "grids_log.bin"), "rb") as f:
            simdata.grids_log = pickle.load(f)
        with open(os.path.join(path_fields, "grids_phy.bin"), "rb") as f:
            simdata.grids_phy = pickle.load(f)

        # species folders
        species = next(os.walk(path_fields))[1]
        for spec in species:
            simdata._spline_values[spec] = {}
            # simdata.arrays[spec] = {}
            path_spec = os.path.join(path_fields, spec)
            wlk = os.walk(path_spec)
            files = next(wlk)[2]
            print(f"\nFiles in {path_spec}: {files}")
            for file in files:
                if ".bin" in file:
                    var = file.split(".")[0]
                    with open(os.path.join(path_spec, file), "rb") as f:
                        # try:
                        simdata._spline_values[spec][var] = pickle.load(f)
                        # simdata.arrays[spec][var] = pickle.load(f)

    if os.path.exists(path_kinetic):
        # species folders
        species = next(os.walk(path_kinetic))[1]
        print(f"{species =}")
        for spec in species:
            path_spec = os.path.join(path_kinetic, spec)
            wlk = os.walk(path_spec)
            sub_folders = next(wlk)[1]
            for folder in sub_folders:
                path_dat = os.path.join(path_spec, folder)
                sub_wlk = os.walk(path_dat)

                if "orbits" in folder:
                    files = next(sub_wlk)[2]
                    Nt = len(files) // 2
                    n = 0
                    for file in files:
                        # print(f"{file = }")
                        if ".npy" in file:
                            step = int(file.split(".")[0].split("_")[-1])
                            tmp = xp.load(os.path.join(path_dat, file))
                            if n == 0:
                                simdata._orbits[spec] = xp.zeros((Nt, *tmp.shape), dtype=float)
                            simdata._orbits[spec][step] = tmp
                            n += 1

                elif "distribution_function" in folder:
                    simdata._f[spec] = {}
                    slices = next(sub_wlk)[1]
                    # print(f"{slices = }")
                    for sli in slices:
                        simdata._f[spec][sli] = {}
                        # print(f"{sli = }")
                        files = next(sub_wlk)[2]
                        # print(f"{files = }")
                        for file in files:
                            name = file.split(".")[0]
                            tmp = xp.load(os.path.join(path_dat, sli, file))
                            # print(f"{name = }")
                            simdata._f[spec][sli][name] = tmp

                elif "n_sph" in folder:
                    simdata._n_sph[spec] = {}
                    slices = next(sub_wlk)[1]
                    # print(f"{slices = }")
                    for sli in slices:
                        simdata._n_sph[spec][sli] = {}
                        # print(f"{sli = }")
                        files = next(sub_wlk)[2]
                        # print(f"{files = }")
                        for file in files:
                            name = file.split(".")[0]
                            tmp = xp.load(os.path.join(path_dat, sli, file))
                            # print(f"{name = }")
                            simdata._n_sph[spec][sli][name] = tmp

                else:
                    print(f"{folder =}")
                    raise NotImplementedError

    print("\nThe following data has been loaded:")
    print("\ngrids:")
    print(f"{simdata.t_grid.shape =}")
    if simdata.grids_log is not None:
        print(f"{simdata.grids_log[0].shape =}")
        print(f"{simdata.grids_log[1].shape =}")
        print(f"{simdata.grids_log[2].shape =}")
    if simdata.grids_phy is not None:
        print(f"{simdata.grids_phy[0].shape =}")
        print(f"{simdata.grids_phy[1].shape =}")
        print(f"{simdata.grids_phy[2].shape =}")
    print("\nsimdata.spline_values:")
    for k, v in simdata.spline_values.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
    print("\nsimdata.orbits:")
    for k, v in simdata.orbits.items():
        print(f"  {k}")
    print("\nsimdata.f:")
    for k, v in simdata.f.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
            for kkk, vvv in vv.items():
                print(f"      {kkk}")
    print("\nsimdata.n_sph:")
    for k, v in simdata.n_sph.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
            for kkk, vvv in vv.items():
                print(f"      {kkk}")

    return simdata
