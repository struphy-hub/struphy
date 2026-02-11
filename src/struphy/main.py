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
from struphy.io.setup import import_parameters_py, setup_folders
from struphy.models.base import StruphyModel
from struphy.models.species import Species
from struphy.models.variables import FEECVariable
from struphy.physics.physics import Units
from struphy.pic.base import Particles
from struphy.post_processing.orbits import orbits_tools
from struphy.post_processing.post_processing_tools import (
    create_femfields,
    create_vtk,
    eval_femfields,
    get_params_of_run,
    post_process_f,
    post_process_markers,
    post_process_n_sph,
)
from struphy.topology import grids
from struphy.topology.grids import TensorProductGrid
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml

from struphy.simulation.sim import StruphySimulation


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

    sim = StruphySimulation(
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

def pproc(
    path: str,
    *,
    step: int = 1,
    celldivide: int = 1,
    physical: bool = False,
    guiding_center: bool = False,
    classify: bool = False,
    no_vtk: bool = False,
    time_trace: bool = False,
):
    """Post-processing finished Struphy runs.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.

    step : int
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    celldivide : int
        Grid refinement in evaluation of FEM fields. E.g. celldivide=2 evaluates two points per grid cell.

    physical : bool
        Wether to do post-processing into push-forwarded physical (xyz) components of fields.

    guiding_center : bool
        Compute guiding-center coordinates (only from Particles6D).

    classify : bool
        Classify guiding-center trajectories (passing, trapped or lost).

    no_vtk : bool
        whether vtk files creation should be skipped

    time_trace : bool
        whether to plot the time trace of each measured region
    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n*** Start post-processing of {path}:")

    # import parameters
    params_in = get_params_of_run(path)
    model = params_in.model
    domain = params_in.domain

    # create post-processing folder
    path_pproc = os.path.join(path, "post_processing")

    try:
        os.mkdir(path_pproc)
    except:
        shutil.rmtree(path_pproc)
        os.mkdir(path_pproc)

    if time_trace:
        from struphy.post_processing.likwid.plot_time_traces import plot_gantt_chart, plot_time_vs_duration

        path_time_trace = os.path.join(path, "profiling_time_trace.pkl")
        plot_time_vs_duration(path_time_trace, output_path=path_pproc)
        plot_gantt_chart(path_time_trace, output_path=path_pproc)
        return

    # check for fields and kinetic data in hdf5 file that need post processing
    with h5py.File(os.path.join(path, "data/", "data_proc0.hdf5"), "r") as file:
        # save time grid at which post-processing data is created
        xp.save(os.path.join(path_pproc, "t_grid.npy"), file["time/value"][::step].copy())

        if "feec" in file.keys():
            exist_fields = True
        else:
            exist_fields = False

        if "kinetic" in file.keys():
            exist_kinetic = {"markers": False, "f": False, "n_sph": False}
            kinetic_species = []
            kinetic_kinds = []
            for name in file["kinetic"].keys():
                kinetic_species += [name]
                kinetic_kinds += [next(iter(model.species[name].variables.values())).space]

                # check for saved markers
                if "markers" in file["kinetic"][name]:
                    exist_kinetic["markers"] = True
                # check for saved distribution function
                if "f" in file["kinetic"][name]:
                    exist_kinetic["f"] = True
                # check for saved sph density
                if "n_sph" in file["kinetic"][name]:
                    exist_kinetic["n_sph"] = True
        else:
            exist_kinetic = None

    # field post-processing
    if exist_fields:
        fields, t_grid = create_femfields(path, params_in=params_in, step=step)

        point_data, grids_log, grids_phy = eval_femfields(params_in, fields, celldivide=[celldivide] * 3)

        if physical:
            point_data_phy, grids_log, grids_phy = eval_femfields(
                params_in,
                fields,
                celldivide=[celldivide] * 3,
                physical=True,
            )

        # directory for field data
        path_fields = os.path.join(path_pproc, "fields_data")

        try:
            os.mkdir(path_fields)
        except:
            shutil.rmtree(path_fields)
            os.mkdir(path_fields)

        # save data dicts for each field
        for species, vars in point_data.items():
            for name, val in vars.items():
                try:
                    os.mkdir(os.path.join(path_fields, species))
                except:
                    pass

                with open(os.path.join(path_fields, species, name + "_log.bin"), "wb") as handle:
                    pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if physical:
                    with open(os.path.join(path_fields, species, name + "_phy.bin"), "wb") as handle:
                        pickle.dump(point_data_phy[species][name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save grids
        with open(os.path.join(path_fields, "grids_log.bin"), "wb") as handle:
            pickle.dump(grids_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path_fields, "grids_phy.bin"), "wb") as handle:
            pickle.dump(grids_phy, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create vtk files
        if not no_vtk:
            create_vtk(path_fields, t_grid, grids_phy, point_data)
            if physical:
                create_vtk(path_fields, t_grid, grids_phy, point_data_phy, physical=True)

    # kinetic post-processing
    if exist_kinetic is not None:
        # directory for kinetic data
        path_kinetics = os.path.join(path_pproc, "kinetic_data")

        try:
            os.mkdir(path_kinetics)
        except:
            shutil.rmtree(path_kinetics)
            os.mkdir(path_kinetics)

        # kinetic post-processing for each species
        for n, species in enumerate(kinetic_species):
            # directory for each species
            path_kinetics_species = os.path.join(path_kinetics, species)

            try:
                os.mkdir(path_kinetics_species)
            except:
                shutil.rmtree(path_kinetics_species)
                os.mkdir(path_kinetics_species)

            # markers
            if exist_kinetic["markers"]:
                post_process_markers(
                    path,
                    path_kinetics_species,
                    species,
                    domain,
                    kinetic_kinds[n],
                    step,
                )

                if guiding_center:
                    assert kinetic_kinds[n] == "Particles6D"
                    orbits_tools.post_process_orbit_guiding_center(path, path_kinetics_species, species)

                if classify:
                    orbits_tools.post_process_orbit_classification(path_kinetics_species, species)

            # distribution function
            if exist_kinetic["f"]:
                if kinetic_kinds[n] == "DeltaFParticles6D":
                    compute_bckgr = True
                else:
                    compute_bckgr = False

                post_process_f(
                    path,
                    params_in,
                    path_kinetics_species,
                    species,
                    step,
                    compute_bckgr=compute_bckgr,
                )

            # sph density
            if exist_kinetic["n_sph"]:
                post_process_n_sph(
                    path,
                    params_in,
                    path_kinetics_species,
                    species,
                    step,
                )


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
