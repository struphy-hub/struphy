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
import numpy as np
from line_profiler import profile
from psydac.ddm.mpi import MockMPI
from psydac.ddm.mpi import mpi as MPI
from pyevtk.hl import gridToVTK

from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB
from struphy.fields_background.equils import HomogenSlab
from struphy.geometry import domains
from struphy.geometry.base import Domain
from struphy.io.options import BaseUnits, DerhamOptions, EnvironmentOptions, Time, Units
from struphy.io.output_handling import DataContainer
from struphy.io.setup import import_parameters_py, setup_folders
from struphy.models.base import StruphyModel
from struphy.models.species import Species
from struphy.models.variables import FEECVariable
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
from struphy.profiling.profiling import ProfileManager
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
        The model to run. Check https://struphy.pages.mpcdf.de/struphy/sections/models.html for available models.

    params_path : str
        Absolute path to .py parameter file.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_simulation = time.time()

    # check model
    assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
    model_name = model.__class__.__name__
    model.verbose = verbose

    if rank == 0:
        print(f"\n*** Starting run for model '{model_name}':")

    # meta-data
    path_out = env.path_out
    restart = env.restart
    max_runtime = env.max_runtime
    save_step = env.save_step
    sort_step = env.sort_step
    num_clones = env.num_clones
    use_mpi = (not comm is None,)

    meta = {}
    meta["platform"] = sysconfig.get_platform()
    meta["python version"] = sysconfig.get_python_version()
    meta["model name"] = model_name
    meta["parameter file"] = params_path
    meta["output folder"] = path_out
    meta["MPI processes"] = size
    meta["number of domain clones"] = num_clones
    meta["restart"] = restart
    meta["max wall-clock [min]"] = max_runtime
    meta["save interval [steps]"] = save_step

    if rank == 0:
        print("\nMETADATA:")
        for k, v in meta.items():
            print(f"{k}:".ljust(25), v)

    # creating output folders
    setup_folders(
        path_out=path_out,
        restart=restart,
        verbose=verbose,
    )

    # add derived units
    units = Units(base_units)

    # save parameter file
    if rank == 0:
        # save python param file
        if params_path is not None:
            assert params_path[-3:] == ".py"
            shutil.copy2(
                params_path,
                os.path.join(path_out, "parameters.py"),
            )
        # pickle struphy objects
        else:
            with open(os.path.join(path_out, "env.bin"), "wb") as f:
                pickle.dump(env, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "base_units.bin"), "wb") as f:
                pickle.dump(base_units, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "time_opts.bin"), "wb") as f:
                pickle.dump(time_opts, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "domain.bin"), "wb") as f:
                # WORKAROUND: cannot pickle pyccelized classes at the moment
                tmp_dct = {"name": domain.__class__.__name__, "params": domain.params}
                pickle.dump(tmp_dct, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "equil.bin"), "wb") as f:
                # WORKAROUND: cannot pickle pyccelized classes at the moment
                if equil is not None:
                    tmp_dct = {"name": equil.__class__.__name__, "params": equil.params}
                else:
                    tmp_dct = {}
                pickle.dump(tmp_dct, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "grid.bin"), "wb") as f:
                pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "derham_opts.bin"), "wb") as f:
                pickle.dump(derham_opts, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path_out, "model_class.bin"), "wb") as f:
                pickle.dump(model.__class__, f, pickle.HIGHEST_PROTOCOL)

    # config clones
    if comm is None:
        clone_config = None
    else:
        if num_clones == 1:
            clone_config = None
        else:
            # Setup domain cloning communicators
            # MPI.COMM_WORLD     : comm
            # within a clone:    : sub_comm
            # between the clones : inter_comm
            clone_config = CloneConfig(comm=comm, params=None, num_clones=num_clones)
            clone_config.print_clone_config()
            if model.particle_species:
                clone_config.print_particle_config()

    model.clone_config = clone_config
    Barrier()

    ## configure model instance

    # units
    model.units = units
    if model.bulk_species is None:
        A_bulk = None
        Z_bulk = None
    else:
        A_bulk = model.bulk_species.mass_number
        Z_bulk = model.bulk_species.charge_number
    model.units.derive_units(
        velocity_scale=model.velocity_scale,
        A_bulk=A_bulk,
        Z_bulk=Z_bulk,
        verbose=verbose,
    )

    # domain and fluid background
    model.setup_domain_and_equil(domain, equil)

    # feec
    model.allocate_feec(grid, derham_opts)

    # equation paramters
    model.setup_equation_params(units=model.units, verbose=verbose)

    # allocate variables
    model.allocate_variables(verbose=verbose)
    model.allocate_helpers()

    # pass info to propagators
    model.allocate_propagators()

    # plasma parameters
    model.compute_plasma_params(verbose=verbose)

    if rank < 32:
        if rank == 0:
            print("")
        Barrier()
        print(f"Rank {rank}: executing main.run() for model {model_name} ...")

    if size > 32 and rank == 32:
        print(f"Ranks > 31: executing main.run() for model {model_name} ...")

    # store geometry vtk
    if rank == 0:
        grids_log = [
            xp.linspace(1e-6, 1.0, 32),
            xp.linspace(0.0, 1.0, 32),
            xp.linspace(0.0, 1.0, 32),
        ]

        tmp = model.domain(*grids_log)
        grids_phy = [tmp[0], tmp[1], tmp[2]]

        pointData = {}
        det_df = model.domain.jacobian_det(*grids_log)
        pointData["det_df"] = det_df

        if model.equil is not None:
            p0 = model.equil.p0(*grids_log)
            pointData["p0"] = p0
            if isinstance(model.equil, FluidEquilibriumWithB):
                absB0 = model.equil.absB0(*grids_log)
                pointData["absB0"] = absB0

        gridToVTK(os.path.join(path_out, "geometry"), *grids_phy, pointData=pointData)

    # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
    # use MPI.COMM_WORLD as communicator when storing the outputs
    data = DataContainer(path_out, comm=comm)

    # time quantities (current time value, value in seconds and index)
    time_state = {}
    time_state["value"] = xp.zeros(1, dtype=float)
    time_state["value_sec"] = xp.zeros(1, dtype=float)
    time_state["index"] = xp.zeros(1, dtype=int)

    # add time quantities to data object for saving
    for key, val in time_state.items():
        key_time = "time/" + key
        key_time_restart = "restart/time/" + key
        data.add_data({key_time: val})
        data.add_data({key_time_restart: val})

    # retrieve time parameters
    dt = time_opts.dt
    Tend = time_opts.Tend
    split_algo = time_opts.split_algo

    # set initial conditions for all variables
    if restart:
        model.initialize_from_restart(data)

        time_state["value"][0] = data.file["restart/time/value"][-1]
        time_state["value_sec"][0] = data.file["restart/time/value_sec"][-1]
        time_state["index"][0] = data.file["restart/time/index"][-1]

        total_steps = str(int(round((Tend - time_state["value"][0]) / dt)))
    else:
        total_steps = str(int(round(Tend / dt)))

    # compute initial scalars and kinetic data, pass time state to all propagators
    model.update_scalar_quantities()
    model.update_markers_to_be_saved()
    model.update_distr_functions()
    model.add_time_state(time_state["value"])

    # add all variables to be saved to data object
    save_keys_all, save_keys_end = model.initialize_data_output(data, size)

    # ======================== main time loop ======================
    model.update_scalar_quantities()
    if rank == 0:
        print("\nINITIAL SCALAR QUANTITIES:")
        model.print_scalar_quantities()

        print(f"\nSTART TIME STEPPING WITH '{split_algo}' SPLITTING:")

    # time loop
    run_time_now = 0.0
    while True:
        Barrier()

        # stop time loop?
        break_cond_1 = time_state["value"][0] >= Tend
        break_cond_2 = run_time_now > max_runtime

        if break_cond_1 or break_cond_2:
            # save restart data (other data already saved below)
            data.save_data(keys=save_keys_end)
            data.file.close()
            end_simulation = time.time()
            if rank == 0:
                print(f"\nTime steps done: {time_state['index'][0]}")
                print(
                    "wall-clock time of simulation [sec]: ",
                    end_simulation - start_simulation,
                )
                print()
            break

        if sort_step and time_state["index"][0] % sort_step == 0:
            t0 = time.time()
            for key, val in model.pointer.items():
                if isinstance(val, Particles):
                    val.do_sort()
            t1 = time.time()
            if rank == 0 and verbose:
                message = "Particles sorted | wall clock [s]: {0:8.4f} | sorting duration [s]: {1:8.4f}".format(
                    run_time_now * 60,
                    t1 - t0,
                )
                print(message, end="\n")
                print()

        # update time and index (round time to 10 decimals for a clean time grid!)
        time_state["value"][0] = round(time_state["value"][0] + dt, 10)
        time_state["value_sec"][0] = round(time_state["value_sec"][0] + dt * model.units.t, 10)
        time_state["index"][0] += 1

        # perform one time step dt
        t0 = time.time()
        with ProfileManager.profile_region("model.integrate"):
            model.integrate(dt, split_algo)
        t1 = time.time()

        run_time_now = (time.time() - start_simulation) / 60

        # update diagnostics data and save data
        if time_state["index"][0] % save_step == 0:
            # compute scalars and kinetic data
            model.update_scalar_quantities()
            model.update_markers_to_be_saved()
            model.update_distr_functions()

            # extract FEEC coefficients
            feec_species = model.field_species | model.fluid_species | model.diagnostic_species
            for species, val in feec_species.items():
                assert isinstance(val, Species)
                for variable, subval in val.variables.items():
                    assert isinstance(subval, FEECVariable)
                    spline = subval.spline
                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    spline.extract_coeffs(update_ghost_regions=False)

            # save data (everything but restart data)
            data.save_data(keys=save_keys_all)

            # print current time and scalar quantities to screen
            if rank == 0 and verbose:
                step = str(time_state["index"][0]).zfill(len(total_steps))

                message = "time step: " + step + "/" + str(total_steps)
                message += " | " + "time: {0:10.5f}/{1:10.5f}".format(time_state["value"][0], Tend)
                message += " | " + "phys. time [s]: {0:12.10f}/{1:12.10f}".format(
                    time_state["value_sec"][0],
                    Tend * model.units.t,
                )
                message += " | " + "wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}".format(
                    run_time_now * 60,
                    t1 - t0,
                )

                print(message, end="\n")
                model.print_scalar_quantities()
                print()

    # ===================================================================

    meta["wall-clock time[min]"] = (end_simulation - start_simulation) / 60
    comm.Barrier()

    if rank == 0:
        # save meta-data
        dict_to_yaml(meta, os.path.join(path_out, "meta.yml"))
        print("Struphy run finished.")

    if clone_config is not None:
        clone_config.free()


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
    file = h5py.File(os.path.join(path, "data/", "data_proc0.hdf5"), "r")

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

    file.close()

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


if __name__ == "__main__":
    import argparse
    import os

    import struphy
    import struphy.utils.utils as utils
    from struphy.profiling.profiling import (
        ProfileManager,
        ProfilingConfig,
        pylikwid_markerclose,
        pylikwid_markerinit,
    )

    # Read struphy state file
    state = utils.read_state()
    o_path = state["o_path"]

    parser = argparse.ArgumentParser(description="Run an Struphy model.")

    # model
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        metavar="MODEL",
        help="the name of the model to run (default: None)",
    )

    # input (absolute path)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        metavar="FILE",
        help="absolute path of parameter file",
    )

    # output (absolute path)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="DIR",
        help="absolute path of output folder (default=<out_path>/sim_1)",
        default=os.path.join(o_path, "sim_1"),
    )

    # restart
    parser.add_argument(
        "-r",
        "--restart",
        help="restart the simulation in the output folder specified under -o",
        action="store_true",
    )

    # max_runtime
    parser.add_argument(
        "--max-runtime",
        type=int,
        metavar="N",
        help="maximum wall-clock time of program in minutes (default=300)",
        default=300,
    )

    # save step
    parser.add_argument(
        "-s",
        "--save-step",
        type=int,
        metavar="N",
        help="how often to skip data saving (default=1, which means data is saved every time step)",
        default=1,
    )

    # sort step
    parser.add_argument(
        "--sort-step",
        type=int,
        metavar="N",
        help="sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)",
        default=0,
    )

    parser.add_argument(
        "--nclones",
        type=int,
        metavar="N",
        help="number of domain clones (default=1)",
        default=1,
    )

    # verbosity (screen output)
    parser.add_argument(
        "-v",
        "--verbose",
        help="supress screen output during time integration",
        action="store_true",
    )

    parser.add_argument(
        "--likwid",
        help="run with Likwid",
        action="store_true",
    )

    parser.add_argument(
        "--time-trace",
        help="Measure time traces for each call of the regions measured with ProfileManager",
        action="store_true",
    )

    parser.add_argument(
        "--sample-duration",
        help="Duration of samples when measuring time traces with ProfileManager",
        default=1.0,
    )

    parser.add_argument(
        "--sample-interval",
        help="Time between samples when measuring time traces with ProfileManager",
        default=1.0,
    )

    args = parser.parse_args()
    config = ProfilingConfig()
    config.likwid = args.likwid
    config.sample_duration = float(args.sample_duration)
    config.sample_interval = float(args.sample_interval)
    config.time_trace = args.time_trace
    config.simulation_label = ""
    pylikwid_markerinit()
    with ProfileManager.profile_region("main"):
        # solve the model
        run(
            args.model,
            args.input,
            args.output,
            restart=args.restart,
            runtime=args.runtime,
            save_step=args.save_step,
            verbose=args.verbose,
            sort_step=args.sort_step,
            num_clones=args.nclones,
        )
    pylikwid_markerclose()
    if config.time_trace:
        ProfileManager.print_summary()
        ProfileManager.save_to_pickle(os.path.join(args.output, "profiling_time_trace.pkl"))
