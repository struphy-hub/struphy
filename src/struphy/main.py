from typing import Optional
import os
import sysconfig
import time
import datetime
import glob
import numpy as np
from mpi4py import MPI
from pyevtk.hl import gridToVTK

from struphy.feec.psydac_derham import SplineFunction
from struphy.fields_background.base import FluidEquilibriumWithB
from struphy.io.output_handling import DataContainer
from struphy.io.setup import setup_folders, setup_parameters
from struphy.models import fluid, hybrid, kinetic, toy
from struphy.models.base import StruphyModel
from struphy.profiling.profiling import ProfileManager
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml
from struphy.pic.base import Particles
from struphy.models.species import Species
from struphy.models.variables import FEECVariable


def main(
    model_name: Optional[str],
    params_path: Optional[str],
    path_out: str,
    *,
    restart: bool = False,
    runtime: int = 300,
    save_step: int = 1,
    verbose: bool = False,
    supress_out: bool = False,
    sort_step: int = 0,
    num_clones: int = 1,
):
    """
    Run a Struphy model.

    Parameters
    ----------
    model_name : str
        The name of the model to run. Type "struphy run --help" in your terminal to see a list of available models.

    params_path : str
        Path to .py parameter file.

    path_out : str
        The output directory. Will create a folder if it does not exist OR cleans the folder for new runs.

    restart : bool, optional
        Whether to restart a run (default=False).

    runtime : int, optional
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).

    save_step : int, optional
        When to save data output: every time step (save_step=1), every second time step (save_step=2), etc (default=1).

    verbose : bool
        Show full screen output.

    supress_out : bool
        Whether to supress screen output during time integration.

    sort_step: int, optional
        Sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)

    num_clones: int, optional
        Number of domain clones (default=1)
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_simulation = time.time()
    
    # load paramaters and set defaults
    params = setup_parameters(params_path=params_path, 
                            path_out=path_out,
                            verbose=verbose,)
    
    # check model
    model = params.model
    assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
    
    if model_name is None:
        assert model is not None, "If model is not specified, then model: MODEL must be specified in the params!"
        model_name = model.__class__.__name__
    
    # meta-data
    meta = {}
    meta["platform"] = sysconfig.get_platform()
    meta["python version"] = sysconfig.get_python_version()
    meta["model name"] = model_name
    meta["parameter file"] = params_path
    meta["output folder"] = path_out
    meta["MPI processes"] = size
    meta["number of domain clones"] = num_clones
    meta["restart"] = restart
    meta["max wall-clock [min]"] = runtime
    meta["save interval [steps]"] = save_step
    
    print("\nMETADATA:")
    for k, v in meta.items():
        print(f'{k}:'.ljust(25), v) 
    
    dict_to_yaml(meta, os.path.join(path_out, "meta.yml"))

    # creating output folders
    setup_folders(path_out=path_out, 
                  restart=restart, 
                  verbose=verbose,)
    
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
            clone_config = CloneConfig(comm=comm, params=params, num_clones=num_clones)
            clone_config.print_clone_config()
            if model.kinetic_species:
                clone_config.print_particle_config()
    comm.Barrier()
    
    ## configure model instance
    
    # mpi config
    model._comm_world = comm
    model._clone_config = clone_config

    if model.comm_world is None:
        model._rank_world = 0
    else:
        model._rank_world = model.comm_world.Get_rank()

    # units
    model._units = params.units
    if model.bulk_species is None:
        A_bulk = None
        Z_bulk = None
    else:
        A_bulk = model.bulk_species.mass_number
        Z_bulk = model.bulk_species.charge_number
    model.units.derive_units(velocity_scale=model.velocity_scale,
                             A_bulk=A_bulk,
                             Z_bulk=Z_bulk,
                             verbose=verbose,)
    
    # domain and fluid bckground
    model.setup_domain_and_equil(params.domain, params.equil)
    
    # allocate derham-related objects
    if params.grid is not None:
        model.allocate_feec(params.grid, params.derham)
    else:
        model._derham = None
        model._mass_ops = None
        model._projected_equil = None
        print("\nDERHAM:\nMeshless simulation - no Derham complex set up.")
        
    # allocate variables
    model.allocate_variables()
    model.allocate_helpers()
    
    # pass info to propagators
    model.allocate_propagators()
    
    # plasma parameters
    model.compute_plasma_params(verbose=verbose)
    model.setup_equation_params(units=model.units, verbose=verbose)

    if model_name is None:
        assert model is not None, "If model is not specified, then model: MODEL must be specified in the params!"
        model_name = model.__class__.__name__

    if rank < 32:
        if rank == 0:
            print("")
        print(f"Rank {rank}: calling struphy/main.py for model {model_name} ...")
    if size > 32 and rank == 32:
        print(f"Ranks > 31: calling struphy/main.py for model {model_name} ...")

    # store geometry vtk
    if rank == 0:
        grids_log = [
            np.linspace(1e-6, 1.0, 32),
            np.linspace(0.0, 1.0, 32),
            np.linspace(0.0, 1.0, 32),
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
    time_state["value"] = np.zeros(1, dtype=float)
    time_state["value_sec"] = np.zeros(1, dtype=float)
    time_state["index"] = np.zeros(1, dtype=int)

    # add time quantities to data object for saving
    for key, val in time_state.items():
        key_time = "time/" + key
        key_time_restart = "restart/time/" + key
        data.add_data({key_time: val})
        data.add_data({key_time_restart: val})

    # retrieve time parameters
    dt = params.time.dt
    Tend = params.time.Tend
    split_algo = params.time.split_algo

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
        break_cond_2 = run_time_now > runtime

        if break_cond_1 or break_cond_2:
            # save restart data (other data already saved below)
            data.save_data(keys=save_keys_end)
            data.file.close()
            end_simulation = time.time()
            if rank == 0:
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
            if rank == 0 and not supress_out:
                message = "Particles sorted | wall clock [s]: {0:8.4f} | sorting duration [s]: {1:8.4f}".format(
                    run_time_now * 60, t1 - t0
                )
                print(message, end="\n")
                print()

        # perform one time step dt
        t0 = time.time()
        with ProfileManager.profile_region("model.integrate"):
            model.integrate(dt, split_algo)
        t1 = time.time()

        # update time and index (round time to 10 decimals for a clean time grid!)
        time_state["value"][0] = round(time_state["value"][0] + dt, 10)
        time_state["value_sec"][0] = round(time_state["value_sec"][0] + dt * model.units.t, 10)
        time_state["index"][0] += 1

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
            if rank == 0 and not supress_out:
                step = str(time_state["index"][0]).zfill(len(total_steps))

                message = "time step: " + step + "/" + str(total_steps)
                message += " | " + "time: {0:10.5f}/{1:10.5f}".format(time_state["value"][0], Tend)
                message += " | " + "phys. time [s]: {0:12.10f}/{1:12.10f}".format(
                    time_state["value_sec"][0], Tend * model.units.t
                )
                message += " | " + "wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}".format(
                    run_time_now * 60, t1 - t0
                )

                print(message, end="\n")
                model.print_scalar_quantities()
                print()

    # ===================================================================

    with open(path_out + "/meta.txt", "a") as f:
        # f.write('wall-clock time [min]:'.ljust(30) + str((end_simulation - start_simulation)/60.) + '\n')
        f.write(f"{rank} {'wall-clock time[min]: '.ljust(30)}{(end_simulation - start_simulation) / 60}\n")
    Barrier()
    if rank == 0:
        print("Struphy run finished.")

    if clone_config is not None:
        clone_config.free()


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

    # runtime
    parser.add_argument(
        "--runtime",
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

    # supress screen output
    parser.add_argument(
        "--supress-out",
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
        # Call main
        main(
            args.model,
            args.input,
            args.output,
            restart=args.restart,
            runtime=args.runtime,
            save_step=args.save_step,
            verbose=args.verbose,
            supress_out=args.supress_out,
            sort_step=args.sort_step,
            num_clones=args.nclones,
        )
    pylikwid_markerclose()
    if config.time_trace:
        ProfileManager.print_summary()
        ProfileManager.save_to_pickle(os.path.join(args.output, "profiling_time_trace.pkl"))
