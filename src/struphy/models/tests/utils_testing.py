import inspect
import os
import shutil
from types import ModuleType

from feectools.ddm.mpi import mpi as MPI

from struphy import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models.base import StruphyModel
from struphy.simulation.codes import StruphySimulation

rank = MPI.COMM_WORLD.Get_rank()


# generic function for calling model tests
def call_test(model: StruphyModel, test_profiling: bool = False, verbose: bool = True):
    model_name = model.name()
    if rank == 0:
        print(f"\n*** Testing '{model_name}':")

    # exceptions
    if model_name == "TwoFluidQuasiNeutralToy" and MPI.COMM_WORLD.Get_size() > 1:
        print(f"WARNING: Model {model_name} cannot be tested for {MPI.COMM_WORLD.Get_size() =}")
        return

    assert isinstance(model, StruphyModel), f"{model} of {type(model) = } is not a StruphyModel"

    # generate paramater file for testing
    test_folder = os.path.join(os.getcwd(), "struphy_model_test")
    path = os.path.join(test_folder, f"params_{model_name}.py")

    if rank == 0:
        model.generate_default_parameter_file(path=path, prompt=False)
        del model
    MPI.COMM_WORLD.Barrier()

    # set environment options
    env = EnvironmentOptions(
        out_folders=test_folder,
        sim_folder=f"{model_name}",
        profiling_activated=test_profiling,
        profiling_trace=test_profiling,
    )

    # read parameters
    params_in = import_parameters_py(path)
    base_units = params_in.base_units
    time_opts = params_in.time_opts
    domain = params_in.domain
    equil = params_in.equil
    grid = params_in.grid
    derham_opts = params_in.derham_opts
    model = params_in.model

    # test
    sim = StruphySimulation(
        model=model,
        params_path=path,
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

    # test restart
    env.restart = True
    time_opts.Tend += time_opts.dt
    
    sim.run(verbose=verbose)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        sim.pproc(verbose=verbose)
        sim.load_plotting_data(verbose=verbose)
        shutil.rmtree(test_folder)
    MPI.COMM_WORLD.Barrier()
