import inspect
import os
from types import ModuleType

import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy import main
from struphy.io.options import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models import fluid, hybrid, kinetic, toy
from struphy.models.base import StruphyModel

rank = MPI.COMM_WORLD.Get_rank()

# available models
toy_models = []
for name, obj in inspect.getmembers(toy):
    if inspect.isclass(obj) and "models.toy" in obj.__module__:
        toy_models += [name]
if rank == 0:
    print(f"\n{toy_models =}")

fluid_models = []
for name, obj in inspect.getmembers(fluid):
    if inspect.isclass(obj) and "models.fluid" in obj.__module__:
        fluid_models += [name]
if rank == 0:
    print(f"\n{fluid_models =}")

kinetic_models = []
for name, obj in inspect.getmembers(kinetic):
    if inspect.isclass(obj) and "models.kinetic" in obj.__module__:
        kinetic_models += [name]
if rank == 0:
    print(f"\n{kinetic_models =}")

hybrid_models = []
for name, obj in inspect.getmembers(hybrid):
    if inspect.isclass(obj) and "models.hybrid" in obj.__module__:
        hybrid_models += [name]
if rank == 0:
    print(f"\n{hybrid_models =}")


# folder for test simulations
test_folder = os.path.join(os.getcwd(), "struphy_model_tests")


# generic function for calling model tests
def call_test(model_name: str, module: ModuleType = None, test_pproc: bool = True, verbose=True,):
    if rank == 0:
        print(f"\n*** Testing '{model_name}':")

    # exceptions
    if model_name == "TwoFluidQuasiNeutralToy" and MPI.COMM_WORLD.Get_size() > 1:
        print(f"WARNING: Model {model_name} cannot be tested for {MPI.COMM_WORLD.Get_size() =}")
        return

    if module is None:
        submods = [toy, fluid, kinetic, hybrid]
        for submod in submods:
            try:
                model = getattr(submod, model_name)()
            except AttributeError:
                continue

    else:
        model = getattr(module, model_name)()

    assert isinstance(model, StruphyModel)

    # generate paramater file for testing
    path = os.path.join(test_folder, f"params_{model_name}.py")

    if rank == 0:
        model.generate_default_parameter_file(path=path, prompt=False)
        del model
    MPI.COMM_WORLD.Barrier()

    # set environment options
    env = EnvironmentOptions(out_folders=test_folder, sim_folder=f"{model_name}")

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
    main.run(
        model,
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

    if test_pproc:
        # MPI.COMM_WORLD.Barrier()
        # if rank == 0:
        path_out = os.path.join(test_folder, model_name)
        main.pproc(path=path_out)
        return
        main.load_data(path=path_out)
        # MPI.COMM_WORLD.Barrier()