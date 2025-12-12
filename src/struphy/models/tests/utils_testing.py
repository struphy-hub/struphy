import inspect
import os
import shutil
from types import ModuleType

from feectools.ddm.mpi import mpi as MPI

import struphy.models as models
from struphy import main
from struphy.io.options import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models.base import StruphyModel
import struphy.models.utils as models_utils
rank = MPI.COMM_WORLD.Get_rank()


# generic function for calling model tests
def call_test(model_name: str, module: ModuleType = None, verbose=True):
    if rank == 0:
        print(f"\n*** Testing '{model_name}':")

    # exceptions
    if model_name == "TwoFluidQuasiNeutralToy" and MPI.COMM_WORLD.Get_size() > 1:
        print(f"WARNING: Model {model_name} cannot be tested for {MPI.COMM_WORLD.Get_size() =}")
        return

    if module is None:
        model = models_utils.get_model_by_name(model_name=model_name)()
        try:
            model = getattr(models, model_name)()
        except AttributeError:
            raise ModuleNotFoundError(f"{model_name} not found!")

    else:
        model = getattr(module, model_name)()

    assert isinstance(model, StruphyModel)

    # generate paramater file for testing
    test_folder = os.path.join(os.getcwd(), "struphy_model_test")
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

    # Restart and run one more timestep
    params_in = import_parameters_py(path)
    base_units = params_in.base_units
    time_opts = params_in.time_opts
    domain = params_in.domain
    equil = params_in.equil
    grid = params_in.grid
    derham_opts = params_in.derham_opts
    model = params_in.model
    env.restart = True
    time_opts.Tend += time_opts.dt

    # test restart
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

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        path_out = os.path.join(test_folder, model_name)
        main.pproc(path=path_out)
        main.load_data(path=path_out)
        shutil.rmtree(test_folder)
    MPI.COMM_WORLD.Barrier()
