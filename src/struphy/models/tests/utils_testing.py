import inspect
import os
import shutil
import tempfile
from types import ModuleType

from feectools.ddm.mpi import mpi as MPI

from struphy import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models.base import StruphyModel
from struphy.simulation.sim import Simulation

rank = MPI.COMM_WORLD.Get_rank()


# generic function for calling model tests
def call_test(model: StruphyModel, test_profiling: bool = False, verbose: bool = True):
    model_name = model.name()

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
    sim = Simulation(
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

    sim_dict = sim.to_dict()  # test the to_dict method
    sim2 = Simulation.from_dict(sim_dict)  # test the from_dict method
    assert sim == sim2, "Simulation to_dict and from_dict methods are not consistent"

    # test the generate_script method
    sim1_script = sim.generate_script()
    sim2_script = sim2.generate_script()
    assert sim1_script == sim2_script

    # Save the generated script to a file and check that it can be imported and run
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as tmp:
        sim.save_script(tmp.name, include_main_guard=True)
        tmp.seek(0)
        spec = import_parameters_py(tmp.name)
        assert isinstance(spec, ModuleType), "Generated script did not import as a module"
        assert hasattr(spec, "sim"), "Generated script does not have a 'sim' object"
        assert isinstance(spec.sim, Simulation), "'sim' object in generated script is not a Simulation instance"
        assert sim.generate_script() == spec.sim.generate_script(), (
            "Generated script does not match original simulation"
        )
        assert sim == spec.sim, "Simulation in generated script is not the same as the original simulation"

        # Run the simulation from the generated script

    # Export to json and import again
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as tmp:
        sim.export(tmp.name)
        tmp.seek(0)
        sim_from_json = Simulation.from_file(tmp.name)
        assert sim == sim_from_json, "Simulation JSON export/import is not consistent"

    # Export to yaml and import again
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as tmp:
        sim.export(tmp.name)
        tmp.seek(0)
        sim_from_yaml = Simulation.from_file(tmp.name)
        assert sim == sim_from_yaml, "Simulation YAML export/import is not consistent"

    sim.show_parameters()

    sim.run(verbose=verbose)

    # test restart
    env.restart = True
    time_opts.Tend += time_opts.dt
    sim.show_parameters()

    sim.run(verbose=verbose)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        sim.pproc(verbose=verbose)
        sim.load_plotting_data(verbose=verbose)
        shutil.rmtree(test_folder)
    MPI.COMM_WORLD.Barrier()
