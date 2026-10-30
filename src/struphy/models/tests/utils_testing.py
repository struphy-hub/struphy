import inspect
import logging
import os
import shutil
import tempfile
from types import ModuleType

from maybempi import MPI, launched_under_mpi

from struphy import EnvironmentOptions
from struphy.io.setup import import_parameters_py
from struphy.models.base import StruphyModel
from struphy.models.tests.regression import utils_regression as ur
from struphy.simulation.sim import Simulation

logger = logging.getLogger("struphy")


def _describe_dict_diff(a, b, path: str = "") -> list[str]:
    """Recursively compare two (nested) dicts/lists and describe every leaf that differs.

    Used to turn a bare ``sim != sim2`` assertion into a readable list of the
    exact fields that diverged, which is otherwise very hard to track down.
    """
    lines = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            lines += _describe_dict_diff(a.get(key, "<MISSING>"), b.get(key, "<MISSING>"), f"{path}.{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            lines.append(f"  length mismatch at {path}: {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            lines += _describe_dict_diff(x, y, f"{path}[{i}]")
    elif a != b:
        lines.append(f"  {path}: {a!r} != {b!r}")
    return lines


# generic function for calling model tests
def call_test(
    model: StruphyModel,
    test_profiling: bool = False,
    case: str | None = None,
    update_baselines: bool = False,
):
    """Run a model with its default parameter file, compare the result against the regression baseline (see
    :mod:`struphy.models.tests.regression.utils_regression`), then test a restart and the post-processing.

    Parameters
    ----------
    model : StruphyModel
        The model to test.

    test_profiling : bool
        Whether to run with profiling.

    case : str | None
        Name of the regression baseline; defaults to the model name. Give a distinct name for every non-default
        model variant (e.g. ``"LinearMHD-b_space=Hcurl-projected=True"``).

    update_baselines : bool
        Overwrite the baseline instead of comparing against it.

    Raises
    ------
    RegressionMismatch
        If the run differs from the baseline.
    """
    model_name = model.name()
    if case is None:
        case = model_name

    # Every communicator access goes through this one check: without an MPI launcher
    # the test still runs, as a single process, and must not touch MPI at all.
    comm = MPI.COMM_WORLD if launched_under_mpi() else None
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1

    # exceptions
    if model_name == "TwoFluidQuasiNeutralToy" and size > 1:
        logger.info(f"WARNING: Model {model_name} cannot be tested for {size = }")
        return

    assert isinstance(model, StruphyModel), f"{model} of {type(model) = } is not a StruphyModel"

    # generate paramater file for testing
    test_folder = os.path.join(os.getcwd(), "struphy_model_test")
    path = os.path.join(test_folder, f"params_{model_name}.py")

    if launched_under_mpi():
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        # If not launched with mpirun,
        # we can still run the test in a single process.
        rank = 0

    if rank == 0:
        model.generate_default_parameter_file(path=path, prompt=False)
        del model
    if comm is not None:
        comm.Barrier()

    # set environment options
    env = EnvironmentOptions(
        out_folders=test_folder,
        sim_folder=f"{model_name}",
    )

    # read parameters
    params_in = import_parameters_py(path)
    time_opts = params_in.time_opts
    domain = params_in.domain
    equil = params_in.equil
    grid = params_in.grid
    derham_opts = params_in.derham_opts
    model = params_in.model

    # reproducible random marker loading and noise
    ur.set_seeds(model)

    # test
    sim = Simulation(
        model=model,
        params_path=path,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )

    sim_dict = sim.to_dict()  # test the to_dict method
    sim2 = Simulation.from_dict(sim_dict)  # test the from_dict method
    if sim != sim2:
        raise AssertionError(
            f"Simulation to_dict and from_dict methods are not consistent for {model_name}:\n"
            + "\n".join(_describe_dict_diff(sim_dict, sim2.to_dict()))
        )

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

        # Output the simulation from the generated script

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

    sim.run(profiling_activated=test_profiling)

    # regression: compare (or update) the baseline of this run on rank 0, then tell all ranks;
    # the restart below is not part of the regression data
    regression_data = ur.collect(sim)
    mismatch = None
    if rank == 0:
        mismatch = ur.check(case, size, regression_data, update_baselines=update_baselines)
    if comm is not None:
        mismatch = comm.bcast(mismatch, root=0)

    # test restart
    env.restart = True
    time_opts.Tend += time_opts.dt
    sim.show_parameters()

    run = sim.run(profiling_activated=test_profiling)

    if comm is not None:
        comm.Barrier()
    run.pproc(create_vtk=True)
    if rank == 0:
        # discover (but do not load) every product
        for catalog in (run.field_catalog, run.distribution_catalog, run.density_catalog, run.orbit_catalog):
            tuple(catalog)
        shutil.rmtree(test_folder)
    if comm is not None:
        comm.Barrier()

    if mismatch is not None:
        raise ur.RegressionMismatch(mismatch)
