"""Tests for the link between a Simulation and its output."""

import os

import h5py
import pytest

from struphy import BaseUnits, EnvironmentOptions, Output, Simulation, Time, open_output
from struphy.models import Maxwell, VlasovAmpereOneSpecies
from struphy.post_processing.post_processing_tools import PostProcessor, is_processed


def make_sim(tmp_path, **kwargs):
    env = EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_1")
    return Simulation(model=Maxwell(), env=env, **kwargs)


def test_constructing_a_simulation_writes_nothing(tmp_path):
    sim = make_sim(tmp_path)
    assert not os.path.exists(sim.env.path_out)
    assert sim.derham is None


def test_output_is_the_run_of_the_current_output_folder(tmp_path):
    sim = make_sim(tmp_path)
    run = sim.output
    assert isinstance(run, Output)
    assert run.sim is sim
    assert sim.output is run

    sim.env = EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_2")
    assert sim.output is not run
    assert sim.output.path_out.name == "sim_2"


def test_from_output_restores_config_json_and_follows_a_moved_folder(tmp_path):
    model = VlasovAmpereOneSpecies(base_units=BaseUnits(x=2.0, B=3.0, n=4.0), mass_number=4.0, with_B0=False)
    sim = Simulation(model=model, env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_1"))
    os.makedirs(os.path.join(sim.env.path_out, "data"))
    sim._save_config()

    moved = tmp_path / "moved"
    os.rename(sim.env.path_out, moved)
    restored = open_output(moved).sim

    assert restored.model.to_dict() == model.to_dict()
    assert restored.model.params["mass_number"] == 4.0
    assert float(restored.model.units.t) == float(model.units.t)
    assert restored.domain == sim.domain
    assert restored.env.path_out == str(moved)
    assert restored.derham is None
    assert sorted(os.listdir(tmp_path)) == ["moved"]


def test_run_writes_config_json_and_copies_the_parameter_file(tmp_path):
    params = tmp_path / "params_maxwell.py"
    params.write_text("# a parameter file\n")
    sim = make_sim(tmp_path, params_path=str(params))
    os.makedirs(sim.env.path_out)
    sim._save_config()
    assert sorted(os.listdir(sim.env.path_out)) == ["config.json", "parameters.py"]


def test_from_output_never_executes_the_parameter_file(tmp_path):
    sim = make_sim(tmp_path, time_opts=Time(dt=0.123))
    os.makedirs(os.path.join(sim.env.path_out, "data"))
    sim._save_config()
    with open(os.path.join(sim.env.path_out, "parameters.py"), "w") as stream:
        stream.write("raise RuntimeError('the parameter file was executed')\n")
    assert Simulation.from_output(sim.env.path_out).time_opts.dt == 0.123


def test_from_output_requires_a_configuration(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        Simulation.from_output(tmp_path)


@pytest.mark.parametrize("metadata_only", [False, True])
def test_processor_from_moved_output(tmp_path, metadata_only):
    sim = make_sim(tmp_path, grid=None, derham_opts=None, time_opts=Time(dt=0.123))
    os.makedirs(os.path.join(sim.env.path_out, "data"))
    if metadata_only:
        sim.to_run_metadata(os.path.join(sim.env.path_out, "run_metadata.json"), mpi_ranks=3)
    else:
        sim._save_config()
        with open(os.path.join(sim.env.path_out, "meta.yml"), "w") as stream:
            stream.write("MPI processes: 3\n")
    with h5py.File(os.path.join(sim.env.path_out, "data", "data_proc0.hdf5"), "w") as data:
        data.create_dataset("time/value", data=[0.0, 0.123])
    moved = tmp_path / "moved"
    os.rename(sim.env.path_out, moved)
    products = moved / "post_processing"
    products.mkdir()
    sentinel = products / "existing.txt"
    sentinel.write_text("keep until processing")

    processor = PostProcessor.from_output(moved)

    assert processor.path_out == str(moved)
    assert processor.model.to_dict() == sim.model.to_dict()
    assert processor.domain == sim.domain
    assert processor.comm_size == 3
    assert list(processor.range_ranks) == [0, 1, 2]
    assert sentinel.read_text() == "keep until processing"
    assert open_output(moved).sim.time_opts.dt == 0.123
    assert processor.process(create_vtk=False)
    assert is_processed(moved)


def test_from_output_prefers_config_over_metadata(tmp_path):
    sim = make_sim(tmp_path, time_opts=Time(dt=0.123))
    os.makedirs(sim.env.path_out)
    sim.to_run_metadata(os.path.join(sim.env.path_out, "run_metadata.json"))
    sim.time_opts = Time(dt=0.456)
    sim._save_config()
    assert Simulation.from_output(sim.env.path_out).time_opts.dt == 0.456


def test_deprecated_pproc_delegates_to_the_output(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)
    calls = []
    monkeypatch.setattr(type(sim.output), "process", lambda self, **options: calls.append(options))
    monkeypatch.setattr(type(sim), "load_plotting_data", lambda self: "loaded")

    with pytest.deprecated_call():
        assert sim.pproc(physical=True) is None
    assert calls == [
        dict(
            step=1,
            celldivide=1,
            physical=True,
            guiding_center=False,
            classify=False,
            create_vtk=True,
            parallel=False,
            force=True,
        )
    ]
    with pytest.deprecated_call():
        assert sim.pproc(load=True) == "loaded"


def test_deprecated_load_plotting_data_attaches_the_products(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)
    output = sim.output
    for name, value in (
        ("orbits", "o"),
        ("distributions", "f"),
        ("fields", "s"),
        ("densities", "n"),
        ("grids_log", "gl"),
        ("grids_phy", "gp"),
        ("time", "t"),
    ):
        monkeypatch.setattr(type(output), name, property(lambda self, value=value: value))

    with pytest.deprecated_call():
        assert sim.load_plotting_data() is output
    assert (sim.orbits, sim.f, sim.spline_values, sim.n_sph) == ("o", "f", "s", "n")
    assert (sim.grids_log, sim.grids_phy, sim.t_grid) == ("gl", "gp", "t")
    with pytest.deprecated_call():
        assert sim.plotting_data is output
