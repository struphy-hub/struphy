"""Tests for the link between a Simulation and its output."""

import os

import pytest

from struphy import BaseUnits, EnvironmentOptions, Run, Simulation, open_run
from struphy.models import Maxwell, VlasovAmpereOneSpecies


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
    assert isinstance(run, Run)
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
    restored = open_run(moved).sim

    assert restored.model.to_dict() == model.to_dict()
    assert restored.model.params["mass_number"] == 4.0
    assert float(restored.model.units.t) == float(model.units.t)
    assert restored.domain == sim.domain
    assert restored.env.path_out == str(moved)
    assert restored.derham is None
    assert sorted(os.listdir(tmp_path)) == ["moved"]


def test_from_output_prefers_the_parameter_file(tmp_path):
    path_out = tmp_path / "sim_1"
    os.makedirs(path_out / "data")
    (path_out / "parameters.py").write_text(
        "from struphy import EnvironmentOptions, Simulation, Time\n"
        "from struphy.models import Maxwell\n"
        "sim = Simulation(model=Maxwell(), env=EnvironmentOptions(sim_folder='elsewhere'), time_opts=Time(dt=0.123))\n"
    )
    restored = Simulation.from_output(path_out)
    assert restored.time_opts.dt == 0.123
    assert restored.env.path_out == str(path_out)
    assert not os.path.exists(os.path.join(os.getcwd(), "elsewhere"))


def test_from_output_requires_a_configuration(tmp_path):
    with pytest.raises(FileNotFoundError, match="parameters.py"):
        Simulation.from_output(tmp_path)
