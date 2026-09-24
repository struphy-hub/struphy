"""Tests for the link between a Simulation and its output."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from struphy import BaseUnits, EnvironmentOptions, FieldsBackground, Output, Simulation, Time, maxwellians, perturbations
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import ColdPlasmaVlasov, Maxwell, Poisson, VlasovAmpereOneSpecies
from struphy.ode.utils import ButcherTableau
from struphy.particles.parameters import LoadingParameters
from struphy.pic.accumulation.filter import FilterParameters
from struphy.post_processing.post_processing_tools import PostProcessor, is_processed


def user_density_profile(eta1, eta2, eta3):
    return 1.0 + eta1 * 0.0 + eta2 * 0.0 + eta3 * 0.0


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
    assert run.path_out == Path(sim.env.path_out).resolve()
    assert not hasattr(run, "sim")
    assert "_sim" not in vars(run)
    assert sim.output is run

    sim.env = EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_2")
    assert sim.output is not run
    assert sim.output.path_out.name == "sim_2"


def test_from_output_restores_metadata_and_follows_a_moved_folder(tmp_path):
    model = VlasovAmpereOneSpecies(base_units=BaseUnits(x=2.0, B=3.0, n=4.0), mass_number=4.0, with_B0=False)
    sim = Simulation(model=model, env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim_1"))
    os.makedirs(os.path.join(sim.env.path_out, "data"))
    sim._write_run_metadata()

    moved = tmp_path / "moved"
    os.rename(sim.env.path_out, moved)
    restored = Output(moved)

    assert restored.model.to_dict() == model.to_dict()
    assert restored.model.params["mass_number"] == 4.0
    assert float(restored.model.units.t) == float(model.units.t)
    assert restored.domain == sim.domain
    assert restored.path_out == moved.resolve()
    assert restored.grid == sim.grid
    assert restored.derham_opts == sim.derham_opts
    assert sorted(os.listdir(tmp_path)) == ["moved"]


def test_run_writes_only_metadata_and_copies_the_parameter_file(tmp_path):
    params = tmp_path / "params_maxwell.py"
    params.write_text("# a parameter file\n")
    sim = make_sim(tmp_path, params_path=str(params))
    os.makedirs(sim.env.path_out)
    sim._write_run_metadata()
    sim._copy_parameter_file()
    assert sorted(os.listdir(sim.env.path_out)) == ["parameters.py", "run_metadata.json"]
    metadata = json.loads((tmp_path / "sim_1" / "run_metadata.json").read_text())
    assert metadata["model"] == sim.model.to_dict()
    assert metadata["mpi_ranks"] == sim.comm_size
    assert metadata["started_at_epoch_s"] == sim.start_time


def test_run_metadata_contains_variables_and_propagator_options(tmp_path):
    sim = make_sim(tmp_path)
    sim.model.em_fields.e_field.save_data = False
    sim.model.propagators.maxwell.options = sim.model.propagators.maxwell.Options(
        algo="explicit",
        solver_params=SolverParameters(tol=1e-6, maxiter=42),
        butcher=ButcherTableau("heun2"),
    )
    os.makedirs(sim.env.path_out)

    sim._write_run_metadata()

    metadata = json.loads((tmp_path / "sim_1" / "run_metadata.json").read_text())
    assert metadata["model"] == sim.model.to_dict()
    assert "species" not in metadata
    assert "propagator_options" not in metadata
    assert metadata["model"]["species"]["em_fields"]["variables"]["e_field"] == {
        "class": "FEECVariable",
        "space": "Hcurl",
        "save_data": False,
    }
    assert metadata["model"]["species"]["em_fields"]["variables"]["b_field"]["space"] == "Hdiv"
    options = metadata["model"]["propagator_options"]["maxwell"]
    assert options["algo"] == "explicit"
    assert options["solver_params"]["tol"] == 1e-6
    assert options["solver_params"]["maxiter"] == 42
    assert options["butcher"] == {"algo": "heun2"}


def test_run_metadata_contains_serialized_initial_conditions(tmp_path):
    sim = make_sim(tmp_path)
    velocity = sim.model.em_fields.b_field
    velocity.add_background(FieldsBackground(values=(1.0, 2.0, 3.0)))
    velocity.add_perturbation(perturbations.TorusModesCos(amps=(0.2,)))

    # A nested perturbation inside a summed kinetic distribution exercises the
    # recursive serializer used for PIC initial conditions.
    kinetic_sim = Simulation(model=VlasovAmpereOneSpecies(), env=EnvironmentOptions(out_folders=str(tmp_path)))
    perturbation = perturbations.TorusModesCos(amps=(0.3,))
    background = maxwellians.Maxwellian3D(n=(1.0, None))
    kinetic_sim.model.kinetic_ions.var.add_background(background)
    kinetic_sim.model.kinetic_ions.var.add_initial_condition(
        maxwellians.Maxwellian3D(n=(1.0, perturbation)) + background
    )

    metadata = json.loads(sim.to_run_metadata())
    b_field = metadata["initial_conditions"]["em_fields"]["b_field"]
    assert b_field["backgrounds"] == {
        "type": "FieldsBackground",
        "params": {"type": "LogicalConst", "values": [1.0, 2.0, 3.0], "variable": None},
    }
    assert b_field["perturbations"]["type"] == "TorusModesCos"

    kinetic = json.loads(kinetic_sim.to_run_metadata())["initial_conditions"]["kinetic_ions"]["var"]
    assert kinetic["backgrounds"]["type"] == "Maxwellian3D"
    assert kinetic["initial_condition"]["type"] == "SumKineticBackground"
    assert kinetic["initial_condition"]["params"]["f1"]["params"]["n"][1]["type"] == "TorusModesCos"


def test_run_metadata_embeds_user_function_source(tmp_path):
    sim = Simulation(model=VlasovAmpereOneSpecies(), env=EnvironmentOptions(out_folders=str(tmp_path)))
    sim.model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(user_density_profile, None)))

    density = json.loads(sim.to_run_metadata())["initial_conditions"]["kinetic_ions"]["var"]["backgrounds"][
        "params"
    ]["n"][0]
    assert density["type"] == "python_function"
    assert density["name"] == "user_density_profile"
    assert "def user_density_profile" in density["source"]
    assert len(density["source_sha256"]) == 64


def test_from_output_restores_initial_conditions_and_requires_trust_for_source(tmp_path):
    path_out = tmp_path / "sim_1"
    path_out.mkdir()
    sim = Simulation(model=VlasovAmpereOneSpecies(), env=EnvironmentOptions(out_folders=str(tmp_path)))
    sim.model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(user_density_profile, None)))
    sim.to_run_metadata(str(path_out / "run_metadata.json"))

    with pytest.raises(ValueError, match="trust_initial_condition_source=True"):
        Simulation.from_output(path_out)

    restored = Simulation.from_output(path_out, trust_initial_condition_source=True)
    density = restored.model.kinetic_ions.var.backgrounds.params["n"][0]
    assert density(0.2, 0.3, 0.4) == user_density_profile(0.2, 0.3, 0.4)


def test_run_metadata_names_variable_keys_in_propagator_options(tmp_path):
    sim = Simulation(model=Poisson(), env=EnvironmentOptions(out_folders=str(tmp_path)))
    variable = sim.model.em_fields.source
    sim.model.propagators.poisson.options.filter_params = {variable: FilterParameters("fourier_in_tor", (1, 2))}

    metadata = json.loads(sim.to_run_metadata())
    assert metadata["model"] == sim.model.to_dict()

    assert metadata["model"]["species"]["em_fields"]["variables"]["source"]["space"] == "H1"
    assert metadata["model"]["propagator_options"]["poisson"]["filter_params"] == {
        "em_fields.source": {"use_filter": "fourier_in_tor", "modes": [1, 2], "repeat": 1, "alpha": 0.5}
    }


def test_cold_plasma_vlasov_species_and_variables_own_their_metadata(tmp_path):
    model = ColdPlasmaVlasov(
        thermal_charge_number=-2,
        thermal_mass_number=0.25,
        thermal_alpha=3.0,
        thermal_epsilon=0.5,
        hot_mass_number=0.125,
        hot_epsilon=0.75,
    )
    model.hot_elec.set_markers(loading_params=LoadingParameters(Np=1234))
    model.hot_elec.var.save_data = False
    sim = Simulation(model=model, env=EnvironmentOptions(out_folders=str(tmp_path)))

    species = json.loads(sim.to_run_metadata())["model"]["species"]

    assert species["thermal_elec"] == model.thermal_elec.to_dict()
    assert species["hot_elec"] == model.hot_elec.to_dict()
    assert species["thermal_elec"]["class"] == "ThermalElectrons"
    assert species["thermal_elec"]["charge_number"] == -2
    assert species["thermal_elec"]["mass_number"] == 0.25
    assert species["thermal_elec"]["alpha"] == 3.0
    assert species["thermal_elec"]["epsilon"] == 0.5
    assert species["thermal_elec"]["variables"]["current"] == model.thermal_elec.current.to_dict()
    assert species["hot_elec"]["loading_params"]["Np"] == 1234
    assert species["hot_elec"]["variables"]["var"] == model.hot_elec.var.to_dict()
    assert species["hot_elec"]["variables"]["var"]["save_data"] is False


def test_from_output_never_executes_the_parameter_file(tmp_path):
    sim = make_sim(tmp_path, time_opts=Time(dt=0.123))
    os.makedirs(os.path.join(sim.env.path_out, "data"))
    sim._write_run_metadata()
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
        sim.export(os.path.join(sim.env.path_out, "config.json"))
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
    assert Output(moved).time_opts.dt == 0.123
    assert processor.process(create_vtk=False)
    assert is_processed(moved)


def test_from_output_prefers_metadata_over_legacy_config(tmp_path):
    sim = make_sim(tmp_path, time_opts=Time(dt=0.123))
    os.makedirs(sim.env.path_out)
    sim.export(os.path.join(sim.env.path_out, "config.json"))
    sim.time_opts = Time(dt=0.456)
    sim._write_run_metadata()
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
    views = SimpleNamespace(orbits="o", f="f", spline_values="s", n_sph="n")
    monkeypatch.setattr("struphy.simulation.sim.legacy_views", lambda out: views if out is output else None)
    for name, value in (("grids_log", "gl"), ("grids_phy", "gp"), ("time", np.array([0.0, 0.5]))):
        monkeypatch.setattr(type(output), name, property(lambda self, value=value: value))

    with pytest.deprecated_call():
        assert sim.load_plotting_data() is output
    assert (sim.orbits, sim.f, sim.spline_values, sim.n_sph) == ("o", "f", "s", "n")
    assert (sim.grids_log, sim.grids_phy) == ("gl", "gp")
    assert sim.t_grid.tolist() == [0.0, 0.5]
    with pytest.deprecated_call():
        assert sim.plotting_data is output
