"""Tests for the lazy Output output API."""

import json
import os
import pickle

import h5py
import numpy as np
import pytest

from struphy.post_processing.output import Output, open_output
from struphy.post_processing.post_processing_tools import is_processed, normalize_options, source_fingerprint

NT, N1, N2, N3, NV, N_MARKERS = 3, 4, 5, 6, 7, 10


def write_tree(root):
    pproc = os.path.join(root, "post_processing")
    fields = os.path.join(pproc, "fields_data")
    kinetic = os.path.join(pproc, "kinetic_data")
    os.makedirs(os.path.join(fields, "em_fields"))
    os.makedirs(os.path.join(kinetic, "kinetic_ions", "distribution_function", "e1_v1_density"))
    os.makedirs(os.path.join(kinetic, "kinetic_ions", "orbits"))
    t = np.linspace(0, 1, NT)
    np.save(os.path.join(pproc, "t_grid.npy"), t)
    logical = [np.linspace(0, 1, n) for n in (N1, N2, N3)]
    physical = np.meshgrid(*logical, indexing="ij")
    for name, value in (("grids_log", logical), ("grids_phy", physical)):
        with open(os.path.join(fields, f"{name}.bin"), "wb") as stream:
            pickle.dump(value, stream)
    values = {time: [np.full((N1, N2, N3), i + time) for i in range(3)] for time in t}
    with open(os.path.join(fields, "em_fields", "E.bin"), "wb") as stream:
        pickle.dump(values, stream)
    slice_dir = os.path.join(kinetic, "kinetic_ions", "distribution_function", "e1_v1_density")
    np.save(os.path.join(slice_dir, "grid_e1.npy"), np.linspace(0, 1, N1))
    np.save(os.path.join(slice_dir, "grid_v1.npy"), np.linspace(-3, 3, NV))
    np.save(os.path.join(slice_dir, "f_binned.npy"), np.ones((NT, N1, NV)))
    view_dir = os.path.join(kinetic, "kinetic_ions", "n_sph", "view_0")
    os.makedirs(view_dir)
    for direction, n in zip("123", (N1, N2, 1)):
        np.save(os.path.join(view_dir, f"grid_e{direction}.npy"), np.linspace(0, 1, n))
    np.save(os.path.join(view_dir, "n_sph.npy"), np.ones((NT, N1, N2, 1)))
    orbit_dir = os.path.join(kinetic, "kinetic_ions", "orbits")
    for step in range(NT):
        np.save(os.path.join(orbit_dir, f"kinetic_ions_{step}.npy"), np.full((N_MARKERS, 8), step))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)
    with h5py.File(os.path.join(data_dir, "data_proc0.hdf5"), "w") as file:
        file.create_dataset("time/value", data=t)
        file.create_dataset("scalar/en_tot", data=np.full(NT, 2.0))
    write_manifest(root)
    return root


def write_manifest(root, **options):
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_fingerprint": source_fingerprint(root),
        "options": normalize_options(**options),
    }
    with open(os.path.join(root, "post_processing", "manifest.json"), "w") as stream:
        json.dump(manifest, stream)


class FakeSim:
    """Just enough of a Simulation for Output: no configuration, a single rank."""

    time_opts = grid = derham_opts = domain = None
    rank, comm_size = 0, 1

    def __init__(self):
        self.processed = []

    def Barrier(self):
        pass


@pytest.fixture
def run(tmp_path):
    return Output(write_tree(str(tmp_path)), sim=FakeSim(), time_units="normalized")


def test_products_are_discovered_without_loading_arrays(run):
    assert tuple(run.fields) == ("em_fields",)
    assert tuple(run.distributions) == ("kinetic_ions",)
    assert tuple(run.orbits) == ("kinetic_ions",)
    assert run.field_catalog._cache == {}
    assert run.is_processed


def test_field_has_named_and_curvilinear_coordinates(run):
    field = run.fields["em_fields/E"]
    assert field.dims == ("t", "component", "e1", "e2", "e3")
    assert field.X.dims == ("e1", "e2", "e3")
    np.testing.assert_allclose(field.isel(t=0, component=2), 2)
    assert run.field_catalog._cache["em_fields/E"] is field


def test_binned_products_have_coordinates(run):
    data = run.distributions["kinetic_ions/e1_v1_density/f_binned"]
    assert data.dims == ("t", "e1", "v1")
    np.testing.assert_allclose(data.v1, np.linspace(-3, 3, NV))


def test_sph_density_views_take_dimensions_from_their_grids(run):
    data = run.densities.kinetic_ions.view_0.n_sph
    assert data.dims == ("t", "e1", "e2", "e3")
    assert data.shape == (NT, N1, N2, 1)
    np.testing.assert_allclose(data.e2, np.linspace(0, 1, N2))


def test_orbit_product_keeps_column_semantics(run):
    data = run.orbits["kinetic_ions"]
    assert data.dims == ("t", "marker", "attribute")
    assert data.attrs["columns"]["weight"] == 6


def test_scalar_time_uses_the_same_policy_as_postprocessed_products(run):
    assert set(run.scalars.data_vars) == {"en_tot"}
    np.testing.assert_allclose(run.scalars.en_tot.t, run.time)


def test_saving_scalars_and_bound_plot_accessor(run, tmp_path):
    path = run.save_scalars(tmp_path / "scalars.csv")
    assert os.path.exists(path)
    result = run.plot.timeseries(run.scalars.en_tot, logy=False)
    assert result.ax.get_xlabel() == "$t$"


def test_open_output_needs_an_output_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a Struphy output folder"):
        open_output(tmp_path)
    run = open_output(write_tree(str(tmp_path)))
    assert run.path_out == tmp_path.resolve()


def test_sim_is_restored_from_disk_only_on_access(tmp_path, monkeypatch):
    from struphy.simulation.sim import Simulation

    restored = FakeSim()
    calls = []
    monkeypatch.setattr(Simulation, "from_output", classmethod(lambda cls, path: calls.append(path) or restored))
    run = open_output(write_tree(str(tmp_path)))
    assert calls == []
    assert run.sim is restored and run.sim is restored
    assert calls == [tmp_path.resolve()]


def test_products_trigger_default_processing_when_missing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    run = Output(root, sim=FakeSim(), time_units="normalized")
    calls = []

    def fake_process(self, **options):
        calls.append(options)
        write_manifest(root)
        self._reset()
        return self

    monkeypatch.setattr(Output, "process", fake_process)
    assert set(run.scalars.data_vars) == {"en_tot"}
    assert calls == [], "scalars come from the raw output"
    assert tuple(run.fields) == ("em_fields",)
    assert calls == [{}]


def test_products_refuse_implicit_processing_on_many_ranks(tmp_path):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    sim = FakeSim()
    sim.comm_size = 2
    with pytest.raises(RuntimeError, match="on all ranks"):
        Output(root, sim=sim).fields


def test_processing_options_are_part_of_the_manifest(tmp_path):
    root = write_tree(str(tmp_path))
    write_manifest(root, step=1, celldivide=2, physical=False)
    assert is_processed(root)
    assert is_processed(root, dict(step=1, celldivide=(2, 2, 2), physical=False))
    assert not is_processed(root, dict(step=1, celldivide=2, physical=True))


def test_manifest_is_stale_when_raw_output_changes(tmp_path):
    root = write_tree(str(tmp_path))
    with open(os.path.join(root, "meta.yml"), "w") as stream:
        stream.write("MPI processes: 1\n")
    assert not is_processed(root)


@pytest.mark.parametrize("rank", [0, 1])
def test_serial_process_runs_on_rank_zero_only(tmp_path, monkeypatch, rank):
    from struphy.post_processing import post_processing_tools

    calls = []

    class FakePostProcessor:
        def __init__(self, sim, parallel_pproc=False):
            calls.append(("construct", parallel_pproc))

        def process(self, **options):
            calls.append(("process", options))

    monkeypatch.setattr(post_processing_tools, "PostProcessor", FakePostProcessor)
    sim = FakeSim()
    sim.rank = rank
    run = Output(write_tree(str(tmp_path)), sim=sim)
    assert run.process(physical=True) is run
    expected = [
        ("construct", False),
        (
            "process",
            dict(
                step=1, celldivide=1, physical=True, guiding_center=False, classify=False, create_vtk=False, force=False
            ),
        ),
    ]
    assert calls == (expected if rank == 0 else [])


def test_parallel_process_runs_on_every_rank(tmp_path, monkeypatch):
    from struphy.post_processing import post_processing_tools

    calls = []

    class FakePostProcessor:
        def __init__(self, sim, parallel_pproc=False):
            calls.append(parallel_pproc)

        def process(self, **options):
            pass

    monkeypatch.setattr(post_processing_tools, "PostProcessor", FakePostProcessor)
    sim = FakeSim()
    sim.rank = 3
    Output(write_tree(str(tmp_path)), sim=sim).process(parallel=True)
    assert calls == [True]
