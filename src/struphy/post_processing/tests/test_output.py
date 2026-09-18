"""Tests for the lazy Output output API."""

import json
import os

import h5py
import numpy as np
import pytest
import xarray as xr

from struphy import BaseUnits, Time, domains
from struphy.models import Maxwell

from struphy.post_processing.output import Output, open_output
from struphy.post_processing import store
from struphy.post_processing.arrays import orbit_quantities
from struphy.post_processing.post_processing_tools import is_processed, normalize_options, source_fingerprint

NT, N1, N2, N3, NV, N_MARKERS = 3, 4, 5, 6, 7, 10


def write_tree(root):
    """A small but complete output folder: raw HDF5 plus the product store."""
    pproc = os.path.join(root, "post_processing")
    os.makedirs(pproc)
    t = np.linspace(0, 1, NT)
    np.save(os.path.join(pproc, "t_grid.npy"), t)

    logical = {f"e{axis + 1}": np.linspace(0, 1, n) for axis, n in enumerate((N1, N2, N3))}
    mapped = np.meshgrid(*logical.values(), indexing="ij")
    path = store.store_path(pproc)
    store.create(path)
    store.write_group(path, "/em_fields", xr.Dataset(
        {"E": (("t", "component", "e1", "e2", "e3"),
               np.stack([np.stack([np.full((N1, N2, N3), i + time) for i in range(3)]) for time in t]))},
        coords={"t": t, "component": [0, 1, 2], **logical,
                **{name: (("e1", "e2", "e3"), grid) for name, grid in zip(("X", "Y", "Z"), mapped)}},
    ))
    store.write_group(path, "/kinetic_ions/e1_v1_density", xr.Dataset(
        {"f": (("t", "e1", "v1"), np.ones((NT, N1, NV))), "delta_f": (("t", "e1", "v1"), np.zeros((NT, N1, NV)))},
        coords={"t": t, "e1": logical["e1"], "v1": np.linspace(-3, 3, NV)},
    ))
    store.write_group(path, "/kinetic_ions/view_0", xr.Dataset(
        {"n": (("t", "e1", "e2", "e3"), np.ones((NT, N1, N2, 1)))},
        coords={"t": t, "e1": logical["e1"], "e2": logical["e2"], "e3": np.zeros(1)},
    ))
    store.write_group(path, "/kinetic_ions", xr.Dataset(
        {"orbits": (("t", "marker", "quantity"),
                    np.stack([np.full((N_MARKERS, 8), step) for step in range(NT)]))},
        coords={"t": t, "marker": np.arange(N_MARKERS), "quantity": orbit_quantities(8)},
    ))

    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)
    with h5py.File(os.path.join(data_dir, "data_proc0.hdf5"), "w") as file:
        file.create_dataset("time/value", data=t)
        file.create_group("feec/em_fields")          # the raw output names the species,
        file.create_group("kinetic/kinetic_ions")    # as a real run does
        file.create_dataset("scalar/en_tot", data=np.full(NT, 2.0))
    metadata = {
        "model": Maxwell(base_units=BaseUnits(x=2.0)).to_dict(),
        "domain": domains.Cuboid().to_dict(),
        "equil": None, "grid": None, "derham_opts": None,
        "time_opts": Time().to_dict(), "mpi_ranks": 1,
    }
    with open(os.path.join(root, "run_metadata.json"), "w") as stream:
        json.dump(metadata, stream)
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


class FakeComm:
    def __init__(self, rank=0, size=1):
        self.rank, self.size = rank, size
        self.barriers = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Barrier(self):
        self.barriers += 1


@pytest.fixture
def run(tmp_path):
    return Output(write_tree(str(tmp_path)), time_units="normalized")


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
    data = run.distributions["kinetic_ions/e1_v1_density/f"]
    assert data.dims == ("t", "e1", "v1")
    np.testing.assert_allclose(data.v1, np.linspace(-3, 3, NV))


def test_sph_density_views_take_dimensions_from_their_grids(run):
    data = run.densities.kinetic_ions.view_0.n
    assert data.dims == ("t", "e1", "e2", "e3")
    assert data.shape == (NT, N1, N2, 1)
    np.testing.assert_allclose(data.e2, np.linspace(0, 1, N2))


def test_orbit_product_keeps_column_semantics(run):
    data = run.orbits["kinetic_ions"]
    assert data.dims == ("t", "marker", "quantity")
    assert list(data.quantity.values) == ["x", "y", "z", "v1", "v2", "v3", "weight", "id"]


def test_scalar_time_uses_the_same_policy_as_postprocessed_products(run):
    assert set(run.scalars.data_vars) == {"en_tot"}
    np.testing.assert_allclose(run.scalars.en_tot.t, run.time)


def test_saving_scalars_and_plotting_a_product(run, tmp_path):
    path = run.save_scalars(tmp_path / "scalars.csv")
    assert os.path.exists(path)
    result = run.scalars.en_tot.struphy.plot.timeseries(logy=False)
    assert result.ax.get_xlabel() == "$t$"


def test_open_output_needs_an_output_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a Struphy output folder"):
        open_output(tmp_path)
    run = open_output(write_tree(str(tmp_path)))
    assert run.path_out == tmp_path.resolve()


def test_configuration_is_restored_lazily_without_a_simulation(tmp_path, monkeypatch):
    from struphy import Simulation

    root = write_tree(str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("Output must not construct a Simulation")
    monkeypatch.setattr(Simulation, "__init__", forbidden)
    run = Output(root)
    assert run.metadata["model"] == Maxwell(base_units=BaseUnits(x=2.0)).to_dict()
    assert "model" not in vars(run)
    assert not hasattr(run, "sim")
    assert "_sim" not in vars(run)
    assert run.domain == domains.Cuboid()
    assert run.model.to_dict() == run.metadata["model"]
    assert run.model is run.model
    assert run.time_opts == Time()
    assert run.grid is run.derham_opts is run.equil is None
    assert run.mpi_ranks == 1
    assert run.with_time_units("physical").model.to_dict() == run.model.to_dict()


def test_evaluate_triggers_default_processing_when_missing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    run = Output(root, time_units="normalized")
    calls = []

    def fake_pproc(self, **options):
        calls.append(options)
        write_manifest(root)
        self._reset()
        return self

    monkeypatch.setattr(Output, "pproc", fake_pproc)
    assert set(run.scalars.data_vars) == {"en_tot"}
    assert calls == [], "scalars come from the raw output"
    assert run.evaluate("em_fields/E").name == "E"
    assert calls == [{}]


def test_evaluate_returns_xarray_and_xarray_exposes_the_product_tree(run):
    field = run.evaluate("em_fields/E")
    assert isinstance(field, xr.DataArray)
    assert field is run["em_fields/E"]
    assert run.xarray is run.tree


def test_evaluate_selects_positions_coordinates_and_slices(run):
    field = run.evaluate("em_fields/E", isel={"t": -1, "component": 2})
    assert field.dims == ("e1", "e2", "e3")
    np.testing.assert_allclose(field, 3.0)

    phase_space = run.evaluate(
        "kinetic_ions/e1_v1_density/f",
        sel={"e1": 0.49},
        method="nearest",
        drop=True,
    )
    assert phase_space.dims == ("t", "v1")

    history = run.evaluate("en_tot", isel={"t": slice(1, None)})
    assert history.sizes["t"] == NT - 1

    values = run.evaluate("en_tot", isel={"t": -1}, as_numpy=True)
    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, 2.0)


def test_products_refuse_implicit_processing_on_many_ranks(tmp_path):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    comm = FakeComm(size=2)
    with pytest.raises(RuntimeError, match="on all ranks"):
        Output(root, comm=comm).fields


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
        def __init__(self, output, parallel_pproc=False):
            calls.append(("construct", parallel_pproc))

        def process(self, **options):
            calls.append(("process", options))

    monkeypatch.setattr(post_processing_tools, "PostProcessor", FakePostProcessor)
    comm = FakeComm(rank=rank, size=2)
    run = Output(write_tree(str(tmp_path)), comm=comm)
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
    assert comm.barriers == 1


def test_parallel_process_runs_on_every_rank(tmp_path, monkeypatch):
    from struphy.post_processing import post_processing_tools

    calls = []

    class FakePostProcessor:
        def __init__(self, output, parallel_pproc=False):
            calls.append(parallel_pproc)

        def process(self, **options):
            pass

    monkeypatch.setattr(post_processing_tools, "PostProcessor", FakePostProcessor)
    Output(write_tree(str(tmp_path)), comm=FakeComm(rank=3, size=4)).process(parallel=True)
    assert calls == [True]


def test_unknown_species_never_starts_processing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    run = Output(root, time_units="normalized")
    calls = []
    monkeypatch.setattr(Output, "process", lambda self, **options: calls.append(options))

    with pytest.raises(AttributeError, match="available species"):
        run.typo_here
    assert not hasattr(run, "anything")
    assert calls == [], "a typo must not post-process the run"
    assert {"em_fields", "kinetic_ions"} <= set(dir(run)), "species are known before processing"


def test_info_lists_products_without_loading(run):
    text = run.info()
    assert "out.scalars.en_tot" in text
    assert "out.kinetic_ions.e1_v1_density.f" in text
    assert "out.kinetic_ions.orbits" in text
    assert "out.em_fields.E" in text
    assert run.field_catalog._cache == {}, "listing must not load arrays"


def test_info_hints_distribution_and_density_symbols(run):
    text = run.info()
    assert "out.kinetic_ions.e1_v1_density.f  ($f$)" in text
    assert "out.kinetic_ions.e1_v1_density.delta_f  ($\\delta f$)" in text
    assert "out.kinetic_ions.view_0.n  ($n$)" in text
    assert "out.em_fields.E" in text and "($" not in text.split("out.em_fields.E")[1].split("\n")[0]


def test_normalized_time_carries_seconds_as_a_coordinate(run):
    energy = run.scalars.en_tot
    assert "units" not in energy.t.attrs, "normalized time has no unit"
    np.testing.assert_allclose(energy.t_seconds, energy.t * float(run.model.units.t))
    assert energy.t_seconds.attrs["units"] == "s"

    seconds = Output(run.path_out, time_units="physical").scalars.en_tot
    np.testing.assert_allclose(seconds.t, energy.t * float(run.model.units.t))
    assert "t_seconds" not in seconds.coords


def test_a_failing_property_reports_its_own_error(tmp_path):
    root = write_tree(str(tmp_path))
    (tmp_path / "run_metadata.json").unlink()
    run = Output(root)
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        run.domain


def test_parallel_processing_rejects_a_different_rank_count(tmp_path):
    run = Output(write_tree(str(tmp_path)), comm=FakeComm(size=2))
    with pytest.raises(ValueError, match="same number of MPI ranks"):
        run.process(parallel=True)


def test_saved_rank_count_does_not_block_serial_implicit_processing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    run = Output(root, comm=FakeComm())
    run.metadata["mpi_ranks"] = 8
    (run.path_pproc / "manifest.json").unlink()
    calls = []
    monkeypatch.setattr(Output, "pproc", lambda self: calls.append(self.path_out))
    run._ensure_processed()
    assert calls == [run.path_out]
