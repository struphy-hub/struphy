"""Tests for the lazy Output output API."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import xarray as xr

from struphy import BaseUnits, Time, domains
from struphy.models import Maxwell
from struphy.pic.particles import Particles6D
from struphy.post_processing import output as output_module
from struphy.post_processing import store
from struphy.post_processing.arrays import wrap_orbits
from struphy.post_processing.manifest import is_processed, normalize_options, source_fingerprint
from struphy.post_processing.output import Output, open_output

NT, N1, N2, N3, NV, N_MARKERS = 3, 4, 5, 6, 7, 10


def write_tree(root):
    """A small but complete output folder: raw HDF5 plus the product store."""
    pproc = os.path.join(root, "post_processing")
    os.makedirs(pproc)
    t = np.linspace(0, 1, NT)
    np.save(os.path.join(pproc, "t_grid.npy"), t)

    logical = {f"eta{axis + 1}": np.linspace(0, 1, n) for axis, n in enumerate((N1, N2, N3))}
    mapped = np.meshgrid(*logical.values(), indexing="ij")
    path = store.store_path(pproc)
    store.create(path)
    store.write_group(
        path,
        "/em_fields",
        xr.Dataset(
            {
                "E": (
                    ("t", "component", "eta1", "eta2", "eta3"),
                    np.stack([np.stack([np.full((N1, N2, N3), i + time) for i in range(3)]) for time in t]),
                )
            },
            coords={
                "t": t,
                "component": [0, 1, 2],
                **logical,
                **{name: (("eta1", "eta2", "eta3"), grid) for name, grid in zip(("X", "Y", "Z"), mapped)},
            },
        ),
    )
    store.write_group(
        path,
        "/kinetic_ions/e1_v1_density",
        xr.Dataset(
            {
                "f": (("t", "eta1", "v1"), np.ones((NT, N1, NV))),
                "delta_f": (("t", "eta1", "v1"), np.zeros((NT, N1, NV))),
            },
            coords={"t": t, "eta1": logical["eta1"], "v1": np.linspace(-3, 3, NV)},
        ),
    )
    store.write_group(
        path,
        "/kinetic_ions/view_0",
        xr.Dataset(
            {"n": (("t", "eta1", "eta2", "eta3"), np.ones((NT, N1, N2, 1)))},
            coords={"t": t, "eta1": logical["eta1"], "eta2": logical["eta2"], "eta3": np.zeros(1)},
        ),
    )
    orbits = np.stack([np.full((N_MARKERS, 7), step) for step in range(NT)])
    store.write_group(path, "/kinetic_ions/orbits", wrap_orbits(orbits, t, Particles6D.orbit_quantities))

    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)
    with h5py.File(os.path.join(data_dir, "data_proc0.hdf5"), "w") as file:
        file.create_dataset("time/value", data=t)
        file.create_group("feec/em_fields")  # the raw output names the species,
        file.create_group("kinetic/kinetic_ions")  # as a real run does
        file.create_dataset("scalar/en_tot", data=np.full(NT, 2.0))
    metadata = {
        "model": Maxwell(base_units=BaseUnits(x=2.0)).to_dict(),
        "domain": domains.Cuboid().to_dict(),
        "equil": None,
        "grid": None,
        "derham_opts": None,
        "time_opts": Time().to_dict(),
        "mpi_ranks": 1,
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


def output_with_comm(monkeypatch, path, comm, **kwargs):
    monkeypatch.setattr(output_module, "mpi_comm_world", lambda: comm)
    return Output(path, **kwargs)


def save_raw_field(run, *names):
    """Record ``names`` as raw FEEC fields in the primary output, as a real run saves them."""
    with h5py.File(run.path_out / "data" / "data_proc0.hdf5", "a") as file:
        for name in names:
            file["feec/em_fields"].create_dataset(name, data=np.empty(0))


# Environment prefixes through which MPI launchers tell a process which rank of a job it is.
MPI_LAUNCHER_ENV_PREFIXES = (
    "OMPI_",
    "OPAL_",
    "PMIX_",
    "PMI_",
    "PRTE_",
    "MV2_",
    "HYDRA_",
    "I_MPI_",
    "MPI_LOCALRANKID",
    "ALPS_",
    "PALS_",
)


@pytest.fixture
def outside_mpi_job(monkeypatch):
    """Let child processes start as independent programs, not as ranks of this test's MPI job.

    Importing struphy initializes MPI. A child that inherits the launcher variables of an
    ``mpirun`` rank initializes as that same rank, which hangs or corrupts the parent job.
    """
    for name in list(os.environ):
        if name.startswith(MPI_LAUNCHER_ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MAYBEMPI", "0")


@pytest.fixture
def run(tmp_path):
    return Output(write_tree(str(tmp_path)))


def test_products_are_discovered_without_loading_arrays(run):
    assert tuple(run.fields) == ("em_fields",)
    assert tuple(run.distributions) == ("kinetic_ions",)
    assert tuple(run.orbits) == ("kinetic_ions",)
    assert run.field_catalog._cache == {}
    assert run.is_processed


def test_keys_list_every_evaluable_product_without_loading_arrays(run):
    assert run.keys() == (
        "em_fields/E",
        "en_tot",
        "kinetic_ions",
        "kinetic_ions/e1_v1_density/delta_f",
        "kinetic_ions/e1_v1_density/f",
        "kinetic_ions/view_0/n",
    )
    assert run.field_catalog._cache == {}
    assert run.distribution_catalog._cache == {}


def test_catalog_is_structured_and_report_is_written(run, tmp_path):
    catalog = run.catalog()
    assert list(catalog.product.values) == list(run.keys())
    assert set(catalog.data_vars) == {"kind", "description"}
    assert "dimensions" in run.catalog(details=True)

    report = run.report(tmp_path / "report", products=["en_tot"])
    text = Path(report).read_text()
    assert "Struphy output report" in text
    assert "en_tot" in text


def test_field_has_named_and_curvilinear_coordinates(run):
    field = run.fields["em_fields/E"]
    assert field.dims == ("t", "component", "eta1", "eta2", "eta3")
    assert field.X.dims == ("eta1", "eta2", "eta3")
    np.testing.assert_allclose(field.isel(t=0, component=2), 2)
    assert run.field_catalog._cache["em_fields/E"] is field


def test_binned_products_have_coordinates(run):
    data = run.distributions["kinetic_ions/e1_v1_density/f"]
    assert data.dims == ("t", "eta1", "v1")
    np.testing.assert_allclose(data.v1, np.linspace(-3, 3, NV))


def test_sph_density_views_take_dimensions_from_their_grids(run):
    data = run.densities.kinetic_ions.view_0.n
    assert data.dims == ("t", "eta1", "eta2", "eta3")
    assert data.shape == (NT, N1, N2, 1)
    np.testing.assert_allclose(data.eta2, np.linspace(0, 1, N2))


def test_orbit_product_is_a_dataset_of_named_quantities(run):
    data = run.orbits["kinetic_ions"]
    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars) == ["x", "y", "z", "v1", "v2", "v3", "weight"]
    assert dict(data.sizes) == {"t": NT, "marker": N_MARKERS}
    assert data.x.attrs["description"] == "physical position x"
    assert data.v1.attrs["long_name"] == "$v_x$"
    np.testing.assert_array_equal(data.marker, np.arange(N_MARKERS))


def test_orbits_of_earlier_stores_are_converted_on_read(tmp_path):
    root = write_tree(str(tmp_path))
    path = store.store_path(os.path.join(root, "post_processing"))
    t = np.linspace(0, 1, NT)
    legacy = np.stack([np.full((N_MARKERS, 8), step) for step in range(NT)])
    names = ["x", "y", "z", "v1", "v2", "v3", "weight", "id"]
    store.write_group(
        path,
        "/electrons",
        xr.Dataset(
            {"orbits": (("t", "marker", "quantity"), legacy)},
            coords={"t": t, "marker": np.arange(N_MARKERS), "quantity": names},
        ),
    )
    data = Output(root).orbits["electrons"]
    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars) == names[:-1]
    assert data.x.dims == ("t", "marker")


def test_scalar_time_uses_the_same_policy_as_postprocessed_products(run):
    assert set(run.scalars.data_vars) == {"en_tot"}
    np.testing.assert_allclose(run.scalars.en_tot.t, run.time)


def test_saving_scalars(run, tmp_path):
    path = run.save_scalars(tmp_path / "scalars.csv")
    assert os.path.exists(path)


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


def test_evaluate_raises_without_raw_data(tmp_path):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "data", "data_proc0.hdf5"))
    with pytest.raises(FileNotFoundError, match="No simulation data"):
        Output(root).evaluate("em_fields/E")


def test_evaluate_triggers_default_processing_when_missing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    run = output_with_comm(monkeypatch, root, FakeComm())
    calls = []

    def fake_process(self, *, parallel, **options):
        calls.append(dict(parallel=parallel, **options))
        write_manifest(root)
        self._reset()

    monkeypatch.setattr(Output, "_process", fake_process)
    assert set(run.scalars.data_vars) == {"en_tot"}
    assert calls == [], "scalars come from the raw output"
    assert run.evaluate("em_fields/E").name == "E"
    assert calls == [
        dict(
            parallel=False,
            step=1,
            celldivide=1,
            physical=False,
            guiding_center=False,
            classify=False,
            create_vtk=False,
            force=False,
        )
    ]


def test_evaluate_returns_xarray_and_xarray_exposes_the_product_tree(run):
    field = run.evaluate("em_fields/E")
    assert isinstance(field, xr.DataArray)
    assert field is run.evaluate("em_fields/E")
    assert run.xarray is run.tree


def test_evaluate_selects_positions_coordinates_and_slices(run):
    field = run.evaluate("em_fields/E", t=-1, component=2)
    assert field.dims == ("eta1", "eta2", "eta3")
    assert field.t.ndim == 0, "a single snapshot keeps its time as a scalar coordinate"
    assert run.evaluate("em_fields/E", t=[-1], component=2).dims == ("t", "eta1", "eta2", "eta3")
    np.testing.assert_allclose(field, 3.0)

    every_second = run.evaluate("em_fields/E", t=slice(0, None, 2))
    assert every_second.sizes["t"] == 2

    phase_space = run.evaluate(
        "kinetic_ions/f",
        dataset="e1_v1_density/f",
        eta1=0.49,
        method="nearest",
        drop=True,
    )
    assert phase_space.dims == ("t", "v1")

    history = run.evaluate("scalars", variables="en_tot").en_tot.isel(t=slice(1, None))
    assert history.sizes["t"] == NT - 1

    with pytest.raises(TypeError, match="always returns xarray"):
        run.evaluate("scalars", variables="en_tot", t=-1, as_numpy=True)

    with pytest.raises(ValueError, match="species/variable"):
        run.evaluate("en_tot")


def test_evaluate_scalars_and_particle_defaults(run):
    scalars = run.evaluate("scalars", variables="en_tot", t=-1)
    assert isinstance(scalars, xr.Dataset)
    assert list(scalars.data_vars) == ["en_tot"]
    assert "t" not in scalars.dims

    distribution = run.evaluate("kinetic_ions/f")
    assert distribution.name == "f"
    assert distribution.dims == ("t", "eta1", "v1")

    fallback = run.evaluate("kinetic_ions/any_variable")
    assert fallback.name == "f"

    density = run.evaluate("kinetic_ions/n")
    assert density.name == "n"
    assert density.dims == ("t", "eta1", "eta2", "eta3")

    selected = run.evaluate("kinetic_ions/f", dataset="e1_v1_density/delta_f")
    assert selected.name == "delta_f"

    # the full key selects the same product, with selections as keywords
    by_key = run.evaluate("kinetic_ions/e1_v1_density/delta_f")
    xr.testing.assert_identical(by_key, selected)
    first = float(by_key.eta1[0])
    xr.testing.assert_identical(
        run.evaluate("kinetic_ions/e1_v1_density/delta_f", eta1=first), selected.sel(eta1=first)
    )
    with pytest.raises(ValueError, match="dataset="):
        run.evaluate("kinetic_ions/e1_v1_density/f", dataset="e1_v1_density/f")
    with pytest.raises(KeyError):
        run.evaluate("kinetic_ions/no_such_density/f")

    orbits = run.evaluate("kinetic_ions/orbits")
    assert isinstance(orbits, xr.Dataset) and "weight" in orbits.data_vars


def test_info_lists_particle_dataset_choices_in_default_order(run, capsys):
    run.info("kinetic_ions/f")
    report = capsys.readouterr().out
    assert "Dataset choices (default first):" in report
    assert "kinetic_ions/e1_v1_density/f" in report


def test_evaluate_raw_spline_field_at_logical_point(run, monkeypatch):
    calls = []

    class Field:
        space_id = "H1vec"

        def __call__(self, eta1, eta2, eta3, *, squeeze_out=False):
            calls.append((eta1, eta2, eta3))
            return [np.full((1, 1, 1), eta1), np.full((1, 1, 1), eta2), np.full((1, 1, 1), eta3)]

    field = Field()
    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"e_field": field}})
    save_raw_field(run, "e_field")

    values = run.evaluate("em_fields/e_field", eta1=0.25, eta2=0.5, eta3=0.75, component=2)

    assert values.dims == ("t",)
    np.testing.assert_allclose(values.t, np.arange(NT) / (NT - 1))
    np.testing.assert_allclose(values, 0.75)
    assert calls == [(0.25, 0.5, 0.75)] * NT

    last = run.evaluate("em_fields/e_field", eta1=0.25, eta2=0.5, eta3=0.75, t=-1)
    assert last.dims == ("component",)
    assert float(last.t) == 1.0


def test_evaluate_raw_spline_field_on_mixed_logical_grid(run, monkeypatch):
    class Field:
        space_id = "H1"

        def __call__(self, eta1, eta2, eta3, *, squeeze_out=False):
            eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
            value = eta1 + 10 * eta2 + 100 * eta3
            return value.squeeze() if squeeze_out else value

    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"phi": Field()}})
    save_raw_field(run, "phi")

    values = run.evaluate("em_fields/phi", eta1=[0.25, 0.5], eta2=range(2), eta3=0.75, t=0)

    assert values.dims == ("eta1", "eta2")
    np.testing.assert_allclose(values.eta1, [0.25, 0.5])
    np.testing.assert_allclose(values.eta2, [0.0, 1.0])
    np.testing.assert_allclose(values, [[75.25, 85.25], [75.5, 85.5]])


def test_evaluate_raw_spline_field_defaults_to_simulation_grid_cell_centres(run, monkeypatch):
    class Field:
        space_id = "H1"

        def __call__(self, eta1, eta2, eta3, *, squeeze_out=False):
            return np.ones((len(eta1), len(eta2), len(eta3)))

    with h5py.File(run.path_out / "data" / "data_proc0.hdf5", "a") as file:
        file["feec/em_fields"].create_dataset("phi", data=np.empty(0))

    class Domain:
        def __call__(self, eta1, eta2, eta3):
            eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
            return eta1 + 1.0, eta2 + 2.0, eta3 + 3.0

    run.grid = SimpleNamespace(num_elements=(2, 3, 4))
    run.domain = Domain()
    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"phi": Field()}})

    values = run.evaluate("em_fields/phi", t=0)

    assert values.dims == ("eta1", "eta2", "eta3")
    assert values.shape == (2, 3, 4)
    np.testing.assert_allclose(values.eta1, [0.25, 0.75])
    np.testing.assert_allclose(values.eta2, [1 / 6, 0.5, 5 / 6])
    np.testing.assert_allclose(values.eta3, [0.125, 0.375, 0.625, 0.875])
    assert values.X.dims == values.Y.dims == values.Z.dims == ("eta1", "eta2", "eta3")
    np.testing.assert_allclose(values.X[:, 0, 0], [1.25, 1.75])
    np.testing.assert_allclose(values.Y[0, :, 0], [2 + 1 / 6, 2.5, 2 + 5 / 6])
    np.testing.assert_allclose(values.Z[0, 0, :], [3.125, 3.375, 3.625, 3.875])


def test_evaluate_raw_spline_field_keeps_omitted_directions_on_the_default_grid(run, monkeypatch):
    class Field:
        space_id = "H1"

        def __call__(self, eta1, eta2, eta3, *, squeeze_out=False):
            eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
            return eta1 + 10 * eta2 + 100 * eta3

    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"phi": Field()}})
    save_raw_field(run, "phi")
    run.grid = SimpleNamespace(num_elements=(2, 3, 1))
    eta1_default, eta2_default, _ = run._default_logical_grid()

    plane = run.evaluate("em_fields/phi", t=0, eta3=0)

    assert plane.dims == ("eta1", "eta2")
    np.testing.assert_allclose(plane.eta1, eta1_default)
    np.testing.assert_allclose(plane.eta2, eta2_default)
    np.testing.assert_allclose(plane, eta1_default[:, None] + 10 * eta2_default[None, :])


def test_evaluate_raw_spline_field_rejects_coordinates_outside_unit_cube(run):
    save_raw_field(run, "phi")
    with pytest.raises(ValueError, match="logical unit interval"):
        run.evaluate("em_fields/phi", eta1=-0.01, eta2=0.5, eta3=0.5)


def test_evaluate_raw_spline_field_applies_requested_representation(run, monkeypatch):
    calls = []

    class Field:
        space_id = "L2"

        def __call__(self, *etas, **kwargs):
            return np.ones((1, 1, 1))

    class Domain:
        def transform(self, value, *etas, kind, squeeze_out):
            calls.append((kind, etas, squeeze_out))
            return value

        def __call__(self, eta1, eta2, eta3):
            return np.meshgrid(eta1, eta2, eta3, indexing="ij")

    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"phi": Field()}})
    save_raw_field(run, "phi")
    monkeypatch.setattr(run, "domain", Domain())

    run.evaluate("em_fields/phi", eta1=0.5, eta2=0.5, eta3=0.5, t=0, representation="0")

    assert calls == [("3_to_0", (0.5, 0.5, 0.5), True)]


@pytest.mark.parametrize(
    "domain",
    [
        pytest.param(domains.Cuboid(), id="cuboid"),
        pytest.param(domains.HollowTorus(a1=0.2, a2=0.4, R0=1.0, tor_period=1), id="hollow-torus"),
        pytest.param(domains.Colella(), id="non-orthogonal-colella"),
    ],
)
def test_evaluate_transforms_hcurl_fields_on_mapped_domains(run, monkeypatch, domain):
    """Raw evaluation uses the field's H(curl) source representation on every domain."""
    eta1 = np.linspace(0.2, 0.8, 4)
    eta2 = np.linspace(0.1, 0.9, 5)
    eta3 = 0.25

    class Field:
        space_id = "Hcurl"

        def __call__(self, eta1, eta2, eta3, *, squeeze_out=False):
            eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
            return [1.0 + eta1, 2.0 + eta2, 3.0 + eta3]

    field = Field()
    monkeypatch.setattr(run, "domain", domain)
    monkeypatch.setattr(run, "spline_fields", lambda *, t: {"em_fields": {"e_field": field}})
    save_raw_field(run, "e_field")

    source = field(eta1, eta2, eta3)
    for target in ("1", "2", "v", "norm"):
        result = run.evaluate(
            "em_fields/e_field",
            eta1=eta1,
            eta2=eta2,
            eta3=eta3,
            t=0,
            representation=target,
        )
        expected = (
            source
            if target == "1"
            else domain.transform(
                source,
                eta1,
                eta2,
                eta3,
                kind=f"1_to_{target}",
                squeeze_out=True,
            )
        )
        expected = np.squeeze(np.asarray(expected))
        np.testing.assert_allclose(result, expected)
        assert result.dims == ("component", "eta1", "eta2")


@pytest.mark.parametrize(
    "size, saved_ranks, parallel", [(1, 1, False), (2, 2, True), (4, 4, True), (2, 1, False), (2, 4, False)]
)
def test_first_use_processes_in_parallel_when_the_job_is_as_large_as_the_run(
    tmp_path, monkeypatch, size, saved_ranks, parallel
):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    calls = []
    monkeypatch.setattr(Output, "_setup_processing", lambda self, parallel: calls.append(parallel))
    monkeypatch.setattr(Output, "_process_raw", lambda self, **options: write_manifest(root))
    comm = FakeComm(size=size)
    run = output_with_comm(monkeypatch, root, comm)
    run.metadata["mpi_ranks"] = saved_ranks
    assert tuple(run.fields) == ("em_fields",)
    assert calls == [parallel]
    # no rank looks for products before rank 0 has written them, nor processes before it holds the lock
    assert comm.barriers == (2 if parallel else 1)


@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_passes_parallel_on_to_processing(tmp_path, monkeypatch, parallel):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    calls = []
    monkeypatch.setattr(Output, "_setup_processing", lambda self, parallel: calls.append(parallel))
    monkeypatch.setattr(Output, "_process_raw", lambda self, **options: write_manifest(root))
    run = output_with_comm(monkeypatch, root, FakeComm(size=2))
    run.metadata["mpi_ranks"] = 2
    assert run.evaluate("em_fields/E", parallel=parallel).name == "E"
    assert calls == [parallel]


# Holds the lock in a separate program that loads only the manifest module: importing
# struphy would initialize MPI in the child, which under mpirun makes it join the test's job.
HOLD_LOCK = """
import importlib.util, sys, time

spec = importlib.util.spec_from_file_location("manifest", sys.argv[1])
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)
with manifest.processing_lock(sys.argv[2]):
    with open(sys.argv[3], "a") as stream:
        stream.write("enter\\n")
    time.sleep(0.3)
    with open(sys.argv[3], "a") as stream:
        stream.write("exit\\n")
"""


def test_processing_lock_is_exclusive_between_processes(tmp_path):
    import subprocess
    import sys

    from struphy.post_processing import manifest

    log = tmp_path / "log"
    command = [sys.executable, "-c", HOLD_LOCK, manifest.__file__, str(tmp_path), str(log)]
    workers = [subprocess.Popen(command) for _ in range(3)]
    for worker in workers:
        assert worker.wait(60) == 0
    lines = log.read_text().split()
    assert lines == ["enter", "exit"] * 3  # never two holders at once
    with manifest.processing_lock(str(tmp_path)):
        pass


def test_processing_options_are_part_of_the_manifest(tmp_path):
    root = write_tree(str(tmp_path))
    write_manifest(root, step=1, celldivide=2, physical=False)
    assert is_processed(root)
    assert is_processed(root, dict(step=1, celldivide=(2, 2, 2), physical=False))
    assert not is_processed(root, dict(step=1, celldivide=2, physical=True))


def test_manifest_is_stale_when_raw_output_changes(tmp_path):
    root = write_tree(str(tmp_path))
    os.utime(os.path.join(root, "data", "data_proc0.hdf5"), None)
    assert not is_processed(root)


@pytest.mark.parametrize("rank", [0, 1])
def test_serial_process_runs_on_rank_zero_only(tmp_path, monkeypatch, rank):
    calls = []
    monkeypatch.setattr(Output, "_setup_processing", lambda self, parallel: calls.append(("setup", parallel)))
    monkeypatch.setattr(Output, "_process_raw", lambda self, **options: calls.append(("process", options)))
    comm = FakeComm(rank=rank, size=2)
    run = output_with_comm(monkeypatch, write_tree(str(tmp_path)), comm)
    assert run.pproc(physical=True) is run
    expected = [
        ("setup", False),
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
    calls = []
    monkeypatch.setattr(Output, "_setup_processing", lambda self, parallel: calls.append(parallel))
    monkeypatch.setattr(Output, "_process_raw", lambda self, **options: None)
    output_with_comm(monkeypatch, write_tree(str(tmp_path)), FakeComm(rank=3, size=4)).pproc(parallel=True)
    assert calls == [True]


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("rank", [0, 1])
def test_rank_zero_holds_the_processing_lock_for_the_job(tmp_path, monkeypatch, parallel, rank):
    events = []

    @contextmanager
    def fake_lock(path_out):
        events.append("lock")
        yield
        events.append("unlock")

    comm = FakeComm(rank=rank, size=2)
    comm.Barrier = lambda: events.append("barrier")
    monkeypatch.setattr(output_module, "processing_lock", fake_lock)
    monkeypatch.setattr(Output, "_setup_processing", lambda self, parallel: None)
    monkeypatch.setattr(Output, "_process_raw", lambda self, **options: events.append("process"))
    output_with_comm(monkeypatch, write_tree(str(tmp_path)), comm).pproc(parallel=parallel)

    if parallel:
        # every rank processes, and none starts before rank 0 holds the lock
        held = ["lock", "barrier", "process", "unlock"] if rank == 0 else ["barrier", "process"]
    else:
        held = ["lock", "process", "unlock"] if rank == 0 else []
    assert events == held + ["barrier"]


def test_unknown_species_never_starts_processing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    os.remove(os.path.join(root, "post_processing", "manifest.json"))
    run = Output(root)
    calls = []
    monkeypatch.setattr(Output, "pproc", lambda self, **options: calls.append(options))

    with pytest.raises(AttributeError, match="available species"):
        run.typo_here
    assert not hasattr(run, "anything")
    assert calls == [], "a typo must not post-process the run"
    assert {"em_fields", "kinetic_ions"} <= set(dir(run)), "species are known before processing"


def test_info_lists_evaluable_products_with_descriptions(run, capsys):
    assert run.info() is None
    text = capsys.readouterr().out
    assert "Configuration" not in text and "Propagator options" not in text
    assert "Key" in text and "Description" in text and "Load with" in text and "Hints" in text
    assert "out.evaluate('scalars', variables='en_tot')" in text
    assert "out.evaluate('kinetic_ions/f', dataset='kinetic_ions/e1_v1_density/f')" in text
    assert "out.evaluate('kinetic_ions/orbits')" in text
    assert "en_tot" in text and "scalar time series" in text
    assert "kinetic_ions/e1_v1_density/f" in text and "particle distribution" in text
    assert "kinetic_ions" in text and "marker trajectories" in text
    assert "em_fields/E" in text and "field" in text
    assert run.field_catalog._cache == {}, "listing must not load arrays"


def test_info_evaluate_calls_load_their_keys(run):
    for key in run.keys():
        call = run._evaluate_call(key)
        array = eval(call, {"out": run})
        if key in run.scalars.data_vars:
            array = array[key]
        xr.testing.assert_identical(array, run._product(key))


def test_info_labels_distribution_and_density_symbols(run, capsys):
    run.info()
    text = capsys.readouterr().out
    assert "particle distribution ($f$)" in text
    assert "particle distribution ($\\delta f$)" in text
    assert "SPH density ($n$)" in text


def test_iter_spline_coefficients_reads_one_raw_snapshot_at_a_time(run):
    raw_path = run.path_out / "data" / "data_proc0.hdf5"
    with h5py.File(raw_path, "a") as file:
        file.create_dataset("feec/em_fields/phi", data=np.arange(NT * 2).reshape(NT, 2))
        file.create_group("feec/em_fields/e_field")
        file.create_dataset("feec/em_fields/e_field/1", data=np.full((NT, 2), 1.0))
        file.create_dataset("feec/em_fields/e_field/2", data=np.full((NT, 2), 2.0))

    snapshots = list(run.iter_spline_coefficients(stride=2))
    assert [time for time, _ in snapshots] == [0.0, 1.0]
    assert np.array_equal(snapshots[1][1]["em_fields"]["phi"], np.array([4, 5]))
    assert len(snapshots[0][1]["em_fields"]["e_field"]) == 2
    assert np.array_equal(snapshots[0][1]["em_fields"]["e_field"][1], np.array([2.0, 2.0]))


def test_normalized_time_carries_seconds_as_a_coordinate(run):
    energy = run.scalars.en_tot
    assert "units" not in energy.t.attrs, "normalized time has no unit"
    np.testing.assert_allclose(energy.t_seconds, energy.t * float(run.model.units.t))
    assert energy.t_seconds.attrs["units"] == "s"

    physical = run.with_time_units("physical")
    assert physical is not run
    assert run.time_units == "normalized"
    assert physical.time_units == "physical"
    seconds = physical.scalars.en_tot
    np.testing.assert_allclose(seconds.t, energy.t * float(run.model.units.t))
    assert "t_seconds" not in seconds.coords
    assert "t_seconds" in run.scalars.en_tot.coords

    with pytest.raises(ValueError, match="time_units"):
        run.with_time_units("hours")


def test_a_failing_property_reports_its_own_error(tmp_path):
    root = write_tree(str(tmp_path))
    (tmp_path / "run_metadata.json").unlink()
    run = Output(root)
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        run.domain


def test_parallel_processing_rejects_a_different_rank_count(tmp_path, monkeypatch):
    run = output_with_comm(monkeypatch, write_tree(str(tmp_path)), FakeComm(size=2))
    with pytest.raises(ValueError, match="same number of MPI ranks"):
        run.pproc(parallel=True)


def test_saved_rank_count_does_not_block_serial_implicit_processing(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    run = output_with_comm(monkeypatch, root, FakeComm())
    run.metadata["mpi_ranks"] = 8
    (run.path_pproc / "manifest.json").unlink()
    calls = []
    monkeypatch.setattr(Output, "_process", lambda self, **options: calls.append(self.path_out))
    run._ensure_processed()
    assert calls == [run.path_out]


def test_command_line_lists_keys_and_writes_a_report(tmp_path, monkeypatch, capsys):
    import sys

    from struphy.console.main import struphy

    root = write_tree(str(tmp_path / "run"))
    for argv in (["struphy", "output", "keys", root], ["struphy", "output", "report", root, "--format", "html"]):
        monkeypatch.setattr(sys, "argv", argv)
        struphy()
    printed = capsys.readouterr().out.splitlines()
    assert printed[: len(Output(root).keys())] == list(Output(root).keys())
    assert Path(printed[-1]).name == "report.html" and Path(printed[-1]).exists()


def test_stores_of_schema_version_1_are_read_with_eta_dimensions(tmp_path):
    """Post-processing output written before the rename (e1, e2, e3) keeps working."""
    path = str(tmp_path / "output.nc")
    xr.Dataset(attrs={"schema_version": 1}).to_netcdf(path, mode="w", engine=store.ENGINE)
    legacy = xr.Dataset(
        {"phi": (("t", "e1", "e2"), np.ones((2, 3, 4)))},
        coords={"t": [0.0, 1.0], "e1": np.linspace(0, 1, 3), "e2": np.linspace(0, 1, 4)},
    )
    store.write_group(path, "/em_fields", legacy)
    tree = store.open_tree(path)
    assert tree["em_fields"].ds.phi.dims == ("t", "eta1", "eta2")
    tree.close()
    assert store.SCHEMA_VERSION == 2


def test_processing_lock_falls_back_to_an_exclusive_file(tmp_path, monkeypatch):
    from struphy.post_processing import manifest

    monkeypatch.setattr(manifest, "fcntl", None)
    held = tmp_path / (manifest.LOCK_NAME + ".held")
    with manifest.processing_lock(str(tmp_path)):
        assert held.exists()
    assert not held.exists()


def test_particle_processing_checks_saved_data_per_species(tmp_path, monkeypatch):
    root = write_tree(str(tmp_path))
    with h5py.File(os.path.join(root, "data", "data_proc0.hdf5"), "a") as file:
        del file["kinetic"]
        file.create_group("kinetic/with_data/markers")
        file.create_group("kinetic/with_data/f")
        file.create_group("kinetic/with_data/n_sph")
        file.create_group("kinetic/without_data")
    run = Output(root)
    variable = SimpleNamespace(space="Particles6D")
    species = {name: SimpleNamespace(variables={"var": variable}) for name in ("with_data", "without_data")}
    run.__dict__["model"] = SimpleNamespace(species=species)

    calls = []
    monkeypatch.setattr(Output, "_process_fields", lambda self, **kwargs: None)
    for kind in ("markers", "f", "n_sph"):
        monkeypatch.setattr(
            Output,
            f"_post_process_{kind}",
            lambda self, path, step, kind=kind, **kwargs: calls.append((os.path.basename(path), kind)),
        )
    run._process(parallel=False, force=True)
    assert calls == [("with_data", "markers"), ("with_data", "f"), ("with_data", "n_sph")]
