"""Integration tests for lazy RunOutput discovery."""

import os
import pickle

import h5py
import numpy as np
import pytest

from struphy.post_processing.run_output import RunOutput

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
    orbit_dir = os.path.join(kinetic, "kinetic_ions", "orbits")
    for step in range(NT):
        np.save(os.path.join(orbit_dir, f"kinetic_ions_{step}.npy"), np.full((N_MARKERS, 8), step))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)
    with h5py.File(os.path.join(data_dir, "data_proc0.hdf5"), "w") as file:
        file.create_dataset("time/value", data=t)
        file.create_dataset("scalar/en_tot", data=np.full(NT, 2.0))
    return root


@pytest.fixture
def run(tmp_path):
    return RunOutput.open(write_tree(str(tmp_path)), time_units="normalized")


def test_products_are_discovered_without_loading_arrays(run):
    assert tuple(run.fields) == ("em_fields/E",)
    assert tuple(run.distributions) == ("kinetic_ions/e1_v1_density/f_binned",)
    assert tuple(run.orbits) == ("kinetic_ions",)
    assert run.fields._cache == {}


def test_field_has_named_and_curvilinear_coordinates(run):
    field = run.fields["em_fields/E"]
    assert field.dims == ("t", "component", "e1", "e2", "e3")
    assert field.X.dims == ("e1", "e2", "e3")
    np.testing.assert_allclose(field.isel(t=0, component=2), 2)
    assert run.fields._cache["em_fields/E"] is field


def test_binned_products_have_coordinates(run):
    data = run.distributions["kinetic_ions/e1_v1_density/f_binned"]
    assert data.dims == ("t", "e1", "v1")
    np.testing.assert_allclose(data.v1, np.linspace(-3, 3, NV))


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
