"""Integration test for loading post-processed data into labeled arrays.

No simulation is run: a post-processing folder is written out by hand in the layout
:meth:`PostProcessor.process` produces, then loaded back. This covers the wiring that
turns files on disk into the objects the plotting scripts index into.
"""

import os
import pickle

import numpy as np
import pytest

from struphy.post_processing.arrays import StruphyArray
from struphy.post_processing.post_processing_tools import PlottingData

NT, N1, N2, N3 = 3, 4, 5, 6
NV = 7
N_MARKERS = 10


def write_pproc_tree(root):
    """Write a minimal post-processing folder, returning the output path."""
    pproc = os.path.join(root, "post_processing")
    fields = os.path.join(pproc, "fields_data")
    kinetic = os.path.join(pproc, "kinetic_data")
    os.makedirs(fields)
    os.makedirs(kinetic)

    t_grid = np.linspace(0.0, 1.0, NT)
    np.save(os.path.join(pproc, "t_grid.npy"), t_grid)

    grids_log = [np.linspace(0, 1, n) for n in (N1, N2, N3)]
    grids_phy = list(np.meshgrid(*grids_log, indexing="ij"))
    for name, grids in (("grids_log", grids_log), ("grids_phy", grids_phy)):
        with open(os.path.join(fields, f"{name}.bin"), "wb") as f:
            pickle.dump(grids, f)

    # a vector field and a scalar field, keyed by time as the post-processor writes them
    species_dir = os.path.join(fields, "em_fields")
    os.makedirs(species_dir)
    vector = {t: [np.full((N1, N2, N3), i + t) for i in range(3)] for t in t_grid}
    scalar = {t: [np.full((N1, N2, N3), t)] for t in t_grid}
    for name, data in (("e_field_log", vector), ("phi_phy", scalar)):
        with open(os.path.join(species_dir, f"{name}.bin"), "wb") as f:
            pickle.dump(data, f)

    # binned distribution function
    slice_dir = os.path.join(kinetic, "kinetic_ions", "distribution_function", "e1_v1_density")
    os.makedirs(slice_dir)
    np.save(os.path.join(slice_dir, "grid_e1.npy"), np.linspace(0, 1, N1))
    np.save(os.path.join(slice_dir, "grid_v1.npy"), np.linspace(-3, 3, NV))
    np.save(os.path.join(slice_dir, "f_binned.npy"), np.ones((NT, N1, NV)))
    np.save(os.path.join(slice_dir, "delta_f_binned.npy"), np.zeros((NT, N1, NV)))

    # marker orbits: one .npy and one .txt per saved step
    orbit_dir = os.path.join(kinetic, "kinetic_ions", "orbits")
    os.makedirs(orbit_dir)
    for step in range(NT):
        np.save(os.path.join(orbit_dir, f"kinetic_ions_{step}.npy"), np.full((N_MARKERS, 8), float(step)))
        open(os.path.join(orbit_dir, f"kinetic_ions_{step}.txt"), "w").close()

    return root


@pytest.fixture
def pdata(tmp_path):
    out = write_pproc_tree(str(tmp_path))
    data = PlottingData(path_out=out)
    data.load()
    return data


def test_load_without_raw_data_skips_scalars(pdata):
    """Scalars come from the raw HDF5, which a post-processing-only folder lacks."""
    assert pdata.scalars.keys() == ()


def test_grids_are_loaded(pdata):
    assert len(pdata.grids_log) == 3
    assert pdata.grids_phy[0].shape == (N1, N2, N3)
    np.testing.assert_allclose(pdata.t_grid, np.linspace(0.0, 1.0, NT))


def test_containers_are_discoverable(pdata):
    """Contents can be listed instead of having to be known in advance."""
    assert "em_fields" in pdata.spline_values
    assert set(pdata.spline_values["em_fields"].keys()) == {"e_field_log", "phi_phy"}
    assert "e1_v1_density" in pdata.f["kinetic_ions"]


def test_field_becomes_one_labeled_array(pdata):
    """The chain the migrated plotting scripts use."""
    field = pdata.spline_values["em_fields"]["e_field_log"].array

    assert field.dims == ("t", "comp", "e1", "e2", "e3")
    assert field.shape == (NT, 3, N1, N2, N3)
    np.testing.assert_allclose(field.coord("e2"), np.linspace(0, 1, N2))
    # component i at time t was filled with i + t
    np.testing.assert_allclose(np.asarray(field.isel(t=0, comp=2)), 2.0)


def test_scalar_field_has_no_component_axis(pdata):
    assert pdata.spline_values["em_fields"]["phi_phy"].array.dims == ("t", "e1", "e2", "e3")


def test_raw_field_dict_still_available(pdata):
    """Existing scripts index ``.data[t][component]`` directly."""
    dd = pdata.spline_values["em_fields"]["e_field_log"]
    assert dd.data[0.0][1].shape == (N1, N2, N3)


def test_binned_data_carries_its_grids(pdata):
    f = pdata.f["kinetic_ions"]["e1_v1_density"]["f_binned"]

    assert isinstance(f, StruphyArray)
    assert f.dims == ("t", "e1", "v1")
    np.testing.assert_allclose(f.coord("v1"), np.linspace(-3, 3, NV))
    assert f[0].T.shape == (NV, N1)  # back-compat indexing


def test_orbits_are_labeled_with_columns(pdata):
    orbits = pdata.orbits["kinetic_ions"]

    assert orbits.dims == ("t", "marker", "attribute")
    assert orbits.shape == (NT, N_MARKERS, 8)
    assert orbits.columns["weight"] == 6
    # step n was filled with the value n
    np.testing.assert_allclose(np.asarray(orbits.isel(t=2)), 2.0)


def test_field_slice_matches_the_physical_grid(pdata):
    """A cut field and its grid must have the same shape, which is what the plotters assume."""
    from struphy.diagnostics.plotting import field_slice_grids

    cut = pdata.spline_values["em_fields"]["phi_phy"].array.isel(t=0, e3=1)
    xgrid, _, _, _ = field_slice_grids(pdata.grids_phy, fixed_dim="e3", index=1, plane="XY")
    assert cut.shape == xgrid.shape
