"""Tests for the views of an Output in the shapes of earlier versions."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from struphy.post_processing.legacy import legacy_views

t = np.array([0.0, 0.1, 0.2])
e1, e2, e3 = np.linspace(0, 1, 4), np.linspace(0, 1, 5), np.array([0.0])


def make_output():
    scalar_field = xr.DataArray(
        np.random.rand(3, 4, 5, 1), dims=("t", "e1", "e2", "e3"), coords={"t": t, "e1": e1, "e2": e2, "e3": e3}
    )
    vector_field = xr.DataArray(
        np.random.rand(3, 3, 4, 5, 1),
        dims=("t", "component", "e1", "e2", "e3"),
        coords={"t": t, "component": [0, 1, 2], "e1": e1, "e2": e2, "e3": e3},
    )
    binned = xr.DataArray(
        np.random.rand(3, 6, 7), dims=("t", "e1", "v1"), coords={"t": t, "e1": np.arange(6), "v1": np.arange(7)}
    )
    return SimpleNamespace(
        orbit_catalog={"ions": xr.DataArray(np.random.rand(3, 8, 8), dims=("t", "marker", "quantity"))},
        field_catalog={
            "em_fields/phi": scalar_field,
            "em_fields/e_field": vector_field,
            "em_fields/e_field_xyz": vector_field,
        },
        distribution_catalog={"ions/e1_v1_density/f": binned, "ions/e1_v1_density/delta_f": binned * 2},
        density_catalog={"fluid/view_0/n": scalar_field},
    )


def test_orbits_are_arrays():
    out = make_output()
    orbits = legacy_views(out).orbits.ions
    assert isinstance(orbits, np.ndarray)
    assert orbits.shape == (3, 8, 8)


def test_fields_map_time_to_a_list_of_components():
    out = make_output()
    fields = legacy_views(out).spline_values.em_fields

    assert sorted(vars(fields)) == ["e_field_log", "e_field_phy", "phi_log"]
    data = fields.phi_log.data
    assert list(data) == [0.0, 0.1, 0.2]
    assert len(data[0.2]) == 1
    np.testing.assert_array_equal(data[0.2][0], out.field_catalog["em_fields/phi"].values[2])

    data = fields.e_field_log.data
    assert len(data[0.1]) == 3
    np.testing.assert_array_equal(data[0.1][2], out.field_catalog["em_fields/e_field"].values[1, 2])
    assert max(fields.phi_log.data) == 0.2


def test_binned_distributions_carry_their_grids():
    out = make_output()
    sli = legacy_views(out).f.ions.e1_v1_density

    np.testing.assert_array_equal(sli.f_binned, out.distribution_catalog["ions/e1_v1_density/f"].values)
    np.testing.assert_array_equal(sli.delta_f_binned, 2 * sli.f_binned)
    np.testing.assert_array_equal(sli.grid_e1, np.arange(6))
    np.testing.assert_array_equal(sli.grid_v1, np.arange(7))
    assert not hasattr(sli, "grid_t")


def test_sph_density_comes_with_its_meshgrid():
    out = make_output()
    view = legacy_views(out).n_sph.fluid.view_0

    assert view.n_sph.shape == (3, 4, 5, 1)
    ee1, ee2, ee3 = view.grid_n_sph
    assert ee1.shape == ee2.shape == ee3.shape == (4, 5, 1)
    np.testing.assert_array_equal(ee1[:, 0, 0], e1)
    np.testing.assert_array_equal(ee2[0, :, 0], e2)
