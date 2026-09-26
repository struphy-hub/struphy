"""Contracts for the xarray post-processing representation."""

import numpy as np
import pytest
import xarray as xr

from struphy.post_processing.arrays import (
    axis_label,
    data_array,
    save_scalars,
    scalars_table,
    validate_array,
    value_label,
    wrap_binned_data,
    wrap_field_data,
    wrap_orbits,
)


def test_data_array_carries_names_coordinates_and_units():
    data = data_array(
        np.ones((3, 4)),
        ("t", "e1"),
        {"t": [0, 1, 2], "e1": np.arange(4)},
        name="density",
        label="$n$",
        unit="m^-3",
        coord_units={"t": "s"},
    )
    assert isinstance(data, xr.DataArray)
    assert data.sel(t=1).dims == ("e1",)
    assert axis_label(data, "t") == "$t$ [s]"
    assert value_label(data) == "$n$ [m^-3]"


def test_validation_rejects_missing_dims_and_nonmonotonic_coordinates():
    data = xr.DataArray(np.ones(3), dims="x", coords={"x": [0, 2, 1]})
    with pytest.raises(ValueError, match="monotonic"):
        validate_array(data)
    with pytest.raises(ValueError, match="missing"):
        validate_array(xr.DataArray(np.ones(3), dims="x"), required_dims=("t",))


def test_xarray_arithmetic_preserves_dimension_alignment_and_metadata():
    left = data_array([1, 2], ("t",), {"t": [0, 1]}, label="left", unit="J")
    right = data_array([3, 4], ("t",), {"t": [0, 1]}, label="right", unit="J")
    result = left + right
    assert result.dims == ("t",)
    np.testing.assert_array_equal(result, [4, 6])


def test_field_wrapper_attaches_curvilinear_physical_coordinates():
    logical = [np.linspace(0, 1, n) for n in (2, 3, 4)]
    physical = np.meshgrid(*logical, indexing="ij")
    raw = {0.0: [np.zeros((2, 3, 4))], 1.0: [np.ones((2, 3, 4))]}
    field = wrap_field_data(raw, logical, grids_phy=physical, name="phi", time_scale=2, time_unit="s")
    assert field.dims == ("t", "e1", "e2", "e3")
    assert field.X.dims == ("e1", "e2", "e3")
    np.testing.assert_array_equal(field.t, [0, 2])
    assert field.t.attrs["units"] == "s"


def test_vector_field_has_named_component_dimension():
    component = np.zeros((2, 2, 2))
    field = wrap_field_data({0.0: [component, component, component]}, name="E")
    assert field.dims == ("t", "component", "e1", "e2", "e3")
    assert field.isel(component=1).dims == ("t", "e1", "e2", "e3")


def test_binned_wrapper_keeps_memory_mappable_values():
    values = np.ones((2, 3, 4))
    data = wrap_binned_data(values, ("e1", "v1"), {"t": [0, 1], "e1": range(3), "v1": range(4)}, name="f")
    assert data.dims == ("t", "e1", "v1")
    assert data.attrs["label"] == "$f$"


def test_orbits_are_one_variable_per_quantity():
    quantities = ((0, "x", "$x$", "position x"), (6, "weight", "$w$", "marker weight"))
    data = wrap_orbits(np.arange(20.0).reshape(2, 5, 2), [0, 1], quantities)
    assert list(data.data_vars) == ["x", "weight"]
    assert data.weight.dims == ("t", "marker")
    assert data.weight.attrs["description"] == "marker weight"
    np.testing.assert_array_equal(data.weight.isel(t=0), [1, 3, 5, 7, 9])
    assert data.sel(marker=2).x.dims == ("t",)


def test_scalar_alignment_is_exact():
    good = xr.Dataset({"a": ("t", [1, 2]), "b": ("t", [3, 4])}, coords={"t": [0, 1]})
    time, names, values = scalars_table(good)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(values, [[1, 3], [2, 4]])
    bad = {"a": good.a, "b": xr.DataArray([3, 4], dims="t", coords={"t": [1, 2]})}
    with pytest.raises(ValueError, match="align"):
        scalars_table(bad)


@pytest.mark.parametrize("suffix", ["csv", "npz"])
def test_save_scalars(tmp_path, suffix):
    scalars = xr.Dataset({"a": ("t", [1, 2]), "b": ("t", [3, 4])}, coords={"t": [0, 1]})
    path = save_scalars(scalars, str(tmp_path / f"scalars.{suffix}"))
    assert (tmp_path / f"scalars.{suffix}").stat().st_size > 0
    assert path.endswith(suffix)
