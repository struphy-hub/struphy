"""Unit tests for the labeled arrays used to hand post-processed data to the plotters.

None of these need a simulation: they build small arrays by hand and check that the
dimension bookkeeping, coordinate pairing and back-compatible indexing behave.
"""

import os

import numpy as np
import pytest

from struphy.post_processing.arrays import (
    StruphyArray,
    orbit_columns,
    save_scalars,
    scalar_names,
    scalars_table,
    wrap_binned_slice,
    wrap_field_data,
    wrap_orbits,
)


class Holder:
    """Stand-in for the ``Slice`` container the loader populates by setattr."""


def make_array():
    return StruphyArray(
        np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
        dims=("t", "e1", "v1"),
        coords={"t": np.array([0.0, 1.0]), "e1": np.linspace(0, 1, 3), "v1": np.linspace(-1, 1, 4)},
        label="$f$",
    )


# ---------------------------------------------------------------- StruphyArray


def test_rank_must_match_dims():
    with pytest.raises(ValueError, match="rank"):
        StruphyArray(np.zeros((2, 3)), dims=("t",))


def test_coord_must_match_axis_length():
    with pytest.raises(ValueError, match="shape"):
        StruphyArray(np.zeros((2, 3)), dims=("t", "e1"), coords={"e1": np.zeros(7)})


def test_coord_must_name_a_dim():
    with pytest.raises(ValueError, match="not one of the dims"):
        StruphyArray(np.zeros(2), dims=("t",), coords={"e1": np.zeros(2)})


def test_behaves_as_a_plain_array():
    """Existing code that indexes or reduces the raw arrays must keep working."""
    f = make_array()
    assert np.asarray(f).shape == (2, 3, 4)
    assert f[1].T.shape == (4, 3)
    assert np.sum(f) == pytest.approx(np.sum(np.arange(24)))
    assert len(f) == 2


def test_isel_drops_int_axes_and_keeps_slices():
    f = make_array()

    dropped = f.isel(t=0)
    assert dropped.dims == ("e1", "v1")
    assert "t" not in dropped.coords

    kept = f.isel(t=slice(0, 1))
    assert kept.dims == ("t", "e1", "v1")
    assert kept.shape == (1, 3, 4)


def test_isel_subsets_the_coordinate_of_a_sliced_dim():
    f = make_array()
    sub = f.isel(v1=slice(1, 3))
    assert sub.shape[-1] == 2
    np.testing.assert_allclose(sub.coords["v1"], f.coords["v1"][1:3])


def test_at_picks_the_nearest_coordinate():
    """Replaces the ``abs(t_grid - t).argmin()`` idiom, including ties away from a node."""
    f = make_array()
    np.testing.assert_allclose(np.asarray(f.at(t=0.4)), np.asarray(f.isel(t=0)))
    np.testing.assert_allclose(np.asarray(f.at(t=0.9)), np.asarray(f.isel(t=1)))


def test_transpose_to_reorders_values_and_dims():
    f = make_array().isel(t=0)
    tr = f.transpose_to("v1", "e1")
    assert tr.dims == ("v1", "e1")
    np.testing.assert_allclose(np.asarray(tr), np.asarray(f).T)


def test_transpose_to_rejects_a_different_dim_set():
    with pytest.raises(ValueError, match="cannot transpose"):
        make_array().transpose_to("t", "e1")


def test_coord_falls_back_to_an_index_range():
    f = StruphyArray(np.zeros((2, 3)), dims=("t", "e1"))
    np.testing.assert_allclose(f.coord("e1"), np.arange(3))


def test_axis_label_includes_units_when_known():
    f = make_array().with_coord_units(t="s")
    assert f.axis_label("t") == "$t$ [s]"
    assert f.axis_label("e1") == r"$\eta_1$"
    assert f.value_label == "$f$ [a.u.]"


def test_coord_units_survive_selection():
    f = make_array().with_coord_units(t="s")
    assert f.isel(e1=0).coord_units == {"t": "s"}
    assert f.isel(t=0).transpose_to("v1", "e1").coord_units == {"t": "s"}


def test_unknown_axis_raises():
    with pytest.raises(KeyError):
        make_array().axis("nope")


# ---------------------------------------------------------------- orbit columns


@pytest.mark.parametrize(
    "n_columns, expect_weight",
    [(8, 6), (5, None), (6, None)],
)
def test_orbit_columns_resolves_weight_from_width(n_columns, expect_weight):
    """The saved marker columns depend on the species' velocity dimension.

    A 1V species saves no weight at all, so reading index 6 unconditionally would
    silently return a velocity component instead.
    """
    cols = orbit_columns(n_columns)
    assert cols["position"] == slice(0, 3)
    assert cols["id"] == n_columns - 1
    assert cols.get("weight") == expect_weight


def test_wrap_orbits_attaches_columns():
    orb = wrap_orbits(np.zeros((4, 10, 8)), np.arange(4.0))
    assert orb.dims == ("t", "marker", "attribute")
    assert orb.columns["weight"] == 6


# ---------------------------------------------------------------- binned slices


def test_wrap_binned_slice_pairs_grids_with_data():
    holder = Holder()
    holder.grid_e1 = np.linspace(0, 1, 3)
    holder.grid_v1 = np.linspace(-1, 1, 4)
    holder.f_binned = np.zeros((2, 3, 4))
    holder.delta_f_binned = np.zeros((2, 3, 4))

    wrap_binned_slice(holder, "e1_v1_density", np.array([0.0, 1.0]))

    assert holder.f_binned.dims == ("t", "e1", "v1")
    np.testing.assert_allclose(holder.f_binned.coord("v1"), np.linspace(-1, 1, 4))
    assert holder.f_binned.label == "$f$"
    assert holder.delta_f_binned.label == r"$\delta f$"
    # the grids themselves stay raw
    assert isinstance(holder.grid_e1, np.ndarray)


def test_wrap_binned_slice_takes_dim_order_from_the_name():
    """``v1_v2_density`` must map axis 0 to v1, not to whichever grid was set first."""
    holder = Holder()
    holder.grid_v2 = np.linspace(0, 1, 5)
    holder.grid_v1 = np.linspace(0, 1, 3)
    holder.f_binned = np.zeros((2, 3, 5))

    wrap_binned_slice(holder, "v1_v2_density", np.array([0.0, 1.0]))
    assert holder.f_binned.dims == ("t", "v1", "v2")


def test_wrap_binned_slice_handles_one_dimensional_binning():
    holder = Holder()
    holder.grid_e1 = np.linspace(0, 1, 6)
    holder.f_binned = np.zeros((2, 6))

    wrap_binned_slice(holder, "e1_current_1", np.array([0.0, 1.0]))
    assert holder.f_binned.dims == ("t", "e1")


def test_wrap_binned_slice_leaves_mismatched_entries_alone():
    holder = Holder()
    holder.grid_e1 = np.linspace(0, 1, 3)
    holder.other = np.zeros((9, 9))

    wrap_binned_slice(holder, "e1_density", np.array([0.0, 1.0]))
    assert isinstance(holder.other, np.ndarray)


def test_wrap_binned_slice_without_grids_is_a_noop():
    holder = Holder()
    holder.f_binned = np.zeros((2, 3))
    wrap_binned_slice(holder, "whatever", np.array([0.0, 1.0]))
    assert isinstance(holder.f_binned, np.ndarray)


# ---------------------------------------------------------------- field data


def test_wrap_field_data_stacks_vector_components():
    grids = [np.linspace(0, 1, n) for n in (2, 3, 4)]
    data = {0.0: [np.zeros((2, 3, 4)) for _ in range(3)], 1.0: [np.ones((2, 3, 4)) for _ in range(3)]}

    arr = wrap_field_data(data, grids, label="B")
    assert arr.dims == ("t", "comp", "e1", "e2", "e3")
    assert arr.shape == (2, 3, 2, 3, 4)
    np.testing.assert_allclose(arr.coord("t"), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(arr.isel(t=1, comp=0)), 1.0)


def test_wrap_field_data_drops_comp_for_scalars():
    grids = [np.linspace(0, 1, n) for n in (2, 3, 4)]
    arr = wrap_field_data({0.0: [np.zeros((2, 3, 4))]}, grids, label="phi")
    assert arr.dims == ("t", "e1", "e2", "e3")


def test_wrap_field_data_sorts_times():
    grids = [np.linspace(0, 1, n) for n in (2, 2, 2)]
    data = {1.0: [np.ones((2, 2, 2))], 0.0: [np.zeros((2, 2, 2))]}
    arr = wrap_field_data(data, grids)
    np.testing.assert_allclose(arr.coord("t"), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(arr.isel(t=0)), 0.0)


def test_wrap_field_data_of_empty_dict_is_none():
    assert wrap_field_data({}, None) is None


# ---------------------------------------------------------------- scalar export

NT_SCALARS = 5


def make_scalars(**extra):
    """The shape of ``PlottingData.scalars``: a name -> time series mapping."""
    t = np.linspace(0.0, 2.0, NT_SCALARS)
    scalars = {
        "en_tot": StruphyArray(np.full(NT_SCALARS, 3.0), dims=("t",), coords={"t": t}, label="en tot"),
        "en_e": StruphyArray(np.linspace(1.0, 2.0, NT_SCALARS), dims=("t",), coords={"t": t}, label="en e"),
        "time": StruphyArray(t, dims=("t",), coords={"t": t}),
    }
    scalars.update(extra)
    return scalars


def test_scalar_names_drops_the_excluded_ones():
    assert scalar_names(make_scalars()) == ["en_tot", "en_e"]


def test_scalar_names_keeps_the_requested_order():
    assert scalar_names(make_scalars(), names=["en_e", "time"]) == ["en_e", "time"]


def test_scalar_names_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="no scalars"):
        scalar_names(make_scalars(), names=["en_nope"])


def test_scalars_table_is_one_column_per_scalar():
    t, names, values = scalars_table(make_scalars())

    assert names == ["en_tot", "en_e"]
    assert values.shape == (NT_SCALARS, 2)
    np.testing.assert_allclose(t, np.linspace(0.0, 2.0, NT_SCALARS))
    np.testing.assert_allclose(values[:, 0], 3.0)
    np.testing.assert_allclose(values[:, 1], np.linspace(1.0, 2.0, NT_SCALARS))


def test_scalars_table_drops_a_series_of_the_wrong_length():
    """A short series cannot share the time column, and must not corrupt the table."""
    odd = StruphyArray(np.zeros(2), dims=("t",), coords={"t": np.zeros(2)})
    t, names, values = scalars_table(make_scalars(odd_one=odd))

    assert "odd_one" not in names
    assert values.shape == (NT_SCALARS, 2)


def test_scalars_table_of_nothing_is_empty():
    t, names, values = scalars_table({})
    assert names == [] and len(t) == 0


def test_save_scalars_writes_a_csv_with_a_header(tmp_path):
    path = save_scalars(make_scalars(), str(tmp_path / "scalars.csv"))
    lines = open(path).read().splitlines()

    assert lines[0] == "t,en_tot,en_e"
    assert len(lines) == NT_SCALARS + 1
    # every row is one time step: t, then one value per scalar
    first = [float(v) for v in lines[1].split(",")]
    assert first == pytest.approx([0.0, 3.0, 1.0])


def test_save_scalars_round_trips_through_npz(tmp_path):
    path = save_scalars(make_scalars(), str(tmp_path / "scalars.npz"))
    loaded = np.load(path)

    assert set(loaded.files) == {"t", "en_tot", "en_e"}
    np.testing.assert_allclose(loaded["en_e"], np.linspace(1.0, 2.0, NT_SCALARS))


def test_save_scalars_takes_the_format_from_the_suffix(tmp_path):
    npz = save_scalars(make_scalars(), str(tmp_path / "table"), fmt="npz")
    assert npz.endswith(".npz") and os.path.exists(npz)


def test_save_scalars_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        save_scalars(make_scalars(), str(tmp_path / "scalars.xlsx"))


def test_save_scalars_creates_the_directory(tmp_path):
    path = save_scalars(make_scalars(), str(tmp_path / "new" / "dir" / "scalars.csv"))
    assert os.path.exists(path)
