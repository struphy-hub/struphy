"""Tests for distribution reductions, SI conversion and profiling access."""

import os
import time

import numpy as np
import pytest
import xarray as xr

from struphy.diagnostics.analysis import spatial_average, velocity_moments
from struphy.post_processing.arrays import data_array
from struphy.post_processing.output import Output
from struphy.post_processing.tests.test_output import write_tree

F = "kinetic_ions/e1_v1_density/f"


def gaussian(v, density, mean, variance):
    return density * np.exp(-((v - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def binned(values, dims, coords, name="f"):
    return data_array(values, dims, coords, name=name, label="$f$")


@pytest.fixture
def run(tmp_path):
    return Output(write_tree(str(tmp_path)), time_units="normalized")


# --- reductions ---------------------------------------------------------------------------------


def test_moments_of_a_maxwellian_recover_its_parameters():
    v = np.linspace(-8, 8, 321)
    density = np.array([1.0, 2.0])[:, None, None]  # depends on t
    mean = np.array([-0.5, 0.0, 0.5])[None, :, None]  # depends on e1
    f = binned(
        gaussian(v[None, None, :], density, mean, 0.64),
        ("t", "e1", "v1"),
        {"t": [0.0, 1.0], "e1": [0.1, 0.5, 0.9], "v1": v},
    )
    moments = velocity_moments(f)
    assert moments.density.dims == ("t", "e1")
    np.testing.assert_allclose(moments.density, np.broadcast_to(density[:, :, 0], (2, 3)), rtol=1e-8)
    np.testing.assert_allclose(moments.mean_v1, np.broadcast_to(mean[:, :, 0], (2, 3)), atol=1e-8)
    np.testing.assert_allclose(moments.variance_v1, 0.64, rtol=1e-8)


def test_moments_over_two_velocity_dimensions_are_taken_per_direction():
    v1, v2 = np.linspace(-9, 9, 181), np.linspace(-6, 6, 121)
    f = binned(
        (gaussian(v1[:, None], 1.0, 1.0, 0.5) * gaussian(v2[None, :], 3.0, -0.5, 0.25))[None],
        ("t", "v1", "v2"),
        {"t": [0.0], "v1": v1, "v2": v2},
    )
    moments = velocity_moments(f)
    assert set(moments.data_vars) == {"density", "mean_v1", "variance_v1", "mean_v2", "variance_v2"}
    np.testing.assert_allclose(moments.density, 3.0, rtol=1e-8)
    np.testing.assert_allclose(moments.mean_v1, 1.0, atol=1e-8)
    np.testing.assert_allclose(moments.variance_v1, 0.5, rtol=1e-8)
    np.testing.assert_allclose(moments.mean_v2, -0.5, atol=1e-8)
    np.testing.assert_allclose(moments.variance_v2, 0.25, rtol=1e-8)


def test_one_velocity_dimension_can_be_selected():
    v1, v2 = np.linspace(-9, 9, 181), np.linspace(-6, 6, 121)
    f = binned(np.ones((1, 181, 121)), ("t", "v1", "v2"), {"t": [0.0], "v1": v1, "v2": v2})
    moments = velocity_moments(f, dims="v2")
    assert moments.density.dims == ("t", "v1")
    assert "mean_v1" not in moments


def test_delta_f_has_only_a_density():
    v = np.linspace(-3, 3, 7)
    delta_f = binned(np.ones((1, 7)), ("t", "v1"), {"t": [0.0], "v1": v}, name="delta_f")
    assert tuple(velocity_moments(delta_f).data_vars) == ("density",)


def test_mean_and_variance_are_nan_without_particles():
    v = np.linspace(-3, 3, 7)
    f = binned(np.zeros((1, 7)), ("t", "v1"), {"t": [0.0], "v1": v})
    moments = velocity_moments(f)
    assert moments.density.item() == 0.0
    assert np.isnan(moments.mean_v1.item()) and np.isnan(moments.variance_v1.item())


def test_moments_carry_the_run_and_a_label():
    f = binned(np.ones((1, 7)), ("t", "v1"), {"t": [0.0], "v1": np.linspace(-3, 3, 7)})
    f.attrs.update(run="dt=0.1", run_name="sim_1")
    moments = velocity_moments(f)
    assert moments.attrs["run_name"] == "sim_1"
    assert moments.mean_v1.attrs["run"] == "dt=0.1"
    assert moments.density.attrs["label"] == "density of $f$"


def test_moments_reject_missing_velocity_dimensions_and_single_bins():
    no_velocity = binned(np.ones((2, 3)), ("t", "e1"), {"t": [0.0, 1.0], "e1": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="none of the dimensions"):
        velocity_moments(no_velocity)
    with pytest.raises(ValueError, match="no dimensions"):
        velocity_moments(no_velocity, dims="v1")
    one_bin = binned(np.ones((1, 1)), ("t", "v1"), {"t": [0.0], "v1": [0.0]})
    with pytest.raises(ValueError, match="at least two bins"):
        velocity_moments(one_bin)


def test_spatial_average_removes_the_space_dimensions_only():
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    f = binned(values, ("t", "e1", "v1"), {"t": [0.0, 1.0], "e1": [0.1, 0.5, 0.9], "v1": np.arange(4.0)})
    f.attrs["run_name"] = "sim_1"
    mean = spatial_average(f)
    assert mean.dims == ("t", "v1")
    np.testing.assert_allclose(mean, values.mean(axis=1))
    assert mean.attrs["run_name"] == "sim_1"
    assert mean.attrs["label"] == "average of $f$"
    assert spatial_average(f, dims="e1").dims == ("t", "v1")


def test_spatial_average_drops_physical_coordinates_it_averaged_over():
    logical = {f"e{i + 1}": np.linspace(0, 1, n) for i, n in enumerate((3, 4, 1))}
    mapped = np.meshgrid(*logical.values(), indexing="ij")
    field = data_array(
        np.ones((2, 3, 4, 1)),
        ("t", "e1", "e2", "e3"),
        {"t": [0.0, 1.0], **logical, "X": (("e1", "e2", "e3"), mapped[0])},
        name="E",
    )
    mean = spatial_average(field)
    assert mean.dims == ("t",)
    assert "X" not in mean.coords


def test_spatial_average_needs_space_dimensions():
    series = data_array(np.ones(3), ("t",), {"t": [0.0, 1.0, 2.0]}, name="energy")
    with pytest.raises(ValueError, match="none of the dimensions"):
        spatial_average(series)


def test_reductions_are_available_from_the_run_and_the_accessor(run):
    moments = run.velocity_moments(F)
    # write_tree has f = 1 on v = -3..3 in unit bins: n = 7, u = 0 and <v^2> = 4
    np.testing.assert_allclose(moments.density, 7.0)
    np.testing.assert_allclose(moments.mean_v1, 0.0, atol=1e-12)
    np.testing.assert_allclose(moments.variance_v1, 4.0)
    assert moments.density.attrs["run_name"] == run.path_out.name

    average = run.spatial_average(F)
    assert average.dims == ("t", "v1")
    xr.testing.assert_identical(average, run[F].struphy.analysis.spatial_average())
    xr.testing.assert_identical(moments, run[F].struphy.analysis.velocity_moments())


# --- SI units ---------------------------------------------------------------------------------


def test_coordinates_are_converted_and_values_left_alone(run):
    units = run.units
    f = run.to_si(F)
    np.testing.assert_allclose(f.v1, run[F].v1 * units.v)
    assert f.v1.attrs["units"] == "m/s"
    np.testing.assert_allclose(f.t, run[F].t * units.t)
    assert f.t.attrs["units"] == "s"
    assert "t_seconds" not in f.coords
    np.testing.assert_array_equal(f.e1, run[F].e1)  # logical coordinates are dimensionless
    np.testing.assert_array_equal(f, run[F])
    assert "units" not in f.attrs
    assert run[F].v1.attrs.get("units") is None  # the run's own product is untouched
    assert "t_seconds" in run[F].coords


def test_mapped_coordinates_are_scaled_by_the_length_unit(run):
    assert run.units.x == 2.0
    field = run.to_si("em_fields/E")
    np.testing.assert_allclose(field.X, run["em_fields/E"].X * 2.0)
    assert field.X.attrs["units"] == "m"


def test_values_are_converted_with_a_named_unit(run):
    field = run.to_si("em_fields/E", "B")
    np.testing.assert_allclose(field, run["em_fields/E"] * run.units.B)
    assert field.attrs["units"] == "T"
    assert field.name == "E"
    assert field.attrs["run_name"] == run.path_out.name


def test_values_are_converted_with_a_composite_unit(run):
    field = run.to_si("em_fields/E", run.units.v * run.units.B, label="V/m")
    np.testing.assert_allclose(field, run["em_fields/E"] * run.units.v * run.units.B)
    assert field.attrs["units"] == "V/m"


def test_conversion_is_idempotent_for_coordinates_and_refuses_values_twice(run):
    once = run.to_si(F)
    np.testing.assert_array_equal(run.to_si(once).v1, once.v1)
    converted = run.to_si("em_fields/E", "B")
    with pytest.raises(ValueError, match="already has units"):
        run.to_si(converted, "B")


def test_unknown_units_are_rejected(run):
    with pytest.raises(ValueError, match="unknown unit"):
        run.to_si("em_fields/E", "furlong")


def test_physical_time_units_are_not_converted_twice(tmp_path):
    physical = Output(write_tree(str(tmp_path)), time_units="physical")
    np.testing.assert_allclose(physical.to_si(F).t, physical[F].t)


# --- profiling --------------------------------------------------------------------------------


def write_profile(path_out, *, calls=3, setup=True):
    from scope_profiler import ProfileManager, ProfilingOptions

    with ProfileManager.session(
        options=ProfilingOptions(),
        deactivate_profiling=False,
        file_path=os.path.join(path_out, "profiling_data.h5"),
    ):
        if setup:
            with ProfileManager.profile_region("setup: total"):
                time.sleep(0.001)
        for _ in range(calls):
            with ProfileManager.profile_region("prop: A"):
                with ProfileManager.profile_region("kernel: k"):
                    time.sleep(0.01)


@pytest.fixture
def profiled(tmp_path, capfd):
    write_profile(write_tree(str(tmp_path)))
    return Output(str(tmp_path))


def test_a_run_without_profiling_says_how_to_enable_it(run):
    with pytest.raises(FileNotFoundError, match="profiling_activated=True"):
        run.profile


def test_summary_lists_every_region_with_times(profiled):
    summary = profiled.profile.summary()
    assert summary.sizes == {"region": 4}
    assert list(summary.region.values[:1]) == ["scope_profiler.session"]
    assert summary.region.values[-1] == "setup: total"
    assert summary.calls.sel(region="prop: A").item() == 3
    assert summary.calls.sel(region="setup: total").item() == 1
    assert summary.total_time.sel(region="kernel: k").item() >= 0.03
    assert summary.mean_time.sel(region="kernel: k").item() >= 0.01
    assert summary.fraction.sel(region="scope_profiler.session").item() == pytest.approx(1.0)
    assert 0 < summary.fraction.sel(region="prop: A").item() <= 1.0
    assert summary.attrs["run"] == profiled.label
    assert summary.attrs["num_ranks"] == 1
    assert summary.total_time.attrs["units"] == "s"


def test_summary_filters_and_sorts(profiled):
    profile = profiled.profile
    assert list(profile.summary(prefix="kernel:").region.values) == ["kernel: k"]
    assert len(profile.summary(top=2).region) == 2
    by_calls = profile.summary(sort_by="calls")
    assert by_calls.region.values[0] == "prop: A"
    with pytest.raises(ValueError, match="cannot sort by"):
        profile.summary(sort_by="size")


def test_the_profile_is_cached_and_reset_with_the_output(profiled):
    assert profiled.profile is profiled.profile
    first = profiled.profile
    profiled.clear_cache()
    assert profiled.profile is not first


def test_table_is_readable_text(profiled):
    table = profiled.profile.table(top=3)
    lines = table.splitlines()
    assert lines[0].startswith("Profile: ") and "1 rank(s)" in lines[0]
    assert "Total [s]" in lines[2]
    assert len(lines) == 4 + 3
    assert "scope_profiler.session" in table and "100.0%" in table


def test_runs_are_compared_side_by_side(tmp_path, capfd):
    first = Output(write_tree(str(tmp_path / "a")))
    second = Output(write_tree(str(tmp_path / "b")))
    write_profile(first.path_out, calls=3)
    write_profile(second.path_out, calls=2, setup=False)

    table = first.profile.compare(second, metric="calls")
    assert set(table.dims) == {"region", "run"}
    assert list(table.run.values) == [f"{first.label} [a]", f"{second.label} [b]"]  # identical labels are told apart
    assert table.sel(run=table.run.values[0], region="prop: A").item() == 3
    assert table.sel(run=table.run.values[1], region="prop: A").item() == 2
    assert np.isnan(table.sel(run=table.run.values[1], region="setup: total").item())
    assert table.name == "calls"
    assert first.profile.compare(second, prefix="kernel:").region.values.tolist() == ["kernel: k"]
    assert first.profile.compare(second.profile).shape == table.shape  # a Profile works as well as an Output

    with pytest.raises(ValueError, match="cannot compare"):
        first.profile.compare(second, metric="size")
