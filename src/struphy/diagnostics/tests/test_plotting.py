"""Unit tests for the standardized plotters.

These render into the Agg backend, so they check the geometry and labeling that the
plotters derive from the data rather than the appearance of the result.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from struphy.diagnostics.plotting import (  # noqa: E402
    PLANES,
    AnimationPlot,
    MarkerTrajectoryPlot,
    PanelGridPlot,
    ScalarsPlot,
    Slice2DPlot,
    SliderPlot,
    TimeSeriesPlot,
    drift,
    field_slice_grids,
    growth_rate,
    logical_grids,
    match_to_grid,
    relative_error,
    save_all_scalars,
)
from struphy.post_processing.arrays import StruphyArray, wrap_orbits  # noqa: E402

# FuncAnimation warns when it is collected without having been rendered, which is
# exactly what happens to the animations these tests build and discard.
pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def phase_space(nt=12, n1=6, nv=8):
    return StruphyArray(
        np.random.default_rng(0).random((nt, n1, nv)),
        dims=("t", "e1", "v1"),
        coords={"t": np.linspace(0, 1, nt), "e1": np.linspace(0, 1, n1), "v1": np.linspace(-3, 3, nv)},
        label="$f$",
    )


def meshgrids(n1=6, n2=7, n3=5):
    return np.meshgrid(
        np.linspace(1.0, 2.0, n1),
        np.linspace(0.0, 2 * np.pi, n2),
        np.linspace(-1.0, 1.0, n3),
        indexing="ij",
    )


# ---------------------------------------------------------------- growth rate


def test_growth_rate_recovers_a_known_exponential():
    t = np.linspace(0, 10, 100)
    y = StruphyArray(1e-6 * np.exp(0.3 * t), dims=("t",), coords={"t": t})

    gamma, b, window = growth_rate(y)
    assert gamma == pytest.approx(0.3)
    assert np.exp(b) == pytest.approx(1e-6, rel=1e-6)


def test_growth_rate_of_sqrt_halves_the_exponent():
    """An energy grows at twice the rate of the amplitude it is quadratic in."""
    t = np.linspace(0, 10, 100)
    y = StruphyArray(np.exp(0.3 * t), dims=("t",), coords={"t": t})
    assert growth_rate(y, of_sqrt=True)[0] == pytest.approx(0.15)


def test_growth_rate_honours_the_window():
    t = np.linspace(0, 10, 101)
    y = StruphyArray(np.exp(0.3 * t), dims=("t",), coords={"t": t})
    _, _, window = growth_rate(y, t0=2.0, t1=4.0)
    assert t[window][0] >= 2.0 and t[window][-1] <= 4.0


def test_growth_rate_ignores_non_positive_samples():
    t = np.linspace(0, 10, 50)
    values = np.exp(0.3 * t)
    values[:5] = -1.0
    gamma, _, window = growth_rate(StruphyArray(values, dims=("t",), coords={"t": t}))
    assert window.start >= 5
    assert gamma == pytest.approx(0.3)


def test_growth_rate_gives_up_cleanly_on_degenerate_input():
    t = np.linspace(0, 1, 4)
    y = StruphyArray(-np.ones(4), dims=("t",), coords={"t": t})
    assert growth_rate(y) == (None, None, None)


# ---------------------------------------------------------------- grid helpers


def test_match_to_grid_transposes_when_that_is_what_fits():
    grid = np.zeros((3, 4))
    np.testing.assert_allclose(match_to_grid(np.zeros((4, 3)), grid).shape, (3, 4))
    np.testing.assert_allclose(match_to_grid(np.zeros((3, 4)), grid).shape, (3, 4))


def test_match_to_grid_rejects_an_incompatible_shape():
    with pytest.raises(ValueError, match="cannot match"):
        match_to_grid(np.zeros((5, 9)), np.zeros((3, 4)))


def test_logical_grids_uses_the_non_time_dims():
    """Phase-space slices are (e1, v1), not two ``e<i>`` axes."""
    xgrid, ygrid, xlabel, ylabel = logical_grids(phase_space().isel(t=0))
    assert xgrid.shape == (6, 8)
    assert xlabel == r"$\eta_1$"
    assert ylabel == "$v_1$"


def test_logical_grids_needs_exactly_two_plotted_dims():
    three_d = StruphyArray(np.zeros((2, 3, 4, 5)), dims=("t", "e1", "e2", "e3"))
    with pytest.raises(ValueError, match="two non-time dims"):
        logical_grids(three_d)


@pytest.mark.parametrize("plane", sorted(PLANES))
def test_field_slice_grids_returns_each_plane(plane):
    xgrid, ygrid, xlabel, ylabel = field_slice_grids(meshgrids(), fixed_dim="e3", index=0, plane=plane)
    assert xgrid.shape == ygrid.shape == (6, 7)
    assert (xlabel, ylabel) == (PLANES[plane][2], PLANES[plane][3])


@pytest.mark.parametrize(
    "fixed_dim, expected",
    [("e1", (7, 5)), ("e2", (6, 5)), ("e3", (6, 7))],
)
def test_field_slice_grids_slices_the_named_axis(fixed_dim, expected):
    """Regression: the copy-pasted versions of this disagreed on which axis to cut.

    ``pproc_cyclone`` cut ``arr[:, index, :]`` for the same case where
    ``pproc_drift_kinetic`` cut ``arr[:, :, index]``, so one of the two silently
    plotted the wrong slice.
    """
    xgrid, _, _, _ = field_slice_grids(meshgrids(), fixed_dim=fixed_dim, index=0, plane="XY")
    assert xgrid.shape == expected


@pytest.mark.parametrize("fixed_dim", ["e1", "e2", "e3"])
def test_field_slice_grids_agrees_with_isel(fixed_dim):
    """The grid and the field must be cut on the same axis, by construction."""
    grids = meshgrids()
    field = StruphyArray(
        np.random.default_rng(1).random((2, 6, 7, 5)),
        dims=("t", "e1", "e2", "e3"),
    )
    sliced = field.isel(t=0, **{fixed_dim: 1})
    xgrid, _, _, _ = field_slice_grids(grids, fixed_dim=fixed_dim, index=1, plane="XY")
    assert sliced.shape == xgrid.shape


def test_field_slice_grids_rejects_unknown_inputs():
    with pytest.raises(ValueError, match="unknown plane"):
        field_slice_grids(meshgrids(), plane="QQ")
    with pytest.raises(ValueError, match="fixed_dim"):
        field_slice_grids(meshgrids(), fixed_dim="e9")


# ---------------------------------------------------------------- plotters


def test_time_series_labels_axes_from_the_data():
    t = np.linspace(0, 10, 40)
    y = StruphyArray(np.exp(0.3 * t), dims=("t",), coords={"t": t}, label="energy").with_coord_units(t="s")

    plot = TimeSeriesPlot(y, fit=True, title="Energy").plot()
    assert plot.ax.get_xlabel() == "$t$ [s]"
    assert plot.ax.get_ylabel() == "energy [a.u.]"
    assert plot.ax.get_yscale() == "log"
    assert plot.fit_results[0][0] == pytest.approx(0.3)


def test_time_series_fits_every_series():
    """Comparing runs means each curve gets its own rate, not just the first."""
    t = np.linspace(0, 10, 60)
    series = [StruphyArray(np.exp(rate * t), dims=("t",), coords={"t": t}, label=f"run {rate}") for rate in (0.2, 0.4)]
    plot = TimeSeriesPlot(series, fit=True).plot()
    assert [f[0] for f in plot.fit_results] == pytest.approx([0.2, 0.4])


def test_time_series_draws_every_series():
    t = np.linspace(0, 1, 10)
    a = StruphyArray(np.ones(10), dims=("t",), coords={"t": t}, label="a")
    b = StruphyArray(np.ones(10) * 2, dims=("t",), coords={"t": t}, label="b")
    assert len(TimeSeriesPlot([a, b], fit=False).plot().ax.get_lines()) == 2


def test_slice_2d_draws_into_a_supplied_axes():
    fig, ax = plt.subplots()
    plot = Slice2DPlot(phase_space().isel(t=0), ax=ax).plot()
    assert plot.ax is ax
    assert plot.mesh is not None


def test_panel_grid_spreads_panels_over_the_run():
    plot = PanelGridPlot(phase_space(), nrows=2, ncols=3, shared_clim=True).plot()
    axes = plot.ax.ravel()
    assert len(axes) == 6
    # first and last panel are the first and last time step
    assert axes[0].get_title().endswith("0.00e+00")
    assert axes[-1].get_title().endswith("1.00e+00")


def test_panel_grid_shares_the_colour_range_when_asked():
    plot = PanelGridPlot(phase_space(), nrows=1, ncols=2, shared_clim=True).plot()
    clims = {tuple(c.get_clim()) for ax in plot.ax.ravel() for c in ax.collections}
    assert len(clims) == 1


def test_slider_plot_adds_a_second_slider_for_a_free_axis():
    two_d = phase_space()
    assert len(SliderPlot(two_d).plot().sliders) == 1

    three_d = StruphyArray(np.zeros((4, 5, 6, 7)), dims=("t", "e1", "e2", "e3"))
    plot = SliderPlot(three_d).plot()
    assert plot.slice_dim == "e3"
    assert len(plot.sliders) == 2


def test_slider_grids_may_follow_the_cut():
    """A physical grid that depends on where the cut is taken must not go stale."""
    field = StruphyArray(np.zeros((3, 6, 7, 5)), dims=("t", "e1", "e2", "e3"))
    asked = []

    def grids(index):
        asked.append(index)
        return field_slice_grids(meshgrids(), fixed_dim="e3", index=index, plane="XY")

    plot = SliderPlot(field, grids=grids, slice_dim="e3").plot()
    assert plot.slice_dim == "e3"
    # built at the initial cut, and re-queried when the slider moves
    assert asked == [2]

    plot.sliders[1].set_val(4)
    assert asked[-1] == 4


def test_slider_time_updates_the_title():
    field = StruphyArray(np.zeros((3, 6, 7, 5)), dims=("t", "e1", "e2", "e3"))
    grids = field_slice_grids(meshgrids(), fixed_dim="e3", index=0, plane="XY")
    plot = SliderPlot(field, grids=grids, slice_dim="e3", title="phi").plot()

    plot.sliders[0].set_val(2)
    assert plot.ax.get_title() == "phi at t = 2.0000e+00"


def test_marker_trajectory_handles_a_species_without_weights():
    with_weight = wrap_orbits(np.random.default_rng(2).random((5, 20, 8)), np.arange(5.0))
    assert MarkerTrajectoryPlot(with_weight, max_markers=4).plot().fig is not None

    without_weight = wrap_orbits(np.random.default_rng(2).random((5, 20, 5)), np.arange(5.0))
    assert MarkerTrajectoryPlot(without_weight, max_markers=4).plot().fig is not None


def test_animation_writes_one_frame_per_step(tmp_path):
    plot = AnimationPlot(phase_space(nt=10), step=3)
    assert list(plot.frames) == [0, 3, 6, 9]

    paths = plot.save_frames(tmp_path)
    assert len(paths) == 4
    assert all(p.exists() for p in map(__import__("pathlib").Path, paths))


def test_animation_builds_a_matplotlib_animation():
    anim = AnimationPlot(phase_space(nt=6), step=2).animate()
    assert len(list(anim.new_frame_seq())) == 3


def test_save_writes_a_file(tmp_path):
    out = tmp_path / "fig.png"
    TimeSeriesPlot(
        StruphyArray(np.arange(1.0, 5.0), dims=("t",), coords={"t": np.arange(4.0)}),
        fit=False,
    ).save(out)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------- conservation


def scalars(nt=6):
    """The shape of ``PlottingData.scalars``, with a drifting total energy."""
    t = np.linspace(0.0, 1.0, nt)
    return {
        "en_tot": StruphyArray(2.0 + 0.02 * t, dims=("t",), coords={"t": t}, label="en tot"),
        "en_e": StruphyArray(np.linspace(1.0, 1.5, nt), dims=("t",), coords={"t": t}, label="en e"),
        "en_b": StruphyArray(np.linspace(1.0, 0.5, nt), dims=("t",), coords={"t": t}, label="en b"),
        "time": StruphyArray(t, dims=("t",), coords={"t": t}),
    }


def test_relative_error_is_measured_against_the_first_sample():
    t = np.linspace(0, 1, 5)
    y = StruphyArray(np.array([2.0, 2.0, 2.2, 2.0, 1.8]), dims=("t",), coords={"t": t}, label="E")

    err = relative_error(y)
    # t = 0 is dropped, where the error is identically zero and unplottable on a log axis
    assert err.shape == (4,)
    np.testing.assert_allclose(np.asarray(err), [0.0, 0.1, 0.0, 0.1], atol=1e-12)
    np.testing.assert_allclose(err.coord("t"), t[1:])


def test_relative_error_takes_an_explicit_reference():
    y = StruphyArray(np.array([2.0, 3.0]), dims=("t",), coords={"t": np.arange(2.0)})
    np.testing.assert_allclose(np.asarray(relative_error(y, ref=1.0, skip_first=False)), [1.0, 2.0])


def test_relative_error_refuses_a_zero_reference():
    y = StruphyArray(np.zeros(3), dims=("t",), coords={"t": np.arange(3.0)})
    with pytest.raises(ValueError, match="reference of zero"):
        relative_error(y)


def test_drift_is_the_signed_deviation():
    y = StruphyArray(np.array([2.0, 2.5, 1.0]), dims=("t",), coords={"t": np.arange(3.0)}, label="E")
    d = drift(y)
    np.testing.assert_allclose(np.asarray(d), [0.0, 0.5, -1.0])
    assert d.dims == ("t",)


# ---------------------------------------------------------------- scalars plot


def test_scalars_plot_draws_every_scalar_but_the_excluded():
    plot = ScalarsPlot(scalars()).plot()
    labels = [line.get_label() for line in plot.ax.get_lines()]
    assert labels == ["en_tot", "en_e", "en_b"]


def test_scalars_plot_adds_the_conservation_panel():
    plot = ScalarsPlot(scalars()).plot()

    assert plot.error_ax is not None
    assert plot.error_ax.get_yscale() == "log"
    # en_tot drifts by 1% of its initial value over the run
    assert float(np.asarray(plot.error)[-1]) == pytest.approx(0.01)


def test_scalars_plot_without_a_conserved_quantity_has_no_panel():
    """Not every model tracks ``en_tot``; the overview must still work."""
    without = {k: v for k, v in scalars().items() if k != "en_tot"}
    plot = ScalarsPlot(without).plot()
    assert plot.error_ax is None and plot.error is None
    assert plot.ax.get_xlabel() == "$t$"


def test_scalars_plot_can_normalize_the_mixed_units_away():
    plot = ScalarsPlot(scalars(), relative_to="en_tot").plot()
    assert plot.ax.get_ylabel() == "quantity / en_tot"
    # en_tot against itself is one everywhere
    np.testing.assert_allclose(plot.ax.get_lines()[0].get_ydata(), 1.0)


def test_scalars_plot_labels_the_shared_unit():
    t = np.linspace(0, 1, 4)
    joules = {n: StruphyArray(np.ones(4), dims=("t",), coords={"t": t}, unit="J") for n in ("en_e", "en_b")}
    assert ScalarsPlot(joules, error_panel=None).plot().ax.get_ylabel() == "[J]"


def test_scalars_plot_honours_a_supplied_axes():
    fig, ax = plt.subplots()
    plot = ScalarsPlot(scalars(), ax=ax).plot()
    assert plot.ax is ax
    assert plot.error_ax is None  # a panel cannot be added to someone else's axes


def test_scalars_plot_needs_something_to_plot():
    with pytest.raises(ValueError, match="no scalars to plot"):
        ScalarsPlot({"time": StruphyArray(np.zeros(3), dims=("t",))})


# ---------------------------------------------------------------- saving


def test_save_all_scalars_writes_the_table_and_one_figure_each(tmp_path):
    paths = save_all_scalars(scalars(), str(tmp_path))
    names = sorted(os.path.basename(p) for p in paths)

    assert names == ["en_b.png", "en_e.png", "en_tot.png", "scalars.csv", "scalars.png"]
    assert all(os.path.getsize(p) > 0 for p in paths)


def test_save_all_scalars_closes_its_figures(tmp_path):
    """Saving a run's worth of scalars must not leave every figure open."""
    save_all_scalars(scalars(), str(tmp_path))
    assert plt.get_fignums() == []


def test_save_all_scalars_can_skip_the_table(tmp_path):
    paths = save_all_scalars(scalars(), str(tmp_path), table=None)
    assert not any(p.endswith(".csv") for p in paths)


def test_save_all_scalars_of_nothing_writes_nothing(tmp_path):
    assert save_all_scalars({}, str(tmp_path)) == []


def test_save_closes_the_figure_only_when_asked(tmp_path):
    y = StruphyArray(np.arange(1.0, 5.0), dims=("t",), coords={"t": np.arange(4.0)})

    plot = TimeSeriesPlot(y, fit=False).save(tmp_path / "a.png")
    assert plot.fig is not None and plt.get_fignums() != []
    plt.close("all")

    plot = TimeSeriesPlot(y, fit=False).save(tmp_path / "b.png", close=True)
    assert plot.fig is None and plt.get_fignums() == []


def test_save_does_not_close_an_axes_it_was_given(tmp_path):
    fig, ax = plt.subplots()
    y = StruphyArray(np.arange(1.0, 5.0), dims=("t",), coords={"t": np.arange(4.0)})
    TimeSeriesPlot(y, fit=False, ax=ax).save(tmp_path / "c.png", close=True)
    assert plt.get_fignums() == [fig.number]


def test_slider_plot_writes_frames_at_the_current_cut(tmp_path):
    field = StruphyArray(
        np.arange(3 * 6 * 7 * 5, dtype=float).reshape(3, 6, 7, 5),
        dims=("t", "e1", "e2", "e3"),
        coords={"t": np.linspace(0, 1, 3)},
    )
    plot = SliderPlot(field, slice_dim="e3", slice_index=1)

    paths = plot.save_frames(tmp_path, prefix="phi")
    assert [os.path.basename(p) for p in paths] == ["phi_0000.png", "phi_0001.png", "phi_0002.png"]
    assert plt.get_fignums() == []


def test_slider_frames_follow_the_slider(tmp_path):
    """The cut found interactively is the one that gets written out."""
    field = StruphyArray(np.zeros((2, 4, 4, 5)), dims=("t", "e1", "e2", "e3"))
    plot = SliderPlot(field, slice_dim="e3").plot()
    assert plot.slice_index == 2

    plot.sliders[1].set_val(4)
    assert plot.slice_index == 4


def test_slider_plot_steps_through_time(tmp_path):
    plot = SliderPlot(phase_space(nt=10), step=4)
    assert list(plot.frames) == [0, 4, 8]
    assert len(plot.save_frames(tmp_path)) == 3


def test_scalars_plot_keeps_a_linear_error_axis_when_nothing_drifts():
    """A short run can conserve exactly, which a log axis cannot draw."""
    t = np.linspace(0, 1, 4)
    exact = {
        "en_tot": StruphyArray(np.full(4, 2.0), dims=("t",), coords={"t": t}),
        "en_e": StruphyArray(np.ones(4), dims=("t",), coords={"t": t}),
    }
    plot = ScalarsPlot(exact).plot()
    assert plot.error_ax.get_yscale() == "linear"
