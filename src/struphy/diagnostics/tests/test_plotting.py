"""Unit tests for the standardized plotters.

These render into the Agg backend, so they check the geometry and labeling that the
plotters derive from the data rather than the appearance of the result.
"""

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
    Slice2DPlot,
    SliderPlot,
    TimeSeriesPlot,
    field_slice_grids,
    growth_rate,
    logical_grids,
    match_to_grid,
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
    series = [
        StruphyArray(np.exp(rate * t), dims=("t",), coords={"t": t}, label=f"run {rate}")
        for rate in (0.2, 0.4)
    ]
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
