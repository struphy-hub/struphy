"""Tests for functional plotting and the shared view recipe."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from struphy.diagnostics.plotting import (  # noqa: E402
    GrowthFit,
    InteractiveSliceViewer,
    View,
    animate_slices,
    drift,
    growth_rate,
    logical_grids,
    physical_grids,
    plot_panels,
    plot_scalars,
    plot_slice,
    plot_timeseries,
    relative_error,
    save_all_scalars,
    save_frames,
)
from struphy.post_processing.arrays import data_array  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def phase_space(nt=6):
    return data_array(
        np.arange(nt * 4 * 5).reshape(nt, 4, 5),
        ("t", "e1", "v1"),
        {"t": np.linspace(0, 1, nt), "e1": np.linspace(0, 1, 4), "v1": np.linspace(-2, 2, 5)},
        name="f",
        label="$f$",
        coord_units={"t": "s"},
    )


def physical_field():
    coords = {"t": [0, 1], "e1": range(3), "e2": range(4), "e3": range(5)}
    grids = np.meshgrid(coords["e1"], coords["e2"], coords["e3"], indexing="ij")
    coords.update({name: (("e1", "e2", "e3"), grid) for name, grid in zip(("X", "Y", "Z"), grids)})
    return data_array(np.ones((2, 3, 4, 5)), ("t", "e1", "e2", "e3"), coords, name="phi")


def scalar_dataset():
    t = np.linspace(0, 1, 6)
    return xr.Dataset({"en_tot": ("t", 2 + 0.02 * t), "en_e": ("t", 1 + 0.1 * t)}, coords={"t": t})


def test_growth_rate_uses_only_valid_samples_inside_window():
    data = data_array([1, 0, 4, np.nan, 16], ("t",), {"t": range(5)})
    result = growth_rate(data, GrowthFit((0, 4)))
    assert result is not None and np.isfinite(result.rate)
    np.testing.assert_array_equal(result.time, [0, 2, 4])


def test_growth_rate_does_not_fall_back_outside_requested_window():
    data = data_array(np.exp(np.arange(5)), ("t",), {"t": range(5)})
    assert growth_rate(data, GrowthFit((1.1, 1.2))) is None


def test_growth_rate_of_quadratic_reports_amplitude_rate():
    t = np.linspace(0, 4, 20)
    result = growth_rate(data_array(np.exp(0.6 * t), ("t",), {"t": t}), GrowthFit(amplitude_from_quadratic=True))
    assert result.rate == pytest.approx(0.3)


def test_diagnostics_preserve_time_coordinates():
    data = data_array([2, 2.2, 1.8], ("t",), {"t": [0, 1, 2]}, label="E")
    np.testing.assert_allclose(drift(data), [0, 0.2, -0.2])
    np.testing.assert_allclose(relative_error(data), [0.1, 0.1])
    np.testing.assert_array_equal(relative_error(data).t, [1, 2])


def test_logical_and_physical_grids_follow_selected_dimensions():
    logical = phase_space().isel(t=0)
    assert logical_grids(logical)[0].shape == (4, 5)
    physical = physical_field().isel(t=0, e3=2)
    assert physical_grids(physical, plane="XY")[0].shape == (3, 4)
    assert physical_grids(physical, plane="RZ")[0].shape == (3, 4)


def test_plot_timeseries_renders_once_and_save_does_not_redraw(tmp_path):
    data = data_array(np.exp(np.arange(4)), ("t",), {"t": range(4)}, label="energy", coord_units={"t": "s"})
    result = plot_timeseries(data, fit=GrowthFit(), run_label="dt=.1")
    lines = len(result.ax.lines)
    result.save(tmp_path / "energy.png")
    assert len(result.ax.lines) == lines
    assert len(plt.get_fignums()) == 1
    assert result.fig._suptitle.get_text() == "dt=.1"


def test_plot_slice_accepts_named_value_and_index_selection():
    result = plot_slice(phase_space(), view=View(x="e1", y="v1", select={"t": 0.52}))
    assert result.ax.get_xlabel() == r"$\eta_1$"
    assert len(result.artists) == 1


def test_plot_slice_physical_coordinates_are_intrinsic():
    result = plot_slice(
        physical_field(), view=View(x="e1", y="e2", isel={"t": 0, "e3": 2}, coordinates="physical", plane="XY")
    )
    assert result.ax.get_xlabel() == "X"
    assert result.ax.get_aspect() == 1.0


def test_plot_slice_rejects_underspecified_selection():
    with pytest.raises(ValueError, match="selection leaves"):
        plot_slice(physical_field(), view=View(x="e1", y="e2", isel={"t": 0}))


def test_panels_use_one_recipe_and_keep_full_title():
    result = plot_panels(
        phase_space(), view=View(x="e1", y="v1"), nrows=1, ncols=2, title="Distribution", run_label="dt=.1"
    )
    assert result.fig._suptitle.get_text() == "Distribution — dt=.1"
    assert len(result.artists) == 2


def test_viewer_builds_controls_for_every_non_display_dimension():
    viewer = InteractiveSliceViewer(physical_field(), view=View(x="e1", y="e2", coordinates="physical"))
    result = viewer.draw()
    assert set(viewer.sliders) == {"t", "e3"}
    viewer.sliders["e3"].set_val(3)
    assert result.fig is not None


def test_animation_and_frames_share_the_view(tmp_path):
    data = phase_space(nt=7)
    view = View(x="e1", y="v1")
    animation = animate_slices(data, view=view, step=3)
    assert len(list(animation.new_frame_seq())) == 3
    paths = save_frames(data, tmp_path, view=view, step=3)
    assert len(paths) == 3
    assert all(__import__("pathlib").Path(path).exists() for path in paths)


def test_scalar_overview_and_export(tmp_path):
    result = plot_scalars(scalar_dataset(), run_label="run")
    assert sorted(line.get_label() for line in result.artists) == ["en_e", "en_tot"]
    assert result.fig._suptitle.get_text() == "run"
    paths = save_all_scalars(scalar_dataset(), tmp_path)
    assert sorted(__import__("os").path.basename(path) for path in paths) == [
        "en_e.png",
        "en_tot.png",
        "scalars.csv",
        "scalars.png",
    ]
    assert plt.get_fignums() == [result.fig.number]


@pytest.mark.parametrize("shown", [False, True])
def test_notebook_display_shows_the_figure_once(monkeypatch, shown):
    import IPython.display

    displayed = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline")
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    result = plot_timeseries(scalar_dataset().en_tot, logy=False)
    if shown:
        result.show()
    result._ipython_display_()
    assert displayed == ([] if shown else [result.fig])
    assert (result.fig.number in plt.get_fignums()) == shown, "the inline backend must not show it again"
