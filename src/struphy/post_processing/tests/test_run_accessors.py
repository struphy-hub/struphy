"""Tests for run.plot, run.analysis and product lookup by name."""

import os

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from struphy.post_processing.run import Run  # noqa: E402
from struphy.post_processing.tests.test_run import NT, FakeSim, write_manifest, write_tree  # noqa: E402

RATE = 2.0


def make_run(root, name="sim_1"):
    path = os.path.join(root, name)
    os.makedirs(path)
    write_tree(path)
    with h5py.File(os.path.join(path, "data", "data_proc0.hdf5"), "a") as file:
        time = np.asarray(file["time/value"])
        file.create_dataset("scalar/en_phi", data=np.exp(RATE * time))
    write_manifest(path)
    return Run(path, sim=FakeSim(), time_units="normalized")


@pytest.fixture
def run(tmp_path):
    return make_run(str(tmp_path))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_products_are_found_by_name(run):
    assert run["en_tot"].dims == ("t",)
    assert run["em_fields/E"].dims[:2] == ("t", "component")
    assert run["kinetic_ions/e1_v1_density/f_binned"].dims == ("t", "e1", "v1")
    assert run["kinetic_ions/view_0/n_sph"].dims == ("t", "e1", "e2", "e3")
    assert run["kinetic_ions"].dims == ("t", "marker", "attribute")
    with pytest.raises(KeyError, match="available products"):
        run["t"]


def test_every_array_carries_its_run(run):
    for array in (run.scalars.en_tot, run.fields.em_fields.E, run["kinetic_ions"]):
        assert array.attrs["run"] == run.label
        assert array.attrs["run_name"] == "sim_1"
    assert run.scalars.en_tot.isel(t=slice(1, None)).attrs["run_name"] == "sim_1"


def test_timeseries_by_name_with_growth_fit(run):
    result = run.plot.timeseries("en_phi", fit=True)
    assert result.fit_results[0].rate == pytest.approx(RATE)
    assert result.fig._suptitle.get_text() == run.label


def test_timeseries_of_several_runs_are_labeled_by_run(tmp_path):
    first, second = make_run(str(tmp_path), "sim_1"), make_run(str(tmp_path), "sim_2")
    result = first.plot.timeseries(first.scalars.en_phi, second.scalars.en_phi)
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert labels == ["en phi (sim_1)", "en phi (sim_2)"]


def test_timeseries_into_given_axes_keeps_the_figure_layout(run):
    fig, ax = plt.subplots()
    fig.suptitle("mine")
    run.plot.timeseries("en_tot", ax=ax, logy=False)
    assert fig._suptitle.get_text() == "mine"


def test_scalar_overview_picks_the_total_energy(run):
    result = run.plot.scalars()
    assert result.data["relative_error"].sizes["t"] == NT - 1
    assert run.plot.scalars(conservation=None).data["relative_error"] is None


def test_slices_panels_and_viewer_take_keyword_views(run):
    name = "kinetic_ions/e1_v1_density/f_binned"
    assert run.plot.slice(name, x="e1", y="v1", isel={"t": -1}).ax.get_xlabel() == r"$\eta_1$"
    assert len(run.plot.panels(name, x="e1", y="v1", nrows=1, ncols=2).artists) == 2
    viewer = run.plot.viewer("em_fields/E", x="e1", y="e2", isel={"component": 0})
    viewer.draw()
    assert set(viewer.sliders) == {"t", "e3"}


def test_orbits_default_to_the_only_species(run):
    assert run.plot.orbits().ax.name == "3d"


def test_report_is_written_below_post_processing(run):
    paths = run.save_report()
    assert all(path.startswith(str(run.path_pproc / "report")) for path in paths)
    assert {os.path.basename(path) for path in paths} >= {"scalars.csv", "scalars.png", "en_phi.png"}


def test_analysis_by_name(run):
    assert run.analysis.growth_rate("en_phi", window=(0.0, None)).rate == pytest.approx(RATE)
    assert run.analysis.growth_rate("en_phi", amplitude=True).rate == pytest.approx(RATE / 2)
    np.testing.assert_allclose(run.analysis.relative_error("en_tot"), 0.0)
    np.testing.assert_allclose(run.analysis.drift("en_phi").isel(t=0), 0.0)


def test_dispersion_rejects_fields_in_seconds(run):
    physical = run.with_time_units("physical")
    physical._sim.model = type("Model", (), {"units": type("Units", (), {"t": 2.0})()})()
    with pytest.raises(ValueError, match="normalized"):
        physical.analysis.dispersion(physical.fields.em_fields.E)
