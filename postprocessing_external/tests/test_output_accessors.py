"""Tests for run.plot, run.analysis and product lookup by name."""

import os

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

import struphy_plots  # noqa: F401, E402
from struphy_plots.analysis import damping_rate, envelope, growth_rate, norm
from struphy_plots.output_accessors import OutputPlots
from struphy_plots.plotting import save_all_scalars
from struphy.post_processing.output import Output  # noqa: E402
from struphy.post_processing.tests.test_output import write_manifest, write_tree  # noqa: E402

RATE = 2.0


def make_run(root, name="sim_1"):
    path = os.path.join(root, name)
    os.makedirs(path)
    write_tree(path)
    with h5py.File(os.path.join(path, "data", "data_proc0.hdf5"), "a") as file:
        time = np.asarray(file["time/value"])
        file.create_dataset("scalar/en_phi", data=np.exp(RATE * time))
    write_manifest(path)
    return Output(path)


@pytest.fixture
def run(tmp_path):
    return make_run(str(tmp_path))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def scalar(run, name):
    return run.evaluate("scalars", variables=name)[name]


def distribution(run):
    return run.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")


def density(run):
    return run.evaluate("kinetic_ions/n", dataset="view_0/n")


def orbits(run):
    return run.evaluate("kinetic_ions/orbits")


def test_products_are_found_by_name(run):
    assert scalar(run, "en_tot").dims == ("t",)
    assert run.evaluate("em_fields/E").dims[:2] == ("t", "component")
    assert distribution(run).dims == ("t", "e1", "v1")
    assert density(run).dims == ("t", "e1", "e2", "e3")
    assert orbits(run).dims == ("t", "marker", "quantity")
    with pytest.raises(ValueError, match="species/variable"):
        run.evaluate("t")


def test_every_array_carries_its_run(run):
    for array in (run.scalars.en_tot, run.fields.em_fields.E, orbits(run)):
        assert array.attrs["run"] == run.label
        assert array.attrs["run_name"] == "sim_1"
    assert run.scalars.en_tot.isel(t=slice(1, None)).attrs["run_name"] == "sim_1"


def test_timeseries_by_name_with_growth_fit(run):
    result = scalar(run, "en_phi").struphy.plot.timeseries(fit=True)
    assert result.fit_results[0].rate == pytest.approx(RATE)
    assert result.fig._suptitle.get_text() == run.label


def test_output_keeps_only_the_core_quick_plot(run):
    assert not hasattr(run, "timeseries")
    assert callable(run.plot)


def test_timeseries_of_several_runs_are_labeled_by_run(tmp_path):
    first, second = make_run(str(tmp_path), "sim_1"), make_run(str(tmp_path), "sim_2")
    result = first.scalars.en_phi.struphy.plot.timeseries(second.scalars.en_phi)
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert labels == ["en phi (sim_1)", "en phi (sim_2)"]


def test_timeseries_into_given_axes_keeps_the_figure_layout(run):
    fig, ax = plt.subplots()
    fig.suptitle("mine")
    scalar(run, "en_tot").struphy.plot.timeseries(ax=ax, logy=False)
    assert fig._suptitle.get_text() == "mine"


def test_scalar_overview_draws_every_scalar_in_one_axes(run):
    result = OutputPlots(run).scalars()
    fig, ax = result.fig, result.ax
    assert sorted(line.get_label() for line in ax.lines) == ["en_phi", "en_tot"]
    assert fig.axes == [ax]


def test_slices_panels_and_viewer_take_keyword_views(run):
    product = distribution(run)
    assert product.struphy.plot.slice(x="e1", y="v1", t="last").ax.get_xlabel() == r"$\eta_1$"
    assert len(product.struphy.plot.panels(x="e1", y="v1", nrows=1, ncols=2).artists) == 2
    viewer = run.evaluate("em_fields/E").struphy.plot.viewer(x="e1", y="e2", component=0)
    viewer.draw()
    assert set(viewer.sliders) == {"t", "e3"}


def test_orbits_plot_their_trajectories(run):
    assert run.kinetic_ions.orbits.struphy.plot.trajectories().ax.name == "3d"


def test_report_is_written_below_post_processing(run):
    paths = save_all_scalars(run.scalars, run.path_pproc / "report", run_label=run.label)
    assert all(path.startswith(str(run.path_pproc / "report")) for path in paths)
    assert {os.path.basename(path) for path in paths} >= {"scalars.csv", "scalars.png", "en_phi.png"}


def test_analysis_by_name(run):
    assert scalar(run, "en_phi").struphy.analysis.growth_rate(window=(0.0, None)).rate == pytest.approx(RATE)
    assert scalar(run, "en_phi").struphy.analysis.growth_rate(amplitude=True).rate == pytest.approx(RATE / 2)
    np.testing.assert_allclose(scalar(run, "en_tot").struphy.analysis.relative_error(), 0.0)
    np.testing.assert_allclose(scalar(run, "en_phi").struphy.analysis.drift().isel(t=0), 0.0)


def test_dispersion_rejects_fields_in_seconds(run):
    physical = run.with_time_units("physical")
    with pytest.raises(ValueError, match="normalized"):
        physical.fields.em_fields.E.struphy.analysis.dispersion()


def test_selection_keywords_take_positions_values_and_ends(run):
    product = distribution(run)
    times = product.t.values

    by_position = product.struphy.plot.slice(x="e1", y="v1", t=-1)
    by_value = product.struphy.plot.slice(x="e1", y="v1", t=float(times[-1]))
    by_end = product.struphy.plot.slice(x="e1", y="v1", t="last")
    for result in (by_value, by_end):
        np.testing.assert_allclose(result.artists[0].get_array(), by_position.artists[0].get_array())

    with pytest.raises(TypeError, match="not a dimension"):
        product.struphy.plot.slice(x="e1", y="v1", time=-1)
    with pytest.raises(TypeError, match="use a number"):
        product.struphy.plot.slice(x="e1", y="v1", t="final")


def test_products_of_one_species_sit_on_the_output(run):
    assert run.kinetic_ions.e1_v1_density.f.dims == ("t", "e1", "v1")
    assert run.kinetic_ions.view_0.n.dims == ("t", "e1", "e2", "e3")
    assert run.kinetic_ions.orbits.dims == ("t", "marker", "quantity")
    assert run.em_fields.E.dims[:2] == ("t", "component")
    assert {"kinetic_ions", "em_fields"} <= set(dir(run))
    with pytest.raises(AttributeError, match="available species"):
        run.electrons


def test_product_namespaces_expose_a_scoped_lazy_catalog(run):
    products = run.kinetic_ions
    assert tuple(sorted(products.catalog)) == ("e1_v1_density/delta_f", "e1_v1_density/f", "orbits", "view_0/n")
    assert "e1_v1_density/f" in products.catalog
    assert "em_fields/E" not in products.catalog
    assert "e1_v1_density/f" in repr(products)
    assert run.distribution_catalog._cache == {}
    assert products["e1_v1_density/f"].dims == ("t", "e1", "v1")
    assert run.distribution_catalog._cache["kinetic_ions/e1_v1_density/f"] is products.catalog["e1_v1_density/f"]


def test_arrays_plot_themselves(run):
    phase_space = run.kinetic_ions.e1_v1_density.f
    assert phase_space.struphy.plot.slice(x="e1", y="v1", t="last").ax.get_xlabel() == r"$\eta_1$"
    assert len(phase_space.struphy.plot.panels(x="e1", y="v1", nrows=1, ncols=2).artists) == 2
    assert set(phase_space.struphy.plot.viewer(x="e1", y="v1").sliders) == set()
    assert run.kinetic_ions.orbits.struphy.plot.trajectories(max_markers=2).ax.name == "3d"


def test_the_accessor_works_on_derived_arrays(run):
    energy = run.scalars.en_phi
    assert energy.isel(t=slice(1, None)).struphy.analysis.growth_rate().rate == pytest.approx(RATE)
    error = energy.struphy.analysis.relative_error()
    assert error.struphy.plot.timeseries(logy=False).fig._suptitle.get_text() == run.label


def test_products_by_name_and_by_attribute_agree(run):
    by_output = distribution(run).struphy.plot.slice(x="e1", y="v1", t="last")
    by_attribute = run.kinetic_ions.e1_v1_density.f.struphy.plot.slice(x="e1", y="v1", t="last")
    np.testing.assert_allclose(by_output.artists[0].get_array(), by_attribute.artists[0].get_array())
    assert by_output.fig._suptitle.get_text() == by_attribute.fig._suptitle.get_text() == run.label


def test_selection_rejects_unknown_dimensions(run):
    with pytest.raises(TypeError, match="not a dimension"):
        run.kinetic_ions.e1_v1_density.f.struphy.plot.slice(x="e1", y="v1", time=-1)


def oscillating_energy(rate=-0.3, omega=3.0):
    import xarray as xr

    time = np.linspace(0.0, 20.0, 4001)
    values = np.exp(2 * rate * time) * np.cos(omega * time) ** 2 + 1e-12
    return xr.DataArray(values, dims="t", coords={"t": time}, name="energy")


def test_damping_rate_fits_the_envelope_not_the_oscillation(run):
    energy = oscillating_energy(rate=-0.3)
    fit = energy.struphy.analysis.damping_rate(amplitude=True)
    assert fit.rate == pytest.approx(-0.3, rel=1e-2)
    assert energy.struphy.analysis.damping_rate(window=(2.0, 10.0), amplitude=True).rate == pytest.approx(
        -0.3, rel=1e-2
    )

    peaks = energy.struphy.analysis.envelope()
    assert 0 < peaks.sizes["t"] < energy.sizes["t"] // 10
    assert np.all(peaks > 1e-3 * np.exp(-0.6 * peaks.t))


def test_damping_rate_without_peaks_is_none(run):
    assert scalar(run, "en_phi").struphy.analysis.damping_rate() is None


def test_norm_reduces_all_but_time(run):
    e_field = run.evaluate("em_fields/E")
    squared = e_field.struphy.analysis.norm(squared=True)
    assert squared.dims == ("t",)
    np.testing.assert_allclose(squared, (np.asarray(e_field) ** 2).sum(axis=(1, 2, 3, 4)))
    assert e_field.struphy.analysis.norm(dims=["e1"]).dims == ("t", "component", "e2", "e3")
    assert growth_rate(squared, fit=None) is not None


def test_physical_coords_are_attached_to_products_without_them(run):
    density_data = density(run)
    assert "X" not in density_data.coords
    mapped = run.with_physical_coords(density_data)
    expected = run.domain(*(np.asarray(density_data[dim]) for dim in ("e1", "e2", "e3")))
    for name, values in zip(("X", "Y", "Z"), expected):
        assert mapped[name].dims == ("e1", "e2", "e3")
        np.testing.assert_allclose(mapped[name], values)

    plane = run.with_physical_coords(density_data.isel(e3=0, drop=True))
    assert plane.X.dims == ("e1", "e2")

    phase_space = run.with_physical_coords(distribution(run))
    assert phase_space.X.dims == ("e1",)

    field = run.evaluate("em_fields/E")
    assert run.with_physical_coords(field) is field
    with pytest.raises(ValueError, match="no logical dimensions"):
        run.with_physical_coords(scalar(run, "en_tot"))
