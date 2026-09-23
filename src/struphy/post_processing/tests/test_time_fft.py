"""Fourier normalization, labeled coordinates and the migrated TAE filtering workflow."""

import numpy as np
import pytest
import xarray as xr

from struphy.post_processing.time_fft import fft, filter_time, fwhm_window, inverse_time_fft, time_fft


def signal(n=400, bins=5.0, dt=0.5):
    t = np.arange(n) * dt
    x = np.linspace(0, 1, 16)
    profile = np.exp(-(((x - 0.5) / 0.15) ** 2))
    main = np.sin(2 * np.pi * bins * np.arange(n) / n)[:, None] * profile
    other = 0.4 * np.cos(2 * np.pi * 12 * np.arange(n) / n)[:, None] * profile
    data = xr.DataArray(main + other + 0.7, dims=("t", "e1"), coords={"t": t, "e1": x}, name="velocity")
    data.attrs.update(units="m/s", run="synthetic")
    return data, main


@pytest.mark.parametrize("n", [399, 400])
def test_normalization_and_inverse_for_odd_even_records(n):
    data, _ = signal(n=n)
    data = data.assign_coords(X=("e1", data.e1.values * 2), moving=("t", data.t.values + 1))
    spectrum = time_fft(data)
    assert spectrum.coefficients.dims == ("omega", "e1")
    assert "X" in spectrum.coords and "moving" not in spectrum.coords
    assert spectrum.attrs["run"] == "synthetic"
    assert spectrum.power.attrs["units"] == "(m/s)^2"
    np.testing.assert_allclose(spectrum.power.sum("omega"), (data**2).mean("t"), rtol=1e-12)
    xr.testing.assert_allclose(inverse_time_fft(spectrum.coefficients, data), data)


def test_dc_and_nyquist_are_not_doubled():
    t = np.arange(64)
    data = xr.DataArray(3 + 2 * (-1.0) ** t, dims="t", coords={"t": t})
    spectrum = time_fft(data)
    assert float(spectrum.power.isel(omega=0)) == pytest.approx(9)
    assert float(spectrum.power.isel(omega=-1)) == pytest.approx(4)
    assert float(spectrum.power.sum()) == pytest.approx(13)


def test_filter_recovers_on_bin_mode_and_preserves_input():
    data, main = signal()
    original = data.copy(deep=True)
    result = filter_time(data)
    np.testing.assert_allclose(result.filtered, main, atol=1e-14)
    xr.testing.assert_identical(data, original)
    spectrum = result.spectrum
    assert int(spectrum.idx_dominant) == 5
    assert int(spectrum.idx_lo) == int(spectrum.idx_hi) == 5
    assert float(spectrum.dominant_frequency) == pytest.approx(5 * 2 * np.pi / 200)
    assert result.filtered.attrs["units"] == "m/s"


@pytest.mark.parametrize("bins", [5.3, 5.5])
def test_off_bin_padding_migrated_from_tae_branch(bins):
    data, main = signal(bins=bins)
    errors = []
    for padding in range(6):
        result = filter_time(data, pad_bins=padding)
        spectrum = result.spectrum
        if padding == 0:
            lo, hi = int(spectrum.idx_lo), int(spectrum.idx_hi)
            assert 5 <= lo <= hi <= 6
        assert int(spectrum.idx_lo) == max(lo - padding, 1)
        assert int(spectrum.idx_hi) == hi + padding < 12
        errors.append(np.linalg.norm(result.filtered - main) / np.linalg.norm(main))
    assert all(right < left for left, right in zip(errors, errors[1:]))
    assert errors[-1] < 0.2


def test_component_bands_and_nonleading_time_axis():
    first, main = signal()
    second, second_main = signal(bins=8)
    data = xr.concat([first, second, xr.zeros_like(first), xr.ones_like(first)], dim="component")
    data = data.assign_coords(component=["radial", "poloidal", "zero", "constant"]).transpose("e1", "component", "t")
    result = filter_time(data)
    assert result.filtered.dims == data.dims
    assert result.spectrum.power.dims == ("omega", "component")
    np.testing.assert_array_equal(result.spectrum.idx_dominant, [5, 8, -1, -1])
    np.testing.assert_array_equal(result.spectrum.has_peak, [True, True, False, False])
    np.testing.assert_allclose(result.filtered.sel(component="radial").T, main, atol=1e-14)
    np.testing.assert_allclose(result.filtered.sel(component="poloidal").T, second_main, atol=1e-14)
    assert not result.filtered.sel(component=["zero", "constant"]).values.any()
    assert np.isnan(result.spectrum.dominant_frequency.sel(component="zero"))
    # A shared band can also be selected explicitly across components.
    assert filter_time(data, dims=("e1", "component")).spectrum.power.dims == ("omega",)
    assert filter_time(data, dims=()).spectrum.power.dims == ("omega", "e1", "component")


def test_fwhm_window_and_cutoff():
    power = np.array([0, 1, 3, 8, 10, 6, 4, 1, 0])
    assert fwhm_window(power, 4) == (3, 5)
    assert fwhm_window(power, 4, pad_bins=1) == (2, 6)
    assert fwhm_window(power, 4, idx_min=1, pad_bins=10) == (1, 8)
    assert fwhm_window([9, 10, 2], 1, idx_min=1) == (1, 1)
    data, _ = signal()
    spectrum = filter_time(data, omega_min=0.1, pad_bins=999).spectrum
    assert float(spectrum.omega_lo) >= 0.1
    assert int(spectrum.idx_hi) == 200


def test_hann_window_detrending_and_complex_spatial_fft():
    data, _ = signal()
    transformed = time_fft(data, detrend=True, window="hann")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    expected = (data - data.mean("t")) * xr.DataArray(window, dims="t", coords={"t": data.t})
    np.testing.assert_allclose(transformed.power.sum("omega"), (expected**2).mean("t"), atol=1e-14)
    xr.testing.assert_allclose(inverse_time_fft(transformed.coefficients, data), expected)
    theta = np.arange(64) / 64
    wave = xr.DataArray(np.exp(2j * np.pi * 11 * theta), dims="e2", coords={"e2": theta})
    modes = fft(wave, dim="e2")
    peak = int(abs(modes).argmax(dim="k_e2"))
    assert float(modes.k_e2[peak]) == pytest.approx(2 * np.pi * 11)
    assert abs(complex(modes[peak])) == pytest.approx(1)
    assert float((abs(modes) ** 2).sum()) == pytest.approx(1)


def test_seconds_units_and_saved_sample_spacing():
    data, _ = signal(dt=2e-9)
    data.t.attrs["units"] = "s"
    spectrum = time_fft(data.isel(t=slice(None, None, 2)))
    assert spectrum.omega.attrs["units"] == "rad / s"
    assert spectrum.attrs["sample_spacing"] == pytest.approx(4e-9)
    assert spectrum.attrs["frequency_resolution"] == pytest.approx(2 * np.pi / (400 * 2e-9))


@pytest.mark.parametrize("times", [[0], [0, 1, 1], [2, 1, 0], [0, 1, 3], [0, np.nan], [0, 1e-9, 3e-9]])
def test_invalid_time_grids(times):
    with pytest.raises(ValueError):
        time_fft(xr.DataArray(np.ones(len(times)), dims="t", coords={"t": times}))


@pytest.mark.parametrize(
    "options",
    [
        {"pad_bins": -1},
        {"pad_bins": 1.5},
        {"omega_min": 0},
        {"omega_min": np.nan},
        {"omega_min": 100},
        {"dims": "t"},
        {"dims": ["e1", "e1"]},
        {"dims": "missing"},
    ],
)
def test_invalid_filter_options(options):
    with pytest.raises(ValueError):
        filter_time(signal()[0], **options)


def test_invalid_values_and_inverse_template():
    data, _ = signal()
    with pytest.raises(ValueError, match="real"):
        time_fft(data.astype(complex))
    with pytest.raises(ValueError, match="finite"):
        time_fft(data * np.nan)
    with pytest.raises(ValueError, match="window"):
        time_fft(data, window="wrong")
    with pytest.raises(ValueError, match="coordinate"):
        time_fft(data.drop_vars("t"))
    with pytest.raises(ValueError, match="frequency grid"):
        inverse_time_fft(time_fft(data).coefficients, data.isel(t=slice(None, -2)))


def test_output_and_array_entry_points(tmp_path):
    from struphy.post_processing.tests.test_output_accessors import make_run

    out = make_run(str(tmp_path))
    data, _ = signal()
    xr.testing.assert_identical(out.time_fft(data), data.struphy.analysis.time_fft())
    xr.testing.assert_identical(out.fft(data, dim="e1"), data.struphy.analysis.fft(dim="e1"))
    xr.testing.assert_identical(out.filter_time(data).filtered, data.struphy.analysis.filter_time().filtered)
    xr.testing.assert_identical(out.time_fft("en_tot"), time_fft(out.evaluate("en_tot")))


@pytest.mark.single
def test_linear_mhd_two_alfven_modes(tmp_path):
    """Port of the old branch's simulation check through Output and xarray (no pickle products)."""
    from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
    from struphy.models import LinearMHD

    model = LinearMHD()
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ns=(1, 3), amps=(1e-3, 3e-4), Lz=20, comp=0, given_in_basis="physical")
    )
    simulation = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="two_modes", save_step=2),
        time_opts=Time(dt=0.1, Tend=100),
        domain=domains.Cuboid(r3=20),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 32)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
        equil=equils.HomogenSlab(B0x=0, B0y=0, B0z=1, beta=1, n0=1),
    )
    output = simulation.run().pproc(physical=True)
    for product in ("mhd/velocity", "mhd/velocity_xyz"):
        velocity = output.evaluate(product).isel(component=0, e1=0, e2=0)
        result = output.filter_time(velocity, pad_bins=2)
        spectrum = result.spectrum
        assert abs(float(spectrum.dominant_frequency) - 2 * np.pi / 20) < spectrum.attrs["frequency_resolution"]
        assert float(spectrum.omega_hi) < 3 * 2 * np.pi / 20

        def mode_amplitude(field, n):
            return abs((field * np.sin(2 * np.pi * n * field.e3)).sum("e3")).max().item()

        raw_ratio = mode_amplitude(velocity, 3) / mode_amplitude(velocity, 1)
        filtered_ratio = mode_amplitude(result.filtered, 3) / mode_amplitude(result.filtered, 1)
        assert raw_ratio > 0.2
        assert filtered_ratio < 0.05 * raw_ratio
        assert output.time_fft(velocity).attrs["sample_spacing"] == pytest.approx(0.2)
