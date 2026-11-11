"""Mode analysis (struphy.post_processing.mode_analysis) of analytic signals with known frequencies and profiles."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from struphy.post_processing import mode_analysis as ma

T_END, DT = 600.0, 2.0
ETA1 = (np.arange(40) + 0.5) / 40


def gauss(r, r0, w=0.1):
    return np.exp(-((r - r0) ** 2) / (2 * w**2))


class AnalyticOutput:
    """Stand-in for an Output: a HollowTorus (r = 0.1 + 0.9 eta1) and a field given by a function of
    (eta1, eta2, eta3, t) on meshgrids."""

    def __init__(self, fun, equil=None):
        from struphy import domains

        self.domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, tor_period=1)
        self.equil = equil
        self._fun = fun
        self._t = np.arange(0.0, T_END + DT / 2, DT)

    def evaluate(self, name, representation=None, eta1=None, eta2=None, eta3=None, t=None, component=None):
        E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        values = np.stack([self._fun(E1, E2, E3, ti) for ti in self._t])
        return xr.DataArray(values, dims=("t", "eta1", "eta2", "eta3"), coords={"t": self._t})


def synthetic(profiles: dict, omegas: dict, travelling=False) -> xr.DataArray:
    """Coefficients c_m(eta1, t) = f_m(r) cos(omega_m t) (standing) or f_m(r) exp(-i omega_m t) (travelling)."""
    t = np.arange(0.0, T_END + DT / 2, DT)
    r = 0.1 + 0.9 * ETA1
    ms = sorted(profiles)
    c = np.zeros((t.size, ETA1.size, len(ms)), dtype=complex)
    for k, m in enumerate(ms):
        phase = np.exp(-1j * omegas[m] * t) if travelling else np.cos(omegas[m] * t)
        c[:, :, k] = phase[:, None] * profiles[m](r)[None, :]
    return xr.DataArray(
        c, dims=("t", "eta1", "m"), coords={"t": t, "eta1": ETA1, "m": ms, "r": ("eta1", r)}, attrs={"n": -1}
    )


@pytest.mark.mpi_skip
@pytest.mark.parametrize("travelling", [False, True])
def test_frequency_and_profile(travelling):
    """The peak frequency is the one of the signal (to a fraction of the resolution 2 pi/T) and the filtered profile is
    its radial profile, for standing and travelling waves alike."""
    omega0 = 0.0621
    profiles = {1: lambda r: gauss(r, 0.5) + 0.5 * (r < 0.5), 2: lambda r: gauss(r, 0.55, 0.08)}
    spectrum = ma.time_spectrum(synthetic(profiles, {1: omega0, 2: omega0}, travelling))
    res = spectrum.attrs["resolution"]
    assert res == pytest.approx(2 * np.pi / T_END)
    peak = ma.peak_frequency(spectrum, m=1, r=0.5)
    assert abs(peak - omega0) < 0.2 * res
    prof = ma.filtered_profile(spectrum, omega=peak)
    r = prof["r"].values
    for m, f in profiles.items():
        got = prof.sel(m=m).values
        ref = f(r)
        assert np.linalg.norm(got / got.max() - ref / ref.max()) < 1e-2 * np.linalg.norm(ref / ref.max())


@pytest.mark.mpi_skip
def test_band_separates_two_oscillations():
    """Two oscillations with different frequencies and profiles: each band (and each peak) recovers its own profile."""
    w1, w2 = 0.06, 0.12
    f1, f2 = (lambda r: gauss(r, 0.3)), (lambda r: gauss(r, 0.7))
    t = np.arange(0.0, T_END + DT / 2, DT)
    r = 0.1 + 0.9 * ETA1
    c = (np.cos(w1 * t)[:, None] * f1(r)[None, :] + np.cos(w2 * t)[:, None] * f2(r)[None, :])[:, :, None]
    coeffs = xr.DataArray(c, dims=("t", "eta1", "m"), coords={"t": t, "eta1": ETA1, "m": [1], "r": ("eta1", r)})
    spectrum = ma.time_spectrum(coeffs)
    for (lo, hi), f in (((0.04, 0.08), f1), ((0.1, 0.14), f2)):
        prof = ma.filtered_profile(spectrum, band=(lo, hi)).sel(m=1).values
        ref = f(r)
        assert np.linalg.norm(prof / prof.max() - ref / ref.max()) < 0.02 * np.linalg.norm(ref / ref.max())
    assert abs(ma.peak_frequency(spectrum, m=1, r=0.7, omega_max=0.2) - w2) < 0.2 * spectrum.attrs["resolution"]
    with pytest.raises(ValueError):
        ma.filtered_profile(spectrum)


@pytest.mark.mpi_skip
def test_mode_coefficients_convention():
    """For f = g(r) cos(2 pi (eta2 - eta3) - omega t), the harmonic (m, n) = (1, -1) is g(r) e^{-i omega t} / 2, the
    other harmonics vanish, and the radius coordinate is the minor radius of the HollowTorus."""
    omega = 0.07

    def fun(e1, e2, e3, t):
        r = 0.1 + 0.9 * e1
        return gauss(r, 0.5) * np.cos(2 * np.pi * (e2 - e3) - omega * t)

    out = AnalyticOutput(fun)
    coeffs = ma.mode_coefficients(out, "mhd/velocity", ms=(0, 1, 2), n=-1, num_eta1=40)
    r = coeffs["r"].values
    assert np.allclose(r, 0.1 + 0.9 * ETA1)
    t = coeffs["t"].values
    expected = 0.5 * gauss(r, 0.5)[None, :] * np.exp(-1j * omega * t)[:, None]
    assert np.allclose(coeffs.sel(m=1).values, expected, atol=1e-12)
    assert np.allclose(coeffs.sel(m=0).values, 0.0, atol=1e-12)
    assert np.allclose(coeffs.sel(m=2).values, 0.0, atol=1e-12)
    # n = +1 holds the complex conjugate part of the m = -1 harmonic only: zero for m = 1
    coeffs_p = ma.mode_coefficients(out, "mhd/velocity", ms=(1,), n=1, num_eta1=40)
    assert np.allclose(coeffs_p.values, 0.0, atol=1e-12)
    spectrum = ma.time_spectrum(coeffs)
    assert abs(ma.peak_frequency(spectrum, m=1, r=0.5) - omega) < 0.2 * spectrum.attrs["resolution"]


@pytest.mark.mpi_skip
def test_cylindrical_continuum_crossing():
    """The m = 1 and m = 2 continua of n = 1 cross at q = 3/2 (r = 0.5 for q = 1.25 + r^2) at the TAE frequency
    n/(2m+1) v_A/R0, and vanish where q = m/n."""
    from struphy import domains, equils

    equil = equils.AdhocTorus(a=1.0, R0=5.0, B0=1.0, q_kind=0, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0)
    out = AnalyticOutput(lambda *a: 0.0, equil=equil)
    equil.domain = out.domain
    w1 = ma.cylindrical_continuum(out, np.array([0.5]), m=1, n=1)
    w2 = ma.cylindrical_continuum(out, np.array([0.5]), m=2, n=1)
    assert w1[0] == pytest.approx(w2[0], rel=1e-10)
    abs_b = np.mean(equil.absB0(np.array([(0.5 - 0.1) / 0.9]), np.linspace(0, 1, 32, endpoint=False), np.array([0.0])))
    assert w1[0] == pytest.approx(abs_b / 3 / 5.0, rel=1e-10)
    # q = 2 at r = sqrt(0.75): the m = 2 continuum vanishes there
    assert ma.cylindrical_continuum(out, np.array([np.sqrt(0.75)]), m=2, n=1)[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(NotImplementedError):
        ma.cylindrical_continuum(SimpleNamespace(equil=SimpleNamespace(params={}), domain=out.domain), [0.5], 1, 1)


@pytest.mark.mpi_skip
def test_plots():
    """The plots run and return their figures."""
    from matplotlib import pyplot as plt

    spectrum = ma.time_spectrum(synthetic({1: lambda r: gauss(r, 0.5), 2: lambda r: gauss(r, 0.6)}, {1: 0.06, 2: 0.06}))
    peak = ma.peak_frequency(spectrum, m=1, r=0.5)
    for plot in (
        ma.plot_spectrum(spectrum, r=0.5, expected=0.067, measured=peak, tolerance=0.1),
        ma.plot_omega_r(spectrum, expected=0.067),
        ma.plot_profiles(ma.filtered_profile(spectrum, omega=peak), mark_r=0.5),
        ma.plot_profiles(ma.filtered_profile(spectrum, band=(0.05, 0.07))),
    ):
        assert plot.fig.axes
        plt.close(plot.fig)


@pytest.mark.mpi_skip
def test_no_peak_without_oscillation():
    """A constant signal has no local maximum in the spectrum (zero after the mean is removed): the peak frequency is
    NaN."""
    t = np.arange(0.0, T_END + DT / 2, DT)
    c = np.ones((t.size, ETA1.size, 1))
    coeffs = xr.DataArray(c, dims=("t", "eta1", "m"), coords={"t": t, "eta1": ETA1, "m": [1], "r": ("eta1", ETA1)})
    assert np.isnan(ma.peak_frequency(ma.time_spectrum(coeffs), m=1, r=0.5))


def test_mode_analysis_of_a_run(mpi_out_folder):
    """On a short Maxwell run on a torus with an AdhocTorus equilibrium and an m = 1, n = 1 electric field (2-form),
    the (1, 1) harmonic dominates, the coefficients have the documented dims and coordinates, and the spectrum and the
    (r, omega) map with the continua of the equilibrium work on the Output."""
    from matplotlib import pyplot as plt

    from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Output, Simulation, Time, domains, equils, grids
    from struphy.initial import perturbations
    from struphy.models import Maxwell

    n_saved = 6
    model = Maxwell(base_units=BaseUnits())
    model.em_fields.e_field.save_data = True
    sim = Simulation(
        model=model,
        params_path=__file__,
        env=EnvironmentOptions(out_folders=mpi_out_folder, sim_folder="sim_mode_analysis"),
        time_opts=Time(dt=0.05, Tend=(n_saved - 1) * 0.05),
        domain=domains.HollowTorus(a1=0.2, a2=1.0, R0=3.0, tor_period=1),
        equil=equils.AdhocTorus(a=1.0, R0=3.0, B0=1.0, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 8, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    model.em_fields.e_field.add_perturbation(
        perturbations.TorusModesSin(ms=(1,), ns=(1,), amps=(1e-3,), comp=0, given_in_basis="2")
    )
    sim.run()
    out = Output(sim.env.path_out)

    coeffs = ma.mode_coefficients(out, "em_fields/e_field", ms=(0, 1, 2), n=1, component=0, num_eta1=12)
    assert coeffs.dims == ("t", "eta1", "m")
    assert coeffs.sizes["t"] == n_saved
    assert np.allclose(coeffs["r"].values, 0.2 + 0.8 * coeffs["eta1"].values)
    amp = np.abs(coeffs.isel(t=0)).max("eta1")
    assert amp.sel(m=1) > 10 * max(amp.sel(m=0), amp.sel(m=2))
    other = ma.mode_coefficients(out, "em_fields/e_field", ms=(1,), n=-1, component=0, num_eta1=12)
    assert np.abs(other.isel(t=0)).max() < 0.1 * amp.sel(m=1)

    spectrum = ma.time_spectrum(coeffs)
    assert spectrum.dims == ("omega", "eta1", "m")
    plot = ma.plot_omega_r(spectrum, out=out)
    # the continua of the equilibrium (m = 0..3) are drawn in each panel
    assert all(len(ax.lines) >= 4 for ax in plot.axes)
    plt.close(plot.fig)
