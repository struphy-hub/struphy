r"""Mode analysis of FEEC fields: frequencies, (r, omega) maps and mode structures.

The functions of this module analyse a scalar component :math:`f` of a saved FEEC field (e.g. the radial velocity of a
shear Alfven wave) in a torus or cylinder with logical coordinates :math:`(\eta_1, \eta_2, \eta_3)` = (radius,
poloidal angle, toroidal angle or axial coordinate):

1. **Fourier coefficients in the angles** (:func:`mode_coefficients`): the field is evaluated at all saved times on a
   fixed logical grid (``num_eta1`` cell-centred points in :math:`\eta_1`, which avoids the axis and the walls, and
   equidistant points in :math:`\eta_2, \eta_3`; the grid is independent of the resolution of the run, so runs can be
   compared point by point), and transformed with a 2D FFT in :math:`(\eta_2, \eta_3)`:

   .. math::

       f(\eta_1, \eta_2, \eta_3, t) = \sum_{m, n} c_{m,n}(\eta_1, t) \, e^{2 \pi i (m \eta_2 + n \eta_3)}\,.

   The coefficients of the poloidal mode numbers ``ms`` and of one (logical) toroidal mode number ``n`` are kept; the
   physical mode numbers are ``m`` times the ``pol_period`` and ``n`` times the ``tor_period`` of the domain (if it has
   them).

2. **Spectrum in time** (:func:`time_spectrum`): at every radius and for every :math:`m`, the time mean is subtracted,
   a Hann window :math:`w(t)` is applied (it reduces the leakage of a frequency into its neighbours) and the FFT in
   time is taken, zero-padded to ``pad`` times the length (this interpolates the spectrum on a finer :math:`\omega`
   grid but does not improve the frequency resolution :math:`2 \pi / T` of a run of length :math:`T`). The amplitudes
   at :math:`+\omega` and :math:`-\omega` are combined,

   .. math::

       A_m(\eta_1, \omega) = \sqrt{|\hat c_m(\eta_1, \omega)|^2 + |\hat c_m(\eta_1, -\omega)|^2}\,,
       \qquad \omega \geq 0\,,

   so that a standing wave (peaks at :math:`\pm\omega`) and a wave travelling in either direction (one peak) give
   spectra of the same shape (a standing wave :math:`\cos(\omega t)` peaks lower by a factor :math:`\sqrt 2` than a
   travelling wave :math:`e^{-i \omega t}` of the same amplitude). Frequencies are in the time units of the output
   (normalized by default; with :meth:`~struphy.Output.with_time_units` in rad/s).

3. **Frequencies** (:func:`peak_frequency`): the largest local maximum of :math:`A_m(\eta_1, \omega)` for one
   :math:`m` at one radius (the frequency of a global eigenmode is the same at all radii).

4. **Mode structure** (:func:`filtered_profile`): the amplitude :math:`A_m(\eta_1, \omega_0)` at one frequency,
   as a function of the radius for every :math:`m`, or the root of the power summed over a frequency band. For an
   eigenmode with frequency :math:`\omega_0` this is the radial profile of the modulus of its :math:`(m, n)` harmonics,
   filtered from the other oscillations of the run (e.g. the continuum oscillations, whose radial structure gets finer
   in time by phase mixing and is not converged on any fixed grid).

5. **Shear Alfven continua** (:func:`cylindrical_continuum`) for the :math:`(r, \omega)` maps:
   :math:`\omega_m(r) = |n q(r) - m| / (q(r) R_0) \, v_A(r)` with the flux-surface averages of :math:`|B_0|` and of the
   number density in :math:`v_A = |B_0| / \sqrt{n_0}` (in units of the Alfven velocity), for equilibria with a safety
   factor profile ``q_r`` (e.g. :class:`~struphy.fields_background.equils.AdhocTorus`,
   :class:`~struphy.fields_background.equils.ScrewPinch`). Toroidal coupling (gaps) is not included.

Plots: :func:`plot_spectrum` (spectra at one radius with the expected frequency), :func:`plot_omega_r`
(:math:`\log_{10} A_m` over :math:`(r, \omega)` with the continua) and :func:`plot_profiles` (mode structure).

Limitations:

- The profiles are moduli: the phase of the harmonics (e.g. a sign change across the radius) is not shown.
- The frequency resolution is finite: the Hann window has a main lobe of half-width :math:`2 \cdot 2\pi / T`
  (:math:`T` from the first to the last saved time), so the amplitude at :math:`\omega_0` also contains oscillations
  with frequencies up to about :math:`\pm 2\pi / T` away (at half height), e.g. continuum oscillations near the
  frequency of the eigenmode. Longer runs separate them better.
- The radius and the continua assume circular, concentric flux surfaces labelled by :math:`\eta_1` and a poloidal
  angle :math:`2 \pi \eta_2` (e.g. ``HollowTorus`` with ``sfl=False``); for a cylinder (``ScrewPinch``) the parameter
  ``R0`` of the equilibrium is the length divided by :math:`2\pi`.
- :func:`mode_coefficients` loads all saved time steps (``Output`` keeps them in memory); select ``t`` for long runs
  of large grids.
- The profile is the spectral component of the response to the initial condition, not an eigenfunction computed
  by an eigenvalue solver; near a resonance with the continuum it contains resonant contributions.

Example (radial velocity of a shear Alfven wave, n = -1 along :math:`\eta_3`)::

    from struphy import Output
    from struphy.post_processing import mode_analysis as ma

    out = Output("sim_data")
    coeffs = ma.mode_coefficients(out, "mhd/velocity", ms=(0, 1, 2, 3), n=-1, component=0)
    spectrum = ma.time_spectrum(coeffs)
    omega = ma.peak_frequency(spectrum, m=1, r=0.5)
    profiles = ma.filtered_profile(spectrum, omega=omega)
    ma.plot_omega_r(spectrum, out=out, expected=0.067).fig.savefig("omega_r.png")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

__all__ = [
    "minor_radius",
    "mode_coefficients",
    "time_spectrum",
    "peak_frequency",
    "filtered_profile",
    "cylindrical_continuum",
    "plot_spectrum",
    "plot_omega_r",
    "plot_profiles",
]


def minor_radius(domain) -> Callable[[np.ndarray], np.ndarray] | None:
    r"""The minor radius :math:`r(\eta_1)` of domains with circular cross-sections (``HollowTorus`` and
    ``HollowCylinder``: :math:`a_1 + (a_2 - a_1) \eta_1`; ``IGAPolarTorus`` and ``IGAPolarCylinder``: :math:`a \eta_1`),
    else None."""
    params = getattr(domain, "params", {}) or {}
    if "a1" in params and "a2" in params:
        a1, a2 = params["a1"], params["a2"]
        return lambda eta1: a1 + (a2 - a1) * np.asarray(eta1)
    if type(domain).__name__ in ("IGAPolarTorus", "IGAPolarCylinder") and "a" in params:
        a = params["a"]
        return lambda eta1: a * np.asarray(eta1)
    return None


def mode_coefficients(
    out,
    name: str,
    ms: Sequence[int] = (0, 1, 2, 3),
    n: int = -1,
    component: int | None = 0,
    representation: str | None = None,
    num_eta1: int = 48,
    num_eta2: int = 32,
    num_eta3: int = 8,
    t=None,
) -> xr.DataArray:
    r"""The Fourier coefficients :math:`c_{m,n}(\eta_1, t)` in :math:`(\eta_2, \eta_3)` of one component of a FEEC
    field at all (or the selected) saved times, see :mod:`struphy.post_processing.mode_analysis`.

    Parameters
    ----------
    out : Output
        The output of the run.

    name : str
        The FEEC field, e.g. ``"mhd/velocity"`` (see :meth:`~struphy.Output.evaluate`).

    ms : Sequence[int]
        Poloidal mode numbers (along :math:`\eta_2`).

    n : int
        Toroidal mode number along :math:`\eta_3` (logical; the physical one is ``n`` times ``tor_period``).

    component : int | None
        Component of a vector field; None for scalar fields.

    representation : str | None
        Representation of the field (see :meth:`~struphy.Output.evaluate`); None for the default of
        :meth:`~struphy.Output.evaluate`: ``"norm"`` for vector fields (components in the normalized basis, component 0
        is the physical radial component for circular cross-sections with ``sfl=False``), ``"0"`` for scalar fields.

    num_eta1, num_eta2, num_eta3 : int
        Number of evaluation points: cell-centred in :math:`\eta_1`, equidistant in :math:`\eta_2, \eta_3` (at least
        ``2 max|m| + 1`` and ``2 |n| + 1`` to avoid aliasing).

    t : optional
        Saved time steps (see :meth:`~struphy.Output.evaluate`); None for all.

    Returns
    -------
    xr.DataArray
        Complex coefficients with dims ``("t", "eta1", "m")`` and the coordinate ``r`` (minor radius, if
        :func:`minor_radius` knows the domain; else :math:`\eta_1`) along ``eta1``; attribute ``n``.
    """
    ms = tuple(int(m) for m in ms)
    if num_eta2 < 2 * max(abs(m) for m in ms) + 1:
        raise ValueError("num_eta2 is too small for the poloidal mode numbers.")
    if num_eta3 < 2 * abs(n) + 1:
        raise ValueError("num_eta3 is too small for the toroidal mode number.")
    eta1 = (np.arange(num_eta1) + 0.5) / num_eta1
    eta2 = np.linspace(0.0, 1.0, num_eta2, endpoint=False)
    eta3 = np.linspace(0.0, 1.0, num_eta3, endpoint=False)
    kwargs = {"component": component} if component is not None else {}
    data = out.evaluate(name, representation=representation, eta1=eta1, eta2=eta2, eta3=eta3, t=t, **kwargs)
    for dim in ("eta1", "eta2", "eta3"):
        if dim not in data.dims:
            data = data.expand_dims(dim)
    if "t" not in data.dims:
        data = data.expand_dims("t")
    extra = [d for d in data.dims if d not in ("t", "eta1", "eta2", "eta3")]
    data = data.squeeze(extra, drop=True) if extra else data
    values = np.asarray(data.transpose("t", "eta1", "eta2", "eta3").values)
    c = np.fft.fft2(values, axes=(2, 3)) / (num_eta2 * num_eta3)
    coeffs = np.stack([c[:, :, m % num_eta2, n % num_eta3] for m in ms], axis=-1)
    radius = minor_radius(out.domain)
    r = radius(eta1) if radius is not None else eta1
    return xr.DataArray(
        coeffs,
        dims=("t", "eta1", "m"),
        coords={"t": np.asarray(data["t"].values, dtype=float), "eta1": eta1, "m": list(ms), "r": ("eta1", r)},
        attrs={"n": int(n), "field": name, "component": component, "representation": representation},
        name=f"c_m{n:+d}",
    )


def time_spectrum(coeffs: xr.DataArray, pad: int = 8) -> xr.DataArray:
    r"""The amplitude spectrum :math:`A_m(\eta_1, \omega)` (:math:`\omega \geq 0`, both signs combined) of the
    coefficients of :func:`mode_coefficients`: time mean subtracted, Hann window, FFT zero-padded ``pad`` times.

    Parameters
    ----------
    coeffs : xr.DataArray
        Coefficients with dims ``("t", "eta1", "m")`` at equidistant times (at least 4).

    pad : int
        Zero-padding factor (interpolation of the spectrum, not more resolution).

    Returns
    -------
    xr.DataArray
        Amplitudes with dims ``("omega", "eta1", "m")`` and the coordinate ``r``; attributes ``resolution``
        (:math:`2 \pi / T`), ``T`` and ``n``.
    """
    t = np.asarray(coeffs["t"].values, dtype=float)
    if t.size < 4:
        raise ValueError("A spectrum needs at least 4 saved times.")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6):
        raise ValueError("The saved times must be equidistant.")
    c = np.asarray(coeffs.transpose("t", "eta1", "m").values)
    window = np.hanning(t.size)[:, None, None]
    nfft = pad * t.size
    spec = np.fft.fft((c - c.mean(axis=0)) * window, n=nfft, axis=0)
    omega = 2 * np.pi * np.fft.fftfreq(nfft, d=dt[0])
    neg = (-np.arange(nfft)) % nfft
    keep = omega >= 0
    amp = np.sqrt(np.abs(spec) ** 2 + np.abs(spec[neg]) ** 2)[keep]
    T = t[-1] - t[0]
    return xr.DataArray(
        amp,
        dims=("omega", "eta1", "m"),
        coords={"omega": omega[keep], "eta1": coeffs["eta1"].values, "m": coeffs["m"].values, "r": coeffs["r"]},
        attrs={**coeffs.attrs, "resolution": 2 * np.pi / T, "T": T},
        name="A",
    )


def _radial_index(spectrum: xr.DataArray, r: float | None, eta1: float | None) -> int:
    if r is not None:
        return int(np.argmin(np.abs(spectrum["r"].values - r)))
    if eta1 is not None:
        return int(np.argmin(np.abs(spectrum["eta1"].values - eta1)))
    raise ValueError("Give r or eta1.")


def peak_frequency(
    spectrum: xr.DataArray, m: int, r: float | None = None, eta1: float | None = None, omega_max: float | None = None
) -> float:
    r"""The frequency of the largest local maximum of :math:`A_m(\eta_1, \omega)` at the radius ``r`` (or at
    ``eta1``) with :math:`0 < \omega <` ``omega_max`` (a maximum has smaller neighbours on both sides, so the zero
    frequency and the end of the range are excluded); NaN if there is none (e.g. a constant signal). A signal without
    oscillation (growing or decaying) still has the small local maxima of the side lobes of the window: check the
    spectrum (:func:`plot_spectrum`)."""
    i = _radial_index(spectrum, r, eta1)
    w = spectrum["omega"].values
    a = spectrum.sel(m=m).values[:, i]
    inner = np.arange(1, w.size - 1)
    is_max = (a[inner] > a[inner - 1]) & (a[inner] >= a[inner + 1])
    if omega_max is not None:
        is_max &= w[inner] < omega_max
    candidates = inner[is_max]
    if candidates.size == 0:
        return float("nan")
    return float(w[candidates[np.argmax(a[candidates])]])


def filtered_profile(
    spectrum: xr.DataArray, omega: float | None = None, band: tuple[float, float] | None = None
) -> xr.DataArray:
    r"""The mode structure: :math:`A_m(\eta_1, \omega)` at the frequency bin nearest ``omega``, or the root of the
    power summed over the frequency ``band`` (lower, upper); dims ``("eta1", "m")`` with the coordinate ``r``."""
    if (omega is None) == (band is None):
        raise ValueError("Give either omega or band.")
    w = spectrum["omega"].values
    if omega is not None:
        out = spectrum.isel(omega=int(np.argmin(np.abs(w - omega))))
        return out.drop_vars("omega").assign_attrs(omega=float(omega))
    sel = (w >= band[0]) & (w <= band[1])
    if not np.any(sel):
        raise ValueError("No frequency in the band.")
    power = (spectrum.isel(omega=np.flatnonzero(sel)) ** 2).sum("omega")
    return np.sqrt(power).assign_attrs(spectrum.attrs, band=tuple(band))


def cylindrical_continuum(out, r: np.ndarray, m: int, n: int, num_eta2: int = 32) -> np.ndarray:
    r"""The cylindrical shear Alfven continuum :math:`\omega_m(r) = |n q(r) - m| / (q(r) R_0) \, v_A(r)` of the
    equilibrium of ``out`` (it needs ``q_r`` and the parameter ``R0``), with the physical toroidal mode number ``n``
    and :math:`v_A = |B_0| / \sqrt{n_0}` from the flux-surface averages of the equilibrium (at :math:`\eta_3 = 0`,
    in units of the Alfven velocity); the radius is mapped to :math:`\eta_1` by :func:`minor_radius`."""
    equil = out.equil
    if not hasattr(equil, "q_r") or "R0" not in equil.params:
        raise NotImplementedError(f"{type(equil).__name__} has no safety factor profile q_r or no parameter R0.")
    radius = minor_radius(out.domain)
    if radius is None:
        raise NotImplementedError(f"No minor radius for the domain {type(out.domain).__name__}.")
    r = np.asarray(r, dtype=float)
    r0, r1 = float(radius(0.0)), float(radius(1.0))
    eta1 = np.clip((r - r0) / (r1 - r0), 1e-6, 1.0)
    eta2 = np.linspace(0.0, 1.0, num_eta2, endpoint=False)
    eta3 = np.array([0.0])
    abs_b = np.mean(np.asarray(equil.absB0(eta1, eta2, eta3)).reshape(r.size, -1), axis=1)
    density = np.mean(np.asarray(equil.n0(eta1, eta2, eta3)).reshape(r.size, -1), axis=1)
    q = np.asarray(equil.q_r(r), dtype=float)
    return np.abs(n * q - m) / (q * equil.params["R0"]) * abs_b / np.sqrt(density)


@dataclass
class ModePlot:
    """A figure of this module and its axes."""

    fig: plt.Figure
    axes: list


def _lines(ax, expected, measured):
    if expected is not None:
        ax.axvline(expected, color="k", ls="--", lw=1, label=f"expected {expected:.4g}")
    if measured is not None:
        ax.axvline(measured, color="C3", ls=":", lw=1.5, label=f"measured {measured:.4g}")


def plot_spectrum(
    spectrum: xr.DataArray,
    r: float,
    ms: Sequence[int] | None = None,
    expected: float | None = None,
    measured: float | None = None,
    tolerance: float | None = None,
    omega_max: float | None = None,
) -> ModePlot:
    r"""Power spectra :math:`A_m^2` at the radius ``r`` (normalized to the largest one), the spectrum summed over all
    radii and harmonics, the ``expected`` frequency (with a relative ``tolerance`` band) and the ``measured`` one."""
    i = _radial_index(spectrum, r, None)
    w = spectrum["omega"].values
    w_max = omega_max or (3 * expected if expected else w[-1])
    sel = (w > 0) & (w <= w_max)
    ms = list(spectrum["m"].values) if ms is None else list(ms)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    powers = {m: spectrum.sel(m=m).values[:, i] ** 2 for m in ms}
    scale = max(p[sel].max() for p in powers.values())
    for k, m in enumerate(ms):
        ax.semilogy(
            w[sel], powers[m][sel] / scale, color=f"C{k}", label=f"m = {m} at r = {float(spectrum['r'][i]):.2f}"
        )
    total = (spectrum.values**2).sum(axis=(1, 2))
    ax.semilogy(w[sel], total[sel] / total[sel].max(), "k", lw=1, label="all m, all r")
    if expected is not None and tolerance is not None:
        ax.axvspan((1 - tolerance) * expected, (1 + tolerance) * expected, color="grey", alpha=0.2)
    _lines(ax, expected, measured)
    ax.set_xlim(0, w_max)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("power (normalized)")
    res = spectrum.attrs.get("resolution", float("nan"))
    ax.set_title(f"Time spectra, n = {spectrum.attrs.get('n')}  ($2\\pi/T$ = {res:.3g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return ModePlot(fig, [ax])


def plot_omega_r(
    spectrum: xr.DataArray,
    out=None,
    n_physical: int | None = None,
    expected: float | None = None,
    omega_max: float | None = None,
    continua: Sequence[int] | None = None,
    decades: float = 3.0,
) -> ModePlot:
    r""":math:`\log_{10} A_m` over :math:`(r, \omega)`, one panel per harmonic, normalized to the largest amplitude.
    With ``out``, the cylindrical continua (:func:`cylindrical_continuum`, poloidal mode numbers ``continua``, default
    those of the spectrum and their neighbours; the one of the panel solid) of the physical toroidal mode number
    ``n_physical`` (default: ``n`` of the spectrum times ``tor_period`` of the domain, sign ignored)."""
    w = spectrum["omega"].values
    w_max = omega_max or (3 * expected if expected else w[-1])
    sel = w <= w_max
    r = spectrum["r"].values
    ms = list(spectrum["m"].values)
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 4.5), sharey=True, squeeze=False)
    axes = list(axes[0])
    vmax = spectrum.values[sel].max()
    vmax = vmax if vmax > 0 else 1.0
    curves = {}
    if out is not None:
        if n_physical is None:
            n_physical = abs(spectrum.attrs.get("n", 1)) * out.domain.params.get("tor_period", 1)
        r_fine = np.linspace(r.min(), r.max(), 200)
        for mc in continua if continua is not None else range(max(min(ms) - 1, 0), max(ms) + 2):
            try:
                curves[mc] = (r_fine, cylindrical_continuum(out, r_fine, mc, n_physical))
            except NotImplementedError:
                break
    im = None
    for ax, m in zip(axes, ms):
        a = spectrum.sel(m=m).values[sel] / vmax
        im = ax.pcolormesh(r, w[sel], np.log10(a + 10.0 ** (-decades - 1)), vmin=-decades, vmax=0, shading="auto")
        for mc, (rf, wc) in curves.items():
            ax.plot(rf, wc, "w-" if mc == m else "w:", lw=1)
        if expected is not None:
            ax.axhline(expected, color="r", ls="--", lw=1)
        ax.set_ylim(0, w_max)
        ax.set_xlabel("r")
        ax.set_title(f"m = {m}")
    axes[0].set_ylabel(r"$\omega$")
    fig.colorbar(im, ax=axes, label=r"$\log_{10} A_m$ (normalized)")
    return ModePlot(fig, axes)


def plot_profiles(profiles: xr.DataArray, title: str | None = None, mark_r: float | None = None) -> ModePlot:
    """The mode structure of :func:`filtered_profile` over the minor radius, normalized to its maximum."""
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = profiles.transpose("eta1", "m").values
    scale = vals.max() if vals.max() > 0 else 1.0
    r = profiles["r"].values
    for k, m in enumerate(profiles["m"].values):
        ax.plot(r, vals[:, k] / scale, label=f"m = {m}")
    if mark_r is not None:
        ax.axvline(mark_r, color="k", ls=":", lw=1)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$|A_m(r, \omega)|$ (normalized)")
    omega = profiles.attrs.get("omega")
    band = profiles.attrs.get("band")
    default = f"Mode structure at omega = {omega:.4g}" if omega is not None else f"Mode structure, band {band}"
    ax.set_title(title or default)
    ax.legend()
    fig.tight_layout()
    return ModePlot(fig, [ax])
