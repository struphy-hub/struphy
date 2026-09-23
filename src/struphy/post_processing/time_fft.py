"""FFT of post-processed field time series, and band-pass filtering around the dominant frequency."""

import logging

import cunumpy as xp

logger = logging.getLogger("struphy")


def time_fft(field_series: xp.ndarray, t_grid: xp.ndarray):
    """FFT along the time axis (axis 0) of a real-valued field time series.

    Parameters
    ----------
    field_series : xp.ndarray, shape (n_t, ...)
        Real values at each saved time.
    t_grid : xp.ndarray, 1D, length n_t
        The (assumed uniformly spaced) saved times.

    Returns
    -------
    omega : xp.ndarray, 1D
        Non-negative angular frequencies.
    fft_vals : xp.ndarray, shape (len(omega), ...), complex
        Fourier coefficients corresponding to ``omega``.
    """
    dt = t_grid[1] - t_grid[0]
    fft_vals = xp.fft.rfft(field_series, axis=0)
    freqs = xp.fft.rfftfreq(len(t_grid), d=dt)
    omega = 2 * xp.pi * freqs
    return omega, fft_vals


def power_spectrum(fft_vals: xp.ndarray):
    """Power per frequency bin, summed over all spatial points."""
    return xp.sum(xp.abs(fft_vals) ** 2, axis=tuple(range(1, fft_vals.ndim)))


def find_dominant_frequency(omega: xp.ndarray, power: xp.ndarray, omega_min: float = 1e-8):
    """Angular frequency and bin index of the largest peak of the power spectrum.

    ``omega_min`` excludes the DC (omega=0) bin, which otherwise tends to dominate
    due to any nonzero time-averaged offset, drowning out the real oscillation.
    """
    mask = omega >= omega_min
    idx_local = xp.argmax(power[mask])
    idx_global = int(xp.nonzero(mask)[0][idx_local])
    return omega[idx_global], idx_global


def fwhm_window(power: xp.ndarray, idx_peak: int, idx_min: int = 0, pad_bins: int = 0):
    """Bin range ``[lo, hi]`` (inclusive) of the full width at half maximum around ``idx_peak``.

    Starting at the peak, the window grows to the left and right as long as the
    neighbouring bin still has at least half the peak power. ``pad_bins`` widens the
    resulting window by that many extra bins on each side. The window never extends
    below ``idx_min`` (used to keep the DC bin out).
    """
    half_max = 0.5 * power[idx_peak]
    n = power.shape[0]

    lo = idx_peak
    while lo - 1 >= idx_min and power[lo - 1] >= half_max:
        lo -= 1

    hi = idx_peak
    while hi + 1 <= n - 1 and power[hi + 1] >= half_max:
        hi += 1

    lo = max(idx_min, lo - pad_bins)
    hi = min(n - 1, hi + pad_bins)
    return lo, hi


def band_pass(fft_vals: xp.ndarray, lo: int, hi: int):
    """Zero out every frequency bin outside ``[lo, hi]`` (inclusive)."""
    filtered = xp.zeros_like(fft_vals)
    filtered[lo : hi + 1] = fft_vals[lo : hi + 1]
    return filtered


def inverse_time_fft(filtered_fft_vals: xp.ndarray, n_t: int):
    """Back to the time domain -- filtered, cleaned field time series."""
    return xp.fft.irfft(filtered_fft_vals, n=n_t, axis=0)


def clean_field_pipeline(
    field_series: xp.ndarray,
    t_grid: xp.ndarray,
    omega_min: float = 1e-8,
    pad_bins: int = 0,
):
    """Full pipeline: FFT -> find dominant peak -> keep its FWHM band -> inverse FFT.

    Parameters
    ----------
    field_series : xp.ndarray, shape (n_t, ...)
        Real field values at each saved time.
    t_grid : xp.ndarray
        Saved times (uniformly spaced).
    omega_min : float
        Frequencies below this are ignored when searching for the peak and are
        always filtered out (removes the DC offset).
    pad_bins : int
        Extra bins kept on each side of the FWHM band.

    Returns
    -------
    cleaned_field_series : xp.ndarray
        ``field_series`` with everything outside the FWHM band of the dominant peak removed.
    spectrum : dict
        ``omega`` and unfiltered ``power`` (for plotting/inspection), the dominant
        frequency ``omega_dom`` and its bin ``idx_dom``, and the kept band as bin
        indices ``idx_lo``, ``idx_hi`` and frequencies ``omega_lo``, ``omega_hi``.
    """
    omega, fft_vals = time_fft(field_series, t_grid)
    power = power_spectrum(fft_vals)
    omega_dom, idx_dom = find_dominant_frequency(omega, power, omega_min=omega_min)

    idx_min = int(xp.nonzero(omega >= omega_min)[0][0])
    lo, hi = fwhm_window(power, idx_dom, idx_min=idx_min, pad_bins=pad_bins)
    logger.info(
        f"    Dominant frequency: omega = {omega_dom:.4f}, kept band [{omega[lo]:.4f}, {omega[hi]:.4f}] ({hi - lo + 1} bins)"
    )

    cleaned_field_series = inverse_time_fft(band_pass(fft_vals, lo, hi), n_t=len(t_grid))

    spectrum = {
        "omega": omega,
        "power": power,
        "omega_dom": omega_dom,
        "idx_dom": idx_dom,
        "idx_lo": lo,
        "idx_hi": hi,
        "omega_lo": omega[lo],
        "omega_hi": omega[hi],
    }
    return cleaned_field_series, spectrum


def filter_val_dict(val_dict: dict, pad_bins: int = 0, omega_min: float = 1e-8):
    """Band-pass one post-processed field around its dominant frequency, component by component.

    Parameters
    ----------
    val_dict : dict
        ``{t: [comp_0, comp_1, ...]}`` as stored in ``<var>_log.bin``; keys are the
        (uniformly spaced) saved times in increasing order.
    pad_bins : int
        Extra frequency bins kept on each side of the FWHM band.
    omega_min : float
        See :func:`clean_field_pipeline`.

    Returns
    -------
    filtered_val_dict : dict
        Same layout as ``val_dict``, containing only the band around the dominant frequency.
    fft_data : dict
        ``omega``, ``pad_bins`` and, per component (first axis), the unfiltered ``power``
        spectrum, ``dominant_frequency``, ``idx_dominant`` and the kept band ``idx_lo``,
        ``idx_hi``, ``omega_lo``, ``omega_hi``.
    """
    t_grid = xp.array(list(val_dict.keys()))
    n_comps = len(next(iter(val_dict.values())))
    filtered_val_dict = {t: [None] * n_comps for t in val_dict}
    spectra = []

    for c in range(n_comps):
        field_series = xp.array([comps[c] for comps in val_dict.values()])
        cleaned_series, spectrum = clean_field_pipeline(field_series, t_grid, omega_min=omega_min, pad_bins=pad_bins)
        spectra += [spectrum]
        for i, t in enumerate(val_dict):
            filtered_val_dict[t][c] = cleaned_series[i]

    fft_data = {
        "omega": spectra[0]["omega"],
        "pad_bins": pad_bins,
        "power": xp.stack([s["power"] for s in spectra]),
        "dominant_frequency": xp.array([s["omega_dom"] for s in spectra]),
        "idx_dominant": xp.array([s["idx_dom"] for s in spectra]),
        "idx_lo": xp.array([s["idx_lo"] for s in spectra]),
        "idx_hi": xp.array([s["idx_hi"] for s in spectra]),
        "omega_lo": xp.array([s["omega_lo"] for s in spectra]),
        "omega_hi": xp.array([s["omega_hi"] for s in spectra]),
    }
    return filtered_val_dict, fft_data
