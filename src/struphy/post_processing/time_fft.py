"""Fourier diagnostics of labeled arrays, computed on demand without changing saved products.

The dominant-band filter follows the TAE_example_Shrut workflow, with xarray
coordinates replacing the old dictionaries of post-processed snapshots.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.signal.windows import get_window


def _samples(data, dim, *, real=False):
    if not isinstance(data, xr.DataArray):
        raise TypeError("expected an xarray.DataArray")
    if dim not in data.dims or dim not in data.coords or data.coords[dim].dims != (dim,):
        raise ValueError(f"{dim!r} must be a dimension with a one-dimensional coordinate")
    coordinate = np.asarray(data.coords[dim].values)
    values = np.asarray(data.values)
    if not np.issubdtype(coordinate.dtype, np.number) or np.iscomplexobj(coordinate):
        raise ValueError(f"{dim!r} must have a real numeric coordinate")
    if len(coordinate) < 2 or not np.isfinite(coordinate).all():
        raise ValueError(f"{dim!r} needs at least two finite samples")
    spacing = np.diff(coordinate.astype(float))
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=abs(spacing[0]) * 1e-10):
        raise ValueError(f"{dim!r} must be strictly increasing and uniformly spaced; select a uniform interval first")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("FFT input must contain finite numeric values")
    if real and np.iscomplexobj(values):
        raise ValueError("time_fft requires real values; use fft for complex signals")
    return values, float(spacing[0]), data.get_axis_num(dim)


def _prepare(values, axis, detrend, window):
    if not isinstance(detrend, (bool, np.bool_)):
        raise ValueError("detrend must be a boolean (remove the mean or leave it unchanged)")
    if window not in (None, "hann"):
        raise ValueError("window must be None or 'hann'")
    if detrend:
        values = values - values.mean(axis=axis, keepdims=True)
    if window == "hann":
        shape = [1] * values.ndim
        shape[axis] = values.shape[axis]
        values = values * get_window("hann", values.shape[axis], fftbins=True).reshape(shape)
    return values


def _coefficients(data, values, dim, frequency_dim, frequencies, spacing, detrend, window):
    if frequency_dim in data.coords or frequency_dim in data.dims:
        raise ValueError(f"frequency coordinate {frequency_dim!r} already exists")
    coords = {key: value for key, value in data.coords.items() if dim not in value.dims}
    coords[frequency_dim] = frequencies
    dims = tuple(frequency_dim if name == dim else name for name in data.dims)
    result = xr.DataArray(values, dims=dims, coords=coords, name="coefficients", attrs=dict(data.attrs))
    result.attrs.update(
        transform_dim=dim,
        n_samples=data.sizes[dim],
        sample_spacing=spacing,
        sample_origin=float(data.coords[dim].values[0]),
        frequency_resolution=2 * np.pi / (data.sizes[dim] * spacing),
        nyquist_frequency=np.pi / spacing,
        normalization="forward",
        window=window or "boxcar",
        detrend=bool(detrend),
        label=f"Fourier coefficients of {data.name or 'signal'}",
    )
    result.attrs.pop("long_name", None)
    unit = data.coords[dim].attrs.get("units", "")
    result.coords[frequency_dim].attrs = {
        "long_name": "Angular frequency" if dim == "t" else f"Angular wavenumber along {dim}",
        "units": f"rad / {unit}" if unit else "rad / coordinate unit",
    }
    return result


def fft(data: xr.DataArray, *, dim: str, detrend: bool = False, window: str | None = None) -> xr.DataArray:
    """Two-sided, shifted FFT along a named uniform coordinate, normalized by N.

    Returns complex coefficients with ``dim`` replaced by ``omega`` for time or
    ``k_<dim>`` otherwise. Frequencies are angular (2*pi times cycles per unit
    of the supplied coordinate). Remove duplicate endpoints of periodic spatial
    grids before calling this function; no endpoint is dropped automatically.
    A mean subtraction and a periodic Hann window are optional. Windowed powers
    refer to the windowed signal, without amplitude/energy compensation.
    Coefficient phases are relative to the first sample, recorded as sample_origin.
    """
    values, spacing, axis = _samples(data, dim)
    values = _prepare(values, axis, detrend, window)
    coefficients = np.fft.fftshift(np.fft.fft(values, axis=axis, norm="forward"), axes=axis)
    frequencies = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(data.sizes[dim], spacing))
    return _coefficients(
        data, coefficients, dim, "omega" if dim == "t" else f"k_{dim}", frequencies, spacing, detrend, window
    )


def time_fft(data: xr.DataArray, *, detrend: bool = False, window: str | None = None) -> xr.Dataset:
    """One-sided time FFT with complex coefficients and mean-square power per bin.

    Replaces ``t`` by ``omega`` and preserves other dimensions and coordinates.
    ``coefficients = rfft(data) / N``. ``power`` doubles the positive-frequency
    bins, except the even-N Nyquist bin. Its sum over omega equals the temporal
    mean square of the input (after any mean subtraction/windowing), not a PSD
    per unit frequency. The DC and Nyquist amplitudes are never doubled.

    Sampling uses the saved times, including their units, not the simulation dt.
    Bin spacing is 2*pi/(N*dt); padding is not used to claim extra resolution.
    Coordinates depending on t are dropped; time-independent mapped coordinates
    and provenance are retained. Computation eagerly loads the selected array.
    Phases are relative to the first saved time, recorded as sample_origin.
    """
    values, spacing, axis = _samples(data, "t", real=True)
    values = _prepare(values, axis, detrend, window)
    n = data.sizes["t"]
    frequencies = 2 * np.pi * np.fft.rfftfreq(n, spacing)
    coefficients = _coefficients(
        data, np.fft.rfft(values, axis=axis, norm="forward"), "t", "omega", frequencies, spacing, detrend, window
    )
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    power = abs(coefficients) ** 2 * xr.DataArray(weights, dims="omega", coords={"omega": frequencies})
    power.attrs = {"label": "Mean-square power per frequency bin"}
    if data.attrs.get("units"):
        power.attrs["units"] = f"({data.attrs['units']})^2"
    return xr.Dataset({"coefficients": coefficients, "power": power}, attrs=dict(coefficients.attrs))


def inverse_time_fft(coefficients: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """Invert forward-normalized rFFT coefficients, using template's length and coordinates.

    The original length is required to distinguish odd and even sample counts.
    For windowed/detrended coefficients this reconstructs the processed signal;
    it does not undo the window or add the mean back.
    """
    _, spacing, _ = _samples(template, "t", real=True)
    expected_dims = tuple("omega" if dim == "t" else dim for dim in template.dims)
    if set(coefficients.dims) != set(expected_dims):
        raise ValueError("coefficient dimensions must match the template with t replaced by omega")
    coefficients = coefficients.transpose(*expected_dims)
    expected = 2 * np.pi * np.fft.rfftfreq(template.sizes["t"], spacing)
    if (
        coefficients.attrs.get("n_samples", template.sizes["t"]) != template.sizes["t"]
        or coefficients.sizes["omega"] != len(expected)
        or not np.allclose(coefficients.omega, expected, rtol=1e-7, atol=0)
    ):
        raise ValueError("frequency grid does not match the template")
    for dim in template.dims:
        if dim != "t" and (
            coefficients.sizes[dim] != template.sizes[dim] or not coefficients.coords[dim].equals(template.coords[dim])
        ):
            raise ValueError(f"coefficient coordinate {dim!r} does not match the template")
    if not np.isfinite(coefficients.values).all():
        raise ValueError("coefficients must be finite")
    values = np.fft.irfft(coefficients.values, n=template.sizes["t"], axis=template.get_axis_num("t"), norm="forward")
    return template.copy(data=values)


def fwhm_window(power, idx_peak: int, idx_min: int = 0, pad_bins: int = 0):
    """Inclusive contiguous half-power band about a peak, padded and clamped to valid bins."""
    power = np.asarray(power)
    if power.ndim != 1 or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("power must be a finite, nonnegative one-dimensional array")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (idx_peak, idx_min, pad_bins)
    ):
        raise ValueError("bin indices and pad_bins must be integers")
    if not 0 <= idx_min <= idx_peak < len(power) or pad_bins < 0 or power[idx_peak] <= 0:
        raise ValueError("invalid peak, minimum bin or padding")
    half = power[idx_peak] / 2
    lo = hi = idx_peak
    while lo > idx_min and power[lo - 1] >= half:
        lo -= 1
    while hi + 1 < len(power) and power[hi + 1] >= half:
        hi += 1
    return max(idx_min, lo - pad_bins), min(len(power) - 1, hi + pad_bins)


@dataclass(frozen=True)
class TimeFilterResult:
    """Filtered field and reduced spectrum with the selected band for each retained dimension.

    ``spectrum`` contains power, dominant_frequency, idx_dominant, idx_lo/hi,
    omega_lo/hi and has_peak. A zero/constant signal has no oscillatory peak:
    has_peak=False, indices=-1, frequencies=NaN, and a zero filtered signal.
    """

    filtered: xr.DataArray
    spectrum: xr.Dataset


def filter_time(data: xr.DataArray, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0) -> TimeFilterResult:
    """Keep the dominant peak's FWHM frequency band and reconstruct the real signal.

    Sum power over ``dims`` to select a shared band (default: all dimensions
    except t and component). Each remaining component gets its own band, which
    is applied at every spatial point. The sum is unweighted, not a physical
    energy integral. Pass dims=() for independent filtering at each point.
    Frequencies below positive ``omega_min`` (including DC) are always removed,
    even with padding. This rectangular-bin filter uses no taper; finite records
    can exhibit spectral leakage and edge ringing. Source data is not mutated.
    """
    if not np.isfinite(omega_min) or omega_min <= 0:
        raise ValueError("omega_min must be finite and positive to exclude DC")
    if isinstance(pad_bins, bool) or not isinstance(pad_bins, (int, np.integer)) or pad_bins < 0:
        raise ValueError("pad_bins must be a nonnegative integer")
    transformed = time_fft(data)
    dims = (
        [dim for dim in data.dims if dim not in ("t", "component")]
        if dims is None
        else ([dims] if isinstance(dims, str) else list(dims))
    )
    if len(set(dims)) != len(dims) or any(dim not in data.dims or dim == "t" for dim in dims):
        raise ValueError("dims must name distinct non-time dimensions of the input")
    power = transformed.power.sum(dims, keep_attrs=True)
    retained = tuple(dim for dim in power.dims if dim != "omega")
    power = power.transpose("omega", *retained)
    omega = power.omega.values
    eligible = np.flatnonzero(omega >= omega_min)
    if not len(eligible):
        raise ValueError("omega_min exceeds all available frequency bins")
    powers = power.values.reshape(len(omega), -1)
    indices = np.full((3, powers.shape[1]), -1, dtype=int)
    frequencies = np.full((3, powers.shape[1]), np.nan)
    for column, values in enumerate(powers.T):
        peak = eligible[np.argmax(values[eligible])]
        # Do not report floating-point roundoff from a DC-only signal as a mode.
        if values[peak] <= 100 * np.finfo(float).eps ** 2 * max(values.sum(), np.finfo(float).tiny):
            continue
        lo, hi = fwhm_window(values, int(peak), idx_min=int(eligible[0]), pad_bins=pad_bins)
        indices[:, column] = peak, lo, hi
        frequencies[:, column] = omega[[peak, lo, hi]]
    coords = {key: value for key, value in power.coords.items() if "omega" not in value.dims}
    shape = tuple(power.sizes[dim] for dim in retained)
    spectrum = xr.Dataset({"power": power}, attrs=dict(transformed.attrs))
    for names, values in (
        (["idx_dominant", "idx_lo", "idx_hi"], indices),
        (["dominant_frequency", "omega_lo", "omega_hi"], frequencies),
    ):
        for name, value in zip(names, values):
            spectrum[name] = xr.DataArray(value.reshape(shape), dims=retained, coords=coords)
            if name.startswith("omega") or name == "dominant_frequency":
                spectrum[name].attrs["units"] = transformed.omega.attrs["units"]
    spectrum["has_peak"] = spectrum.idx_dominant >= 0
    spectrum.attrs.update(omega_min=float(omega_min), pad_bins=int(pad_bins), power_reduction_dims=dims)
    mask = (transformed.omega >= spectrum.omega_lo) & (transformed.omega <= spectrum.omega_hi)
    filtered = inverse_time_fft(transformed.coefficients.where(mask, 0), data)
    filtered.attrs.update(time_filter="dominant FWHM band", omega_min=float(omega_min), pad_bins=int(pad_bins))
    return TimeFilterResult(filtered, spectrum)
