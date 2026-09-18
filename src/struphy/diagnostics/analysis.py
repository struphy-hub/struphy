"""Numerical diagnostics returning values and labeled arrays, without rendering."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from struphy.post_processing.arrays import validate_array


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


@dataclass(frozen=True)
class GrowthFit:
    """Configuration for an exponential growth-rate fit."""

    window: tuple[float | None, float | None] = (None, None)
    amplitude_from_quadratic: bool = False


@dataclass(frozen=True)
class FitResult:
    rate: float
    intercept: float
    time: np.ndarray
    fitted: np.ndarray


def growth_rate(data: xr.DataArray, fit: GrowthFit | None = None) -> FitResult | None:
    """Fit ``exp(rate*t + intercept)`` using only finite, positive samples."""
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"growth-rate input must have dims ('t',), got {data.dims}")
    fit = fit or GrowthFit()
    time, values = np.asarray(data.t), np.asarray(data)
    if len(time) < 2:
        return None
    lo = time[0] if fit.window[0] is None else fit.window[0]
    hi = time[-1] if fit.window[1] is None else fit.window[1]
    lo, hi = sorted((lo, hi))
    valid = (time >= lo) & (time <= hi) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(valid) < 2:
        return None
    selected_time = time[valid]
    signal = np.log(np.sqrt(values[valid])) if fit.amplitude_from_quadratic else np.log(values[valid])
    rate, intercept = np.polyfit(selected_time, signal, 1)
    scale = 2.0 if fit.amplitude_from_quadratic else 1.0
    fitted = np.exp(scale * (rate * selected_time + intercept))
    return FitResult(float(rate), float(intercept), selected_time, fitted)


def envelope(data: xr.DataArray) -> xr.DataArray:
    """Local maxima of a time series: the interior samples not smaller than their neighbours."""
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"envelope input must have dims ('t',), got {data.dims}")
    values = np.asarray(data)
    peak = np.zeros(len(values), dtype=bool)
    peak[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    return data.isel(t=np.flatnonzero(peak))


def damping_rate(data: xr.DataArray, fit: GrowthFit | None = None) -> FitResult | None:
    """Fit ``exp(rate*t + intercept)`` to the envelope of an oscillating time series.

    Use this for signals such as the field energy in Landau damping, where :func:`growth_rate` on
    the raw series would fit the oscillation. ``fit.window`` restricts the peaks that are used.
    The rate is negative for damping.
    """
    return growth_rate(envelope(data), fit)


def norm(data: xr.DataArray, *, dims=None, squared: bool = False) -> xr.DataArray:
    """L2 norm over ``dims`` (default: every dimension except ``t``), as a function of the rest."""
    validate_array(data)
    dims = [dim for dim in data.dims if dim != "t"] if dims is None else list(dims)
    total = (data**2).sum(dims)
    out = total if squared else np.sqrt(total)
    out.attrs = {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}
    label = _label(data)
    out.attrs["label"] = f"squared norm of {label}".strip() if squared else f"norm of {label}".strip()
    return out


def drift(data: xr.DataArray, *, ref=None) -> xr.DataArray:
    """Signed deviation from an explicit reference or the first time sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    out = data - reference
    out.attrs = dict(data.attrs)
    out.attrs["label"] = f"{_label(data)} drift".strip()
    return out


def relative_error(data: xr.DataArray, *, ref=None, skip_first=True) -> xr.DataArray:
    """Absolute relative deviation from an explicit reference or first sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    if np.any(np.asarray(reference) == 0):
        raise ValueError("cannot take a relative error against a reference of zero")
    out = abs(data - reference) / abs(reference)
    out.attrs = {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}
    out.attrs.update(label=f"relative error of {_label(data)}".strip(), units="")
    return out.isel(t=slice(1, None)) if skip_first else out
