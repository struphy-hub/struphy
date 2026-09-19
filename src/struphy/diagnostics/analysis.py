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


SPATIAL_DIMS = ("e1", "e2", "e3")
VELOCITY_DIMS = ("v1", "v2", "v3")


def _provenance(data: xr.DataArray) -> dict:
    return {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}


def _select_dims(data: xr.DataArray, dims, default) -> list[str]:
    if dims is None:
        selected = [dim for dim in default if dim in data.dims]
        if not selected:
            raise ValueError(f"{data.name!r} has none of the dimensions {default}; its dimensions are {data.dims}")
        return selected
    selected = [dims] if isinstance(dims, str) else list(dims)
    missing = [dim for dim in selected if dim not in data.dims]
    if missing:
        raise ValueError(f"{data.name!r} has no dimensions {missing}; its dimensions are {data.dims}")
    return selected


def spatial_average(data: xr.DataArray, *, dims=None) -> xr.DataArray:
    """Mean over the logical space dimensions, e.g. a binned f(t, e1, v1) becomes f(t, v1).

    ``dims`` defaults to every one of ``e1``, ``e2``, ``e3`` that ``data`` has. The mean is
    uniform in the logical coordinates, which is the volume average on a Cartesian domain; on a
    mapped domain it is not weighted by the Jacobian. Physical ``X``, ``Y``, ``Z`` coordinates
    that depend on the averaged dimensions are dropped.
    """
    validate_array(data)
    averaged = _select_dims(data, dims, SPATIAL_DIMS)
    out = data.mean(averaged, keep_attrs=True)
    out.attrs["label"] = f"average of {_label(data)}".strip()
    out.attrs.pop("long_name", None)
    return out


def _bin_widths(data: xr.DataArray, dim: str) -> xr.DataArray:
    coordinate = np.asarray(data.coords[dim]) if dim in data.coords else None
    if coordinate is None or len(coordinate) < 2:
        raise ValueError(f"dimension {dim!r} needs a coordinate with at least two bins")
    return xr.DataArray(np.gradient(coordinate), dims=(dim,), coords={dim: data.coords[dim]})


def velocity_moments(f: xr.DataArray, *, dims=None) -> xr.Dataset:
    """Moments of a binned distribution function over its velocity dimensions.

    ``dims`` defaults to every one of ``v1``, ``v2``, ``v3`` that ``f`` has; the moments are
    functions of the remaining dimensions, for example ``(t, e1)`` for an ``e1_v1`` product.
    The integrals are sums over the bins, weighted by the bin widths.

    Returns a Dataset with

    * ``density``: the zeroth moment, :math:`\\int f\\,\\mathrm{d}v`.
    * ``mean_<dim>``: the mean velocity :math:`u = \\int v f\\,\\mathrm{d}v / n` along each dimension.
    * ``variance_<dim>``: :math:`\\int (v-u)^2 f\\,\\mathrm{d}v / n`. In normalized units this is the
      temperature over the particle mass along that direction, :math:`T/m`.

    Where the density is not positive, the mean and variance are NaN. A ``delta_f`` product has
    only the density, which is then the density perturbation, because its mean and variance are
    not defined. The values keep the normalization of the run; see ``Output.to_si``.
    """
    validate_array(f)
    integrated = _select_dims(f, dims, VELOCITY_DIMS)
    volume = 1.0
    for dim in integrated:
        volume = volume * _bin_widths(f, dim)
    density = (f * volume).sum(integrated)

    label = _label(f)
    variables = {"density": (density, f"density of {label}")}
    if f.name != "delta_f":
        weight = density.where(density > 0)
        for dim in integrated:
            mean = (f * f[dim] * volume).sum(integrated) / weight
            variance = (f * (f[dim] - mean) ** 2 * volume).sum(integrated) / weight
            variables[f"mean_{dim}"] = (mean, f"mean {dim}")
            variables[f"variance_{dim}"] = (variance, f"variance of {dim}")

    provenance = _provenance(f)
    out = {}
    for name, (values, description) in variables.items():
        values.attrs = {**provenance, "label": description.strip()}
        out[name] = values.rename(name)
    return xr.Dataset(out, attrs=provenance)


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
