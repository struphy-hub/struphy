"""Labeled post-processing arrays built on :mod:`xarray`."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

logger = logging.getLogger("struphy")

DIM_LABELS = {
    "t": r"$t$",
    "e1": r"$\eta_1$",
    "e2": r"$\eta_2$",
    "e3": r"$\eta_3$",
    "v1": r"$v_1$",
    "v2": r"$v_2$",
    "v3": r"$v_3$",
    "x": r"$x$",
    "y": r"$y$",
    "z": r"$z$",
    "R": r"$R$",
    "Z": r"$Z$",
    "component": "component",
    "marker": "marker",
    "attribute": "attribute",
}
BINNED_LABELS = {"f_binned": "$f$", "delta_f_binned": r"$\delta f$", "n_sph": "$n$"}
SCALARS_EXCLUDE = ("time",)


def data_array(
    values,
    dims: Sequence[str],
    coords: Mapping | None = None,
    *,
    name: str | None = None,
    label: str = "",
    unit: str = "",
    coord_units: Mapping[str, str] | None = None,
    attrs: Mapping | None = None,
) -> xr.DataArray:
    """Construct a consistently annotated :class:`xarray.DataArray`.

    ``label`` is also stored as the CF attribute ``long_name``, so that xarray's own plots
    (``array.plot()``) label their axes the same way Struphy's do.
    """
    metadata = dict(attrs or {})
    metadata["label"] = label
    if unit:  # an empty unit would render as "[]" in xarray's own plots
        metadata["units"] = unit
    if label:
        metadata.setdefault("long_name", label)
    out = xr.DataArray(values, dims=tuple(dims), coords=coords, name=name, attrs=metadata)
    for dim, value in (coord_units or {}).items():
        if dim in out.coords and value:
            out.coords[dim].attrs["units"] = value
    for dim in out.dims:
        if dim in out.coords and "long_name" not in out.coords[dim].attrs and dim in DIM_LABELS:
            out.coords[dim].attrs["long_name"] = DIM_LABELS[dim]
    return validate_array(out)


def validate_array(data: xr.DataArray, *, required_dims: Sequence[str] = ()) -> xr.DataArray:
    """Validate the inexpensive invariants diagnostics rely on."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"expected xarray.DataArray, got {type(data).__name__}")
    missing = tuple(dim for dim in required_dims if dim not in data.dims)
    if missing:
        raise ValueError(f"missing dimensions {missing}; available dimensions are {data.dims}")
    for dim in data.dims:
        if dim not in data.coords or data.coords[dim].ndim != 1:
            continue
        coord = np.asarray(data.coords[dim])
        if len(coord) > 1 and np.issubdtype(coord.dtype, np.number):
            delta = np.diff(coord)
            if not (np.all(delta > 0) or np.all(delta < 0)):
                raise ValueError(f"coordinate {dim!r} must be strictly monotonic")
    return data


def axis_label(data: xr.DataArray, dim: str) -> str:
    """Human-readable coordinate label, including a unit when available."""
    if dim not in data.dims:
        raise KeyError(f"dimension {dim!r} not found in {data.dims}")
    coord = data.coords.get(dim)
    label = ("" if coord is None else coord.attrs.get("long_name", "")) or DIM_LABELS.get(dim, dim)
    unit = "" if coord is None else coord.attrs.get("units", "")
    return f"{label} [{unit}]" if unit else label


def value_label(data: xr.DataArray) -> str:
    """Human-readable value label, including the value unit."""
    label = data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""
    unit = data.attrs.get("units", "") or "a.u."
    return f"{label} [{unit}]" if label else f"[{unit}]"


def scalar_names(scalars: xr.Dataset | Mapping, *, names=None, exclude=SCALARS_EXCLUDE) -> list[str]:
    available = tuple(scalars.data_vars if isinstance(scalars, xr.Dataset) else scalars.keys())
    if names is not None:
        missing = [name for name in names if name not in available]
        if missing:
            raise KeyError(f"no scalars {missing}, available: {available}")
        return list(names)
    return [name for name in available if name not in exclude]


def scalars_table(scalars: xr.Dataset | Mapping, *, names=None, exclude=SCALARS_EXCLUDE):
    """Return ``(time, names, values)`` for scalar export."""
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        return np.zeros(0), [], np.zeros((0, 0))
    arrays = [scalars[name] for name in selected]
    for array in arrays:
        validate_array(array, required_dims=("t",))
        if array.dims != ("t",):
            raise ValueError(f"scalar {array.name!r} must have only the 't' dimension, got {array.dims}")
    aligned = xr.align(*arrays, join="exact")
    return np.asarray(aligned[0].coords["t"]), selected, np.column_stack([np.asarray(a) for a in aligned])


def save_scalars(scalars: xr.Dataset | Mapping, path: str, *, names=None, exclude=SCALARS_EXCLUDE, fmt=None) -> str:
    """Write selected scalar time series to CSV or NPZ."""
    time, selected, values = scalars_table(scalars, names=names, exclude=exclude)
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "csv").lower()
    if fmt not in {"csv", "npz"}:
        raise ValueError(f"unknown format {fmt!r}, expected 'csv' or 'npz'")
    if fmt == "npz" and not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "npz":
        np.savez(path, t=time, **{name: values[:, i] for i, name in enumerate(selected)})
    else:
        np.savetxt(path, np.column_stack((time, values)), delimiter=",", header=",".join(("t", *selected)), comments="")
    logger.info("Wrote %d scalars over %d time steps to %s", len(selected), len(time), path)
    return path


def orbit_columns(n_columns: int) -> dict:
    columns = {"position": slice(0, 3), "id": n_columns - 1}
    if n_columns == 8:
        columns.update(velocity=slice(3, 6), weight=6)
    elif n_columns == 5:
        columns["velocity"] = 3
    else:
        columns["velocity"] = slice(3, n_columns - 1)
    return columns


def wrap_orbits(values, time, *, time_unit="") -> xr.DataArray:
    """Label marker orbits with time, marker and attribute dimensions."""
    values = np.asarray(values)
    return data_array(
        values,
        ("t", "marker", "attribute"),
        {"t": time, "marker": np.arange(values.shape[1]), "attribute": np.arange(values.shape[2])},
        name="orbits",
        label="marker orbits",
        coord_units={"t": time_unit},
        attrs={"columns": orbit_columns(values.shape[-1])},
    )


def wrap_field_data(
    values_by_time: Mapping,
    grids_log=None,
    *,
    grids_phy=None,
    name: str = "",
    time_scale: float = 1.0,
    time_unit: str = "",
) -> xr.DataArray | None:
    """Stack one field product and attach logical and physical coordinates."""
    times = sorted(values_by_time)
    if not times:
        return None
    first = values_by_time[times[0]]
    scalar = not isinstance(first, (list, tuple)) or len(first) == 1
    if scalar:
        values = np.stack(
            [
                np.asarray(
                    values_by_time[t] if not isinstance(values_by_time[t], (list, tuple)) else values_by_time[t][0]
                )
                for t in times
            ]
        )
        dims = ("t", "e1", "e2", "e3")
    else:
        values = np.stack([np.stack([np.asarray(c) for c in values_by_time[t]]) for t in times])
        dims = ("t", "component", "e1", "e2", "e3")
    coords: dict = {"t": np.asarray(times) * time_scale}
    if "component" in dims:
        coords["component"] = np.arange(values.shape[1])
    if grids_log is not None:
        for i, grid in enumerate(grids_log, 1):
            dim = f"e{i}"
            if len(grid) == values.shape[dims.index(dim)]:
                coords[dim] = np.asarray(grid)
    spatial_shape = tuple(values.shape[dims.index(dim)] for dim in ("e1", "e2", "e3"))
    if grids_phy is not None and all(np.asarray(grid).shape == spatial_shape for grid in grids_phy):
        for coordinate, grid in zip(("X", "Y", "Z"), grids_phy):
            coords[coordinate] = (("e1", "e2", "e3"), np.asarray(grid))
    return data_array(values, dims, coords, name=name or None, label=name, coord_units={"t": time_unit})


def wrap_binned_data(values, dims: Sequence[str], coords: Mapping, *, name: str, time_unit: str = "") -> xr.DataArray:
    """Label a memory-mapped binned distribution or density product."""
    return data_array(
        values,
        ("t", *dims),
        coords,
        name=name,
        label=BINNED_LABELS.get(name, name.replace("_", " ")),
        coord_units={"t": time_unit},
    )
