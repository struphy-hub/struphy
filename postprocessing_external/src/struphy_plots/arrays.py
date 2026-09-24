"""Small xarray metadata helpers used by :mod:`struphy_plots`.

They intentionally live here rather than in Struphy so the plotting package can
operate on labeled xarray data from any producer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

DIM_LABELS = {
    "t": r"$t$", "e1": r"$\eta_1$", "e2": r"$\eta_2$", "e3": r"$\eta_3$",
    "v1": r"$v_1$", "v2": r"$v_2$", "v3": r"$v_3$", "x": r"$x$", "y": r"$y$",
    "z": r"$z$", "R": r"$R$", "Z": r"$Z$", "component": "component",
    "marker": "marker", "quantity": "quantity",
}
SCALARS_EXCLUDE = ("time",)


def validate_array(data: xr.DataArray, *, required_dims: Sequence[str] = ()) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"expected xarray.DataArray, got {type(data).__name__}")
    missing = tuple(dim for dim in required_dims if dim not in data.dims)
    if missing:
        raise ValueError(f"missing dimensions {missing}; available dimensions are {data.dims}")
    return data


def axis_label(data: xr.DataArray, dim: str) -> str:
    if dim not in data.dims:
        raise KeyError(f"dimension {dim!r} not found in {data.dims}")
    coord = data.coords.get(dim)
    label = ("" if coord is None else coord.attrs.get("long_name", "")) or DIM_LABELS.get(dim, dim)
    unit = "" if coord is None else coord.attrs.get("units", "")
    return f"{label} [{unit}]" if unit else label


def value_label(data: xr.DataArray) -> str:
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


def save_scalars(scalars: xr.Dataset | Mapping, path: str, *, names=None, exclude=SCALARS_EXCLUDE, fmt=None) -> str:
    selected = scalar_names(scalars, names=names, exclude=exclude)
    arrays = [scalars[name] for name in selected]
    for array in arrays:
        validate_array(array, required_dims=("t",))
        if array.dims != ("t",):
            raise ValueError(f"scalar {array.name!r} must have only the 't' dimension, got {array.dims}")
    if arrays:
        arrays = xr.align(*arrays, join="exact")
        time = np.asarray(arrays[0].coords["t"])
        values = np.column_stack([np.asarray(array) for array in arrays])
    else:
        time, values = np.zeros(0), np.zeros((0, 0))
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "csv").lower()
    if fmt == "npz":
        np.savez(path, t=time, **{name: values[:, i] for i, name in enumerate(selected)})
    elif fmt == "csv":
        np.savetxt(path, np.column_stack((time, values)), delimiter=",", header=",".join(("t", *selected)), comments="")
    else:
        raise ValueError(f"unknown format {fmt!r}, expected 'csv' or 'npz'")
    return path
