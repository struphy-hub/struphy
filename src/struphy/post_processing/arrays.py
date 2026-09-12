"""Labeled arrays for post-processed Struphy output data."""

from dataclasses import dataclass, field, replace

import cunumpy as xp

#: LaTeX display labels for the dimension names used across Struphy output.
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
    "comp": "component",
    "marker": "marker",
}

#: Display labels for the binned quantities written by the post-processor.
BINNED_LABELS = {
    "f_binned": "$f$",
    "delta_f_binned": r"$\delta f$",
    "n_sph": "$n$",
}

#: Physical unit attribute on ``Units`` that each dimension is measured in.
DIM_UNITS = {
    "t": "t",
    "e1": None,
    "e2": None,
    "e3": None,
    "v1": "v",
    "v2": "v",
    "v3": "v",
    "x": "x",
    "y": "x",
    "z": "x",
    "R": "x",
    "Z": "x",
}


@dataclass
class StruphyArray:
    """Array of simulation output together with its dimension names, coordinates and unit.

    Passing an instance to ``xp.asarray`` or to a matplotlib call yields ``values``,
    so it can be used anywhere a plain array is expected.

    Parameters
    ----------
    values : xp.ndarray
        The data. Its rank must match the length of ``dims``.
    dims : tuple of str
        Name of each axis, e.g. ``("t", "e1", "v1")``.
    coords : dict
        Maps a dimension name to its 1D coordinate array. Dimensions may be absent,
        in which case they are indexed by position only.
    unit : str
        Physical unit of ``values``, for axis labels. Empty means arbitrary units.
    label : str
        Display name of the quantity, e.g. ``r"$f$"``.
    """

    values: xp.ndarray
    dims: tuple[str, ...]
    coords: dict[str, xp.ndarray] = field(default_factory=dict)
    unit: str = ""
    label: str = ""

    def __post_init__(self):
        self.values = xp.asarray(self.values)
        self.dims = tuple(self.dims)

        if self.values.ndim != len(self.dims):
            raise ValueError(f"values has rank {self.values.ndim} but {len(self.dims)} dims were given: {self.dims}")

        self.coords = {k: xp.asarray(v) for k, v in self.coords.items()}
        for name, c in self.coords.items():
            if name not in self.dims:
                raise ValueError(f"coord {name!r} is not one of the dims {self.dims}")
            if c.shape != (self.values.shape[self.axis(name)],):
                raise ValueError(
                    f"coord {name!r} has shape {c.shape}, expected ({self.values.shape[self.axis(name)]},)"
                )

    def __array__(self, dtype=None):
        return xp.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        """Positional indexing, returning a plain array.

        Kept so that code written against the unlabeled arrays still works; use
        :meth:`isel` or :meth:`at` to index by dimension name and keep the labels.
        """
        return self.values[key]

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def axis(self, dim: str) -> int:
        """Position of ``dim`` in ``dims``."""
        if dim not in self.dims:
            raise KeyError(f"no dim {dim!r} in {self.dims}")
        return self.dims.index(dim)

    def coord(self, dim: str) -> xp.ndarray:
        """Coordinate array of ``dim``, falling back to an integer index range."""
        if dim in self.coords:
            return self.coords[dim]
        return xp.arange(self.shape[self.axis(dim)])

    def axis_label(self, dim: str) -> str:
        """Axis label for ``dim``, including its unit where one is known."""
        base = DIM_LABELS.get(dim, dim)
        unit = self.coord_units.get(dim, "")
        return f"{base} [{unit}]" if unit else base

    @property
    def value_label(self) -> str:
        """Axis or colorbar label for the values themselves."""
        base = self.label or ""
        unit = self.unit or "a.u."
        return f"{base} [{unit}]" if base else f"[{unit}]"

    @property
    def coord_units(self) -> dict[str, str]:
        """Unit string per dimension, populated by the loader where units are known."""
        return getattr(self, "_coord_units", {})

    def with_coord_units(self, **units: str) -> "StruphyArray":
        """Attach unit strings to dimensions, for axis labels."""
        out = replace(self)
        out._coord_units = {**self.coord_units, **units}
        return out

    def isel(self, **sel: int) -> "StruphyArray":
        """Select by integer index along named dimensions.

        A dimension indexed with an ``int`` is dropped; one indexed with a ``slice``
        is kept.
        """
        idx = [slice(None)] * self.ndim
        for dim, i in sel.items():
            idx[self.axis(dim)] = i

        dropped = {dim for dim, i in sel.items() if not isinstance(i, slice)}
        new_dims = tuple(d for d in self.dims if d not in dropped)
        new_coords = {
            d: (self.coords[d][sel[d]] if d in sel else self.coords[d]) for d in self.coords if d not in dropped
        }

        out = StruphyArray(
            self.values[tuple(idx)],
            new_dims,
            new_coords,
            self.unit,
            self.label,
        )
        out._coord_units = dict(self.coord_units)
        return out

    def at(self, **sel: float) -> "StruphyArray":
        """Select the nearest coordinate value along named dimensions.

        Replaces the ``xp.abs(t_grid - t).argmin()`` idiom.
        """
        return self.isel(**{dim: int(xp.abs(self.coord(dim) - v).argmin()) for dim, v in sel.items()})

    def transpose_to(self, *dims: str) -> "StruphyArray":
        """Reorder axes to the given dimension order."""
        if set(dims) != set(self.dims):
            raise ValueError(f"cannot transpose dims {self.dims} to {dims}")

        out = StruphyArray(
            xp.transpose(self.values, [self.axis(d) for d in dims]),
            dims,
            self.coords,
            self.unit,
            self.label,
        )
        out._coord_units = dict(self.coord_units)
        return out

    def __repr__(self):
        dims = ", ".join(f"{d}: {n}" for d, n in zip(self.dims, self.shape))
        name = self.label or "StruphyArray"
        return f"<{name} ({dims}) [{self.unit or 'a.u.'}]>"


def orbit_columns(n_columns: int) -> dict:
    """Meaning of each marker-orbit column for a given saved width.

    The post-processor saves a different set of marker columns depending on the
    velocity dimension of the species, so the weight is not always at the same index.
    Positions occupy the first three columns and the marker id the last in every case.

    Parameters
    ----------
    n_columns : int
        Size of the last axis of the orbit array.

    Returns
    -------
    dict
        Maps ``"position"``, ``"velocity"``, ``"weight"`` and ``"id"`` to an index or
        slice. ``weight`` is absent when the species does not save one.
    """
    cols = {"position": slice(0, 3), "id": n_columns - 1}
    if n_columns == 8:
        cols["velocity"] = slice(3, 6)
        cols["weight"] = 6
    elif n_columns == 5:
        cols["velocity"] = 3
    else:
        cols["velocity"] = slice(3, n_columns - 1)
    return cols


def wrap_orbits(values, t_grid) -> StruphyArray:
    """Label a marker-orbit array with its ``(t, marker, attribute)`` dimensions."""
    values = xp.asarray(values)
    out = StruphyArray(
        values,
        dims=("t", "marker", "attribute"),
        coords={"t": t_grid},
        label="marker orbits",
    )
    out.columns = orbit_columns(values.shape[-1])
    return out


def wrap_field_data(data: dict, grids_log=None, *, label: str = "") -> StruphyArray:
    """Stack a time-keyed dict of evaluated field components into one labeled array.

    The post-processor stores evaluated fields as ``{time: [component, ...]}`` with each
    component on the 3D evaluation grid. This turns that into a single array with
    dimensions ``(t, comp, e1, e2, e3)``, dropping ``comp`` for scalar fields.

    Parameters
    ----------
    data : dict
        Maps time value to a list of component arrays, or to a single array.
    grids_log : list, optional
        The three 1D logical grids, attached as coordinates when given.
    """
    times = sorted(data.keys())
    if not times:
        return None

    first = data[times[0]]
    scalar = not isinstance(first, (list, tuple)) or len(first) == 1

    if scalar:
        stacked = xp.stack([xp.asarray(data[t] if not isinstance(data[t], (list, tuple)) else data[t][0]) for t in times])
        dims = ("t", "e1", "e2", "e3")
    else:
        stacked = xp.stack([xp.stack([xp.asarray(c) for c in data[t]]) for t in times])
        dims = ("t", "comp", "e1", "e2", "e3")

    coords = {"t": xp.asarray(times)}
    if grids_log is not None:
        coords.update({f"e{i + 1}": xp.asarray(g) for i, g in enumerate(grids_log)})

    # a field evaluated on a subset of directions will not match the full grid
    coords = {k: v for k, v in coords.items() if k in dims and len(v) == stacked.shape[dims.index(k)]}

    return StruphyArray(stacked, dims=dims, coords=coords, label=label)


def wrap_binned_slice(holder, slice_name: str, t_grid, coord_units: dict = None):
    """Replace the binned arrays on ``holder`` with labeled ones.

    A binned slice folder is named after the dimensions it bins, e.g. ``e1_v1_density``,
    and holds one ``grid_<dim>`` array per dimension alongside the binned quantities
    (``f_binned``, ``delta_f_binned``, ...). This pairs them up, so the binned data
    carries its coordinates and no caller has to match ``grid_e1`` to axis 0 by hand.

    Parameters
    ----------
    holder : Slice
        Container whose attributes were just populated from disk. Modified in place.
    slice_name : str
        Folder name, used to recover the dimension order.
    t_grid : array
        Time coordinate shared by every binned quantity.
    """
    grids = {k[len("grid_") :]: getattr(holder, k) for k in list(vars(holder)) if k.startswith("grid_")}
    if not grids:
        return holder

    # dimension order follows the slice name, e.g. "e1_v1_density" -> ("e1", "v1")
    dims = [part for part in slice_name.split("_") if part in grids]
    if len(dims) != len(grids):
        return holder

    coords = {d: grids[d] for d in dims}
    coords["t"] = t_grid
    units = coord_units or {}

    for name in list(vars(holder)):
        if name.startswith("grid_"):
            continue
        values = getattr(holder, name)
        if not hasattr(values, "shape"):
            continue

        expected = (len(t_grid), *(len(coords[d]) for d in dims))
        if tuple(values.shape) != expected:
            continue

        setattr(
            holder,
            name,
            StruphyArray(
                values,
                dims=("t", *dims),
                coords=coords,
                label=BINNED_LABELS.get(name, name.replace("_", " ")),
            ).with_coord_units(**units),
        )

    return holder
