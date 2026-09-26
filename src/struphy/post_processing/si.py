"""Conversion of normalized products to SI units."""

from __future__ import annotations

import xarray as xr

# Attributes of :class:`struphy.physics.physics.Units` that can scale a product, with their SI unit.
UNITS = {
    "x": "m",
    "B": "T",
    "n": "m^-3",
    "v": "m/s",
    "t": "s",
    "p": "Pa",
    "rho": "kg/m^3",
    "j": "A/m^2",
    "kBT": "keV",
}

# Coordinates that are converted whenever they occur, and the unit that scales each.
COORDINATE_UNITS = {"t": "t", "X": "x", "Y": "x", "Z": "x", "v1": "v", "v2": "v", "v3": "v"}


def _unit_factor(units, name: str) -> float:
    value = getattr(units, name, None)
    if value is None:
        raise ValueError(f"this run has no unit {name!r}")
    return float(value)


def to_si(array: xr.DataArray, units, unit: str | float | None = None, *, label: str | None = None) -> xr.DataArray:
    """A copy of ``array`` in SI units.

    Coordinates are converted whenever present: time ``t`` to seconds, the mapped ``X``, ``Y``,
    ``Z`` to meters and the velocities ``v1``, ``v2``, ``v3`` to m/s, each with the matching unit
    of ``units``. Logical coordinates ``e1``, ``e2``, ``e3`` are dimensionless and unchanged.

    The values are only converted if ``unit`` is given, because a product does not record which
    unit its variable was normalized with. Pass the name of that unit, one of ``x``, ``B``, ``n``,
    ``v``, ``t``, ``p``, ``rho``, ``j`` or ``kBT``, or a number for a composite unit, together with
    ``label``, its name, e.g. ``unit=units.v * units.B, label="V/m"``.
    """
    out = array.copy()
    for name, unit_name in COORDINATE_UNITS.items():
        if name not in out.coords or out.coords[name].attrs.get("units") == UNITS[unit_name]:
            continue
        attrs = dict(out.coords[name].attrs, units=UNITS[unit_name])
        out = out.assign_coords({name: out.coords[name] * _unit_factor(units, unit_name)})
        out.coords[name].attrs.update(attrs)
    if out.coords.get("t") is not None and "t_seconds" in out.coords:
        out = out.drop_vars("t_seconds")  # the time coordinate is in seconds now

    if unit is not None:
        if out.attrs.get("units"):
            raise ValueError(f"{array.name!r} already has units {out.attrs['units']!r}")
        if isinstance(unit, str):
            if unit not in UNITS:
                raise ValueError(f"unknown unit {unit!r}; choose one of {tuple(UNITS)} or pass a number")
            factor, si_label = _unit_factor(units, unit), UNITS[unit]
        else:
            factor, si_label = float(unit), label or "SI"
        out = out * factor
        out.attrs = {**array.attrs, "units": si_label}
        out.name = array.name
    return out
