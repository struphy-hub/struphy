"""The post-processed products of a run, as one self-describing netCDF file.

``post_processing/output.nc`` holds one group per species, with the products as variables of
that group's dataset::

    /em_fields                     e_field, phi, and e_field_xyz, phi_xyz with physical=True
    /kinetic_ions                  orbits
    /kinetic_ions/e1_v1_density    f, delta_f
    /kinetic_ions/view_0           n

Coordinates (time, the logical grids, and the mapped ``X``, ``Y``, ``Z``), units and labels
travel with the data, so reading needs nothing but the file. Groups are written one at a time
and read lazily, through xarray's ``h5netcdf`` engine.
"""

from __future__ import annotations

import os

import xarray as xr

STORE_NAME = "output.nc"
SCHEMA_VERSION = 1
ENGINE = "h5netcdf"


def store_path(path_pproc) -> str:
    """Path of the product store inside a post-processing directory."""
    return os.path.join(str(path_pproc), STORE_NAME)


def create(path, **attrs):
    """Start a new store, discarding an existing one."""
    xr.Dataset(attrs={"schema_version": SCHEMA_VERSION, **attrs}).to_netcdf(path, mode="w", engine=ENGINE)


def write_group(path, group: str, dataset: xr.Dataset):
    """Add one group to the store; its coordinates and attributes are stored with it."""
    dataset.to_netcdf(path, group=group, mode="a", engine=ENGINE)


def open_tree(path) -> xr.DataTree:
    """Open the store lazily; arrays are read from disk when they are used."""
    return xr.open_datatree(path, engine=ENGINE)
