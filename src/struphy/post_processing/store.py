"""The post-processed products of a run, as one self-describing netCDF file.

``post_processing/output.nc`` holds one group per species, with the products as variables of
that group's dataset::

    /em_fields                     e_field, phi, and e_field_xyz, phi_xyz with physical=True
    /kinetic_ions/orbits           x, y, z, v1, ... (one variable per quantity, see Particles.orbit_quantities)
    /kinetic_ions/e1_v1_density    f, delta_f     (dimensions t, eta1, v1)
    /kinetic_ions/view_0           n

Coordinates (time, the logical grids ``eta1``, ``eta2``, ``eta3``, and the mapped ``X``, ``Y``,
``Z``), units and labels
travel with the data, so reading needs nothing but the file. Groups are written one at a time
and read lazily, through xarray's ``h5netcdf`` engine.
"""

from __future__ import annotations

import os

import xarray as xr

STORE_NAME = "output.nc"
# 2: the logical dimensions are named eta1, eta2, eta3 (they were e1, e2, e3 in version 1)
SCHEMA_VERSION = 2
ENGINE = "h5netcdf"

# struphy's short logical labels (as in binning slice names like "e1_v1") and the dimension names
LOGICAL_DIMS = {"e1": "eta1", "e2": "eta2", "e3": "eta3"}


def store_path(path_pproc) -> str:
    """Path of the product store inside a post-processing directory."""
    return os.path.join(str(path_pproc), STORE_NAME)


def create(path, **attrs):
    """Start a new store, discarding an existing one."""
    xr.Dataset(attrs={"schema_version": SCHEMA_VERSION, **attrs}).to_netcdf(path, mode="w", engine=ENGINE)


def write_group(path, group: str, dataset: xr.Dataset):
    """Add one group to the store; its coordinates and attributes are stored with it."""
    dataset.to_netcdf(path, group=group, mode="a", engine=ENGINE)


def _rename_legacy_dims(dataset: xr.Dataset) -> xr.Dataset:
    """A version-1 group with its logical dimensions (e1, e2, e3) renamed to eta1, eta2, eta3."""
    names = {old: new for old, new in LOGICAL_DIMS.items() if old in dataset.dims or old in dataset.coords}
    return dataset.rename(names) if names else dataset


def open_tree(path) -> xr.DataTree:
    """Open the store lazily; arrays are read from disk when they are used.

    Stores written before schema version 2 name the logical dimensions e1, e2, e3; they are
    renamed to eta1, eta2, eta3 on reading, so older post-processing output keeps working.
    """
    tree = xr.open_datatree(path, engine=ENGINE)
    if int(tree.attrs.get("schema_version", 1)) < 2:
        renamed = tree.map_over_datasets(_rename_legacy_dims)
        renamed.set_close(tree.close)  # closing the renamed tree releases the file
        return renamed
    return tree
