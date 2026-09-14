"""Discover and lazily load the products of one Struphy run."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from struphy.post_processing.arrays import data_array, save_scalars, wrap_binned_data, wrap_field_data, wrap_orbits

logger = logging.getLogger("struphy")


class ProductMapping(Mapping[str, xr.DataArray]):
    """A discoverable mapping whose products are loaded on first access."""

    def __init__(self, loaders: Mapping[str, Callable[[], xr.DataArray]]):
        self._loaders = dict(loaders)
        self._cache: dict[str, xr.DataArray] = {}

    def __getitem__(self, key: str) -> xr.DataArray:
        if key not in self._loaders:
            raise KeyError(f"{key!r} not found; available products: {tuple(self)}")
        if key not in self._cache:
            self._cache[key] = self._loaders[key]()
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaders)

    def __len__(self) -> int:
        return len(self._loaders)

    def clear_cache(self):
        """Drop loaded arrays while keeping product discovery information."""
        self._cache.clear()


class PlotAccessor:
    """Convenient plotting entry points bound to a run."""

    def __init__(self, run: "RunOutput"):
        self._run = run

    def timeseries(self, data, **kwargs):
        from struphy.diagnostics.plotting import plot_timeseries
        return plot_timeseries(data, run_label=self._run.label, **kwargs)

    def scalar(self, name: str, **kwargs):
        return self.timeseries(self._run.scalars[name], **kwargs)

    def slice(self, data, **kwargs):
        from struphy.diagnostics.plotting import plot_slice
        return plot_slice(data, run_label=self._run.label, **kwargs)

    def viewer(self, data, **kwargs):
        from struphy.diagnostics.plotting import InteractiveSliceViewer
        return InteractiveSliceViewer(data, run_label=self._run.label, **kwargs)


class RunOutput:
    """The self-describing, lazily loaded output of a completed simulation.

    Use :meth:`open` rather than constructing this class directly. Product names are
    discovered immediately, while their arrays are loaded only when indexed.

    Parameters
    ----------
    time_units:
        ``"physical"`` converts every time coordinate to seconds. ``"normalized"``
        consistently leaves every product in Struphy time units.
    """

    def __init__(self, path_out=None, *, sim=None, time_units: str = "physical"):
        if sim is not None:
            path_out = sim.env.path_out
        if path_out is None:
            raise ValueError("path_out or sim is required")
        if time_units not in {"physical", "normalized"}:
            raise ValueError("time_units must be 'physical' or 'normalized'")
        self.path_out = Path(path_out).resolve()
        self.path_pproc = self.path_out / "post_processing"
        if not self.path_pproc.is_dir():
            raise FileNotFoundError(f"{self.path_pproc} does not exist; run post-processing first")
        self.time_units = time_units
        self._params = self._units = self._time = self._grids_log = self._grids_phy = self._scalars = None
        self.fields = ProductMapping(self._discover_fields())
        self.distributions = ProductMapping(self._discover_binned("distribution_function"))
        self.densities = ProductMapping(self._discover_binned("n_sph"))
        self.orbits = ProductMapping(self._discover_orbits())
        self.plot = PlotAccessor(self)

    @classmethod
    def open(cls, path_out=None, *, sim=None, time_units="physical") -> "RunOutput":
        if sim is not None:
            path_out = sim.env.path_out
        if path_out is None:
            raise ValueError("path_out or sim is required")
        return cls(path_out, time_units=time_units)

    @property
    def params(self):
        if self._params is None:
            from struphy.post_processing.post_processing_tools import ParamsIn
            self._params = ParamsIn(str(self.path_out))
        return self._params

    @property
    def domain(self):
        return self.params.domain

    @property
    def units(self):
        if self._units is None:
            from struphy.physics.physics import Units
            model = self.params.model
            units = Units(model.base_units)
            bulk = model.bulk_species
            units.derive_units(velocity_scale=model.velocity_scale,
                               A_bulk=None if bulk is None else bulk.mass_number,
                               Z_bulk=None if bulk is None else bulk.charge_number)
            self._units = units
        return self._units

    @property
    def time_scale(self) -> float:
        return float(self.units.t) if self.time_units == "physical" else 1.0

    @property
    def time_unit(self) -> str:
        return "s" if self.time_units == "physical" else ""

    @property
    def time(self):
        if self._time is None:
            path = self.path_pproc / "t_grid.npy"
            self._time = np.load(path, mmap_mode="r") * self.time_scale
        return self._time

    @property
    def grids_log(self):
        if self._grids_log is None:
            with (self.path_pproc / "fields_data" / "grids_log.bin").open("rb") as stream:
                self._grids_log = pickle.load(stream)
        return self._grids_log

    @property
    def grids_phy(self):
        if self._grids_phy is None:
            with (self.path_pproc / "fields_data" / "grids_phy.bin").open("rb") as stream:
                self._grids_phy = pickle.load(stream)
        return self._grids_phy

    @property
    def scalars(self) -> xr.Dataset:
        if self._scalars is None:
            path = self.path_out / "data" / "data_proc0.hdf5"
            if not path.exists():
                self._scalars = xr.Dataset()
                return self._scalars
            with h5py.File(path) as file:
                if "scalar" not in file:
                    self._scalars = xr.Dataset()
                    return self._scalars
                time = np.asarray(file["time/value"]) * self.time_scale
                variables = {}
                for name, dataset in file["scalar"].items():
                    variables[name] = data_array(np.asarray(dataset), ("t",), {"t": time}, name=name,
                                                 label=name.replace("_", " "), coord_units={"t": self.time_unit})
                self._scalars = xr.Dataset(variables)
        return self._scalars

    @property
    def label(self) -> str:
        try:
            values = []
            for holder, attr, name in ((self.params.time_opts, "dt", "dt"),
                                       (self.params.time_opts, "split_algo", "algo"),
                                       (self.params.grid, "num_elements", "Nel"),
                                       (self.params.derham_opts, "degree", "p")):
                value = getattr(holder, attr, None) if holder is not None else None
                if value is not None:
                    values.append(f"{name}={value}")
            return ", ".join(values)
        except FileNotFoundError:
            return self.path_out.name

    def save_scalars(self, path=None, **kwargs) -> str:
        path = Path(path) if path else self.path_pproc / "scalars.csv"
        return save_scalars(self.scalars, str(path), **kwargs)

    def save_scalar_plots(self, directory=None, **kwargs):
        from struphy.diagnostics.plotting import save_all_scalars
        directory = Path(directory) if directory else self.path_pproc / "scalars"
        return save_all_scalars(self.scalars, directory, run_label=self.label, **kwargs)

    def _discover_fields(self):
        loaders = {}
        root = self.path_pproc / "fields_data"
        for path in sorted(root.glob("*/*.bin")) if root.exists() else ():
            key = f"{path.parent.name}/{path.stem}"
            loaders[key] = lambda path=path, key=key: self._load_field(path, key)
        return loaders

    def _load_field(self, path: Path, key: str):
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        try:
            physical = self.grids_phy
        except FileNotFoundError:
            physical = None
        return wrap_field_data(raw, self.grids_log, grids_phy=physical, name=key.split("/")[-1],
                               time_scale=self.time_scale, time_unit=self.time_unit)

    def _discover_binned(self, category: str):
        loaders = {}
        root = self.path_pproc / "kinetic_data"
        pattern = f"*/{category}/*/*.npy"
        for path in sorted(root.glob(pattern)) if root.exists() else ():
            if path.stem.startswith("grid_"):
                continue
            species, slice_name = path.parents[2].name, path.parent.name
            key = f"{species}/{slice_name}/{path.stem}"
            loaders[key] = lambda path=path, slice_name=slice_name: self._load_binned(path, slice_name)
        return loaders

    def _load_binned(self, path: Path, slice_name: str):
        grid_paths = sorted(path.parent.glob("grid_*.npy"))
        grids = {p.stem.removeprefix("grid_"): np.load(p, mmap_mode="r") for p in grid_paths}
        dims = tuple(part for part in slice_name.split("_") if part in grids)
        values = np.load(path, mmap_mode="r")
        expected = (len(self.time), *(len(grids[dim]) for dim in dims))
        if values.shape != expected:
            raise ValueError(f"{path} has shape {values.shape}; expected {expected} from its coordinates")
        coords = {"t": self.time, **{dim: grids[dim] for dim in dims}}
        logical_dims = tuple(dim for dim in dims if dim in {"e1", "e2", "e3"})
        if len(logical_dims) == 2:
            mesh = np.meshgrid(*(np.asarray(grids[dim]) for dim in logical_dims), indexing="ij")
            arguments = {"e1": 0.5, "e2": 0.0, "e3": 0.0}
            arguments.update(dict(zip(logical_dims, mesh)))
            try:
                physical = self.domain(arguments["e1"], arguments["e2"], arguments["e3"], squeeze_out=True)
                for coordinate, grid in zip(("X", "Y", "Z"), physical):
                    coords[coordinate] = (logical_dims, np.asarray(grid))
            except (FileNotFoundError, TypeError, ValueError):
                logger.debug("Could not attach physical coordinates to %s", path, exc_info=True)
        return wrap_binned_data(values, dims, coords, name=path.stem, time_unit=self.time_unit)

    def _discover_orbits(self):
        loaders = {}
        root = self.path_pproc / "kinetic_data"
        for directory in sorted(root.glob("*/orbits")) if root.exists() else ():
            loaders[directory.parent.name] = lambda directory=directory: self._load_orbits(directory)
        return loaders

    def _load_orbits(self, directory: Path):
        paths = sorted(directory.glob("*.npy"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        if not paths:
            raise FileNotFoundError(f"no orbit arrays in {directory}")
        values = np.stack([np.load(path, mmap_mode="r") for path in paths])
        return wrap_orbits(values, self.time[:len(paths)], time_unit=self.time_unit)
