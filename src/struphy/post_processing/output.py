"""Discover and lazily load the products of one Struphy run."""

from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from struphy.post_processing.arrays import data_array, save_scalars, wrap_binned_data, wrap_field_data, wrap_orbits
from struphy.post_processing.output_accessors import OutputAnalysis, OutputPlots

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


class ProductNamespace:
    """Hierarchical, discoverable attribute view over product names.

    A product named ``species/slice/value`` is exposed as
    ``namespace.species.slice.value``. Product names are simulation-dependent, so
    ``__dir__`` is populated from the on-disk catalog for interactive completion.
    Use :attr:`catalog` when generic iteration over arbitrary products is needed.
    """

    def __init__(self, mapping, prefix=""):
        self._mapping, self._prefix = mapping, prefix

    def __getattr__(self, name):
        key = f"{self._prefix}/{name}" if self._prefix else name
        if key in self._mapping:
            return self._mapping[key]
        prefix = key + "/"
        if any(product.startswith(prefix) for product in self._mapping):
            return type(self)(self._mapping, key)
        raise AttributeError(f"{name!r}; available products: {tuple(self._mapping)}")

    def __getitem__(self, key):
        if "/" in key:
            return self._mapping[key]
        return getattr(self, key)

    def __iter__(self):
        prefix = f"{self._prefix}/" if self._prefix else ""
        children = {key[len(prefix):].split("/", 1)[0] for key in self._mapping if key.startswith(prefix)}
        return iter(sorted(children))

    def __len__(self):
        return sum(1 for _ in self)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self))

    @property
    def catalog(self):
        """Flat lazy catalog for algorithms that do not know product names."""
        return self._mapping


class FieldProducts(ProductNamespace):
    """Fields grouped as ``out.fields.<species>.<field>``."""


class DistributionProducts(ProductNamespace):
    """Binned products grouped as ``out.distributions.<species>.<slice>.<name>``."""


class DensityProducts(ProductNamespace):
    """SPH products grouped by species, slice and quantity."""


class OrbitProducts(ProductNamespace):
    """Marker trajectories grouped by species."""


class Output:
    """The output of one Struphy simulation, loaded lazily from its output folder.

    Obtain it from :attr:`Simulation.output` (or the return value of :meth:`Simulation.run`)
    or, in a separate process, from :func:`open_output`. Nothing is read at construction.

    * :attr:`scalars` are read directly from the raw HDF5 output.
    * :attr:`fields`, :attr:`distributions`, :attr:`densities` and :attr:`orbits` need
      post-processed data. When there is none, the first access processes the run with
      default options; call :meth:`process` beforehand to choose options.
    * :attr:`sim` is the :class:`~struphy.Simulation` that produced the output: the live
      object for ``sim.output``, otherwise restored from disk without allocating anything.
    * :attr:`plot` and :attr:`analysis` draw and evaluate standard diagnostics, e.g.
      ``out.plot.timeseries("en_phi", fit=(0, 40))``; ``out["en_phi"]`` looks up any product.
    * Every array carries the run in ``attrs["run"]`` (:attr:`label`) and ``attrs["run_name"]``.

    Parameters
    ----------
    path_out:
        The simulation output folder, ``sim.env.path_out``.
    sim:
        The simulation that wrote ``path_out``, if it is at hand.
    time_units:
        ``"physical"`` converts every time coordinate to seconds. ``"normalized"``
        consistently leaves every product in Struphy time units.
    """

    def __init__(self, path_out, *, sim=None, time_units: str = "physical"):
        if time_units not in {"physical", "normalized"}:
            raise ValueError("time_units must be 'physical' or 'normalized'")
        self.path_out = Path(path_out).resolve()
        self.time_units = time_units
        self._sim = sim
        self._reset()

    def __repr__(self):
        return f"{type(self).__name__}({str(self.path_out)!r}, processed={self.is_processed})"

    def with_time_units(self, time_units: str) -> "Output":
        """The same output with time coordinates in ``"physical"`` or ``"normalized"`` units."""
        return type(self)(self.path_out, sim=self._sim, time_units=time_units)

    def _reset(self):
        self._time = self._grids_log = self._grids_phy = self._scalars = self._products = self._label = None

    def __getitem__(self, name: str) -> xr.DataArray:
        """Any product by name: a scalar (``"en_tot"``), a field (``"em_fields/phi_log"``), a binned
        distribution or SPH density (``"kinetic_ions/e1_v1_density/f_binned"``) or orbits (``"kinetic_ions"``).
        """
        if name in self.scalars.data_vars:
            return self.scalars[name]
        for catalog in (self.field_catalog, self.distribution_catalog, self.density_catalog, self.orbit_catalog):
            if name in catalog:
                return catalog[name]
        available = (*self.scalars.data_vars, *self.field_catalog, *self.distribution_catalog,
                     *self.density_catalog, *self.orbit_catalog)
        raise KeyError(f"{name!r} not found; available products: {available}")

    def _stamp(self, array: xr.DataArray) -> xr.DataArray:
        array.attrs.update(run=self.label, run_name=self.path_out.name)
        return array

    @property
    def path_pproc(self) -> Path:
        return self.path_out / "post_processing"

    @property
    def sim(self):
        """The simulation that produced this output; restored from disk when not given."""
        if self._sim is None:
            from struphy.simulation.sim import Simulation

            self._sim = Simulation.from_output(self.path_out)
        return self._sim

    @property
    def is_processed(self) -> bool:
        """Whether complete post-processing of the current raw output exists."""
        from struphy.post_processing.post_processing_tools import is_processed

        return is_processed(str(self.path_out))

    def process(
        self,
        *,
        step: int = 1,
        celldivide: int | tuple[int, int, int] = 1,
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = False,
        parallel: bool = False,
        force: bool = False,
    ) -> "Output":
        """Post-process the raw output; reuses existing products made with the same options.

        Call this on every MPI rank. Serial processing (the default) runs on rank 0 while
        the other ranks wait; ``parallel=True`` needs the allocated simulation that ran.

        Parameters
        ----------
        step:
            Interval of saved time steps to post-process (1 = every step, 2 = every second step, ...).
        celldivide:
            Evaluation points per cell of FEEC fields, per logical direction or for all three.
        physical:
            Also compute push-forwarded Cartesian components of fields (``*_phy`` products).
        guiding_center:
            Compute guiding-center coordinates for particle orbits (Particles6D only).
        classify:
            Classify orbits (passing, trapped, lost); requires ``guiding_center``.
        create_vtk:
            Also write VTK files of the fields.
        parallel:
            Evaluate fields on all MPI ranks of the simulation's communicator.
        force:
            Reprocess even when matching products exist.

        Returns
        -------
        Output
            This run, so that ``run = open_output(path).process(physical=True)`` reads naturally.
        """
        from struphy.post_processing.post_processing_tools import PostProcessor

        options = dict(step=step, celldivide=celldivide, physical=physical, guiding_center=guiding_center,
                       classify=classify, create_vtk=create_vtk, force=force)
        sim = self.sim
        if parallel:
            PostProcessor(sim, parallel_pproc=True).process(**options)
        else:
            if sim.rank == 0:
                PostProcessor(sim).process(**options)
            sim.Barrier()
        self._reset()
        return self

    def _ensure_processed(self):
        if self.is_processed:
            return
        if self.sim.comm_size > 1:
            raise RuntimeError(f"{self.path_out} has no post-processed data; call out.process() on all ranks first")
        logger.warning("\nNo post-processed data in %s, processing with default options "
                       "(call out.process(...) to choose them)", self.path_out)
        self.process()

    def _product_mappings(self) -> dict[str, ProductMapping]:
        if self._products is None:
            self._ensure_processed()
            discovered = {
                "fields": self._discover_fields(),
                "distributions": self._discover_binned("distribution_function"),
                "densities": self._discover_binned("n_sph"),
                "orbits": self._discover_orbits(),
            }
            self._products = {
                kind: ProductMapping({key: (lambda load=load: self._stamp(load())) for key, load in loaders.items()})
                for kind, loaders in discovered.items()
            }
        return self._products

    @property
    def fields(self) -> FieldProducts:
        """FEEC fields as ``out.fields.<species>.<field>``."""
        return FieldProducts(self.field_catalog)

    @property
    def distributions(self) -> DistributionProducts:
        """Binned distribution functions as ``out.distributions.<species>.<slice>.<name>``."""
        return DistributionProducts(self.distribution_catalog)

    @property
    def densities(self) -> DensityProducts:
        """SPH densities as ``out.densities.<species>.<slice>.<name>``."""
        return DensityProducts(self.density_catalog)

    @property
    def orbits(self) -> OrbitProducts:
        """Marker trajectories as ``out.orbits.<species>``."""
        return OrbitProducts(self.orbit_catalog)

    @property
    def field_catalog(self) -> ProductMapping:
        return self._product_mappings()["fields"]

    @property
    def distribution_catalog(self) -> ProductMapping:
        return self._product_mappings()["distributions"]

    @property
    def density_catalog(self) -> ProductMapping:
        return self._product_mappings()["densities"]

    @property
    def orbit_catalog(self) -> ProductMapping:
        return self._product_mappings()["orbits"]

    @property
    def plot(self) -> OutputPlots:
        """Standard plots, e.g. ``out.plot.scalars()`` or ``out.plot.panels(name, x="e1", y="v1")``."""
        return OutputPlots(self)

    @property
    def analysis(self) -> OutputAnalysis:
        """Quantitative diagnostics, e.g. ``out.analysis.growth_rate("en_phi", window=(0, 40))``."""
        return OutputAnalysis(self)

    @property
    def f(self) -> DistributionProducts:
        """Deprecated alias of :attr:`distributions`."""
        warnings.warn("Output.f is deprecated; use out.distributions instead.", DeprecationWarning, stacklevel=2)
        return self.distributions

    @property
    def spline_values(self) -> FieldProducts:
        """Deprecated alias of :attr:`fields`."""
        warnings.warn("Output.spline_values is deprecated; use out.fields instead.", DeprecationWarning, stacklevel=2)
        return self.fields

    @property
    def n_sph(self) -> DensityProducts:
        """Deprecated alias of :attr:`densities`."""
        warnings.warn("Output.n_sph is deprecated; use out.densities instead.", DeprecationWarning, stacklevel=2)
        return self.densities

    @property
    def t_grid(self):
        """Deprecated alias of :attr:`time`."""
        warnings.warn("Output.t_grid is deprecated; use out.time instead.", DeprecationWarning, stacklevel=2)
        return self.time

    @property
    def time_scale(self) -> float:
        return float(self.sim.model.units.t) if self.time_units == "physical" else 1.0

    @property
    def time_unit(self) -> str:
        return "s" if self.time_units == "physical" else ""

    @property
    def time(self):
        """Time grid of the post-processed products."""
        if self._time is None:
            self._ensure_processed()
            self._time = np.load(self.path_pproc / "t_grid.npy", mmap_mode="r") * self.time_scale
        return self._time

    @property
    def grids_log(self):
        """Logical evaluation grids of the fields; None for a run without FEEC fields."""
        if self._grids_log is None:
            self._grids_log = self._load_grids("grids_log")
        return self._grids_log

    @property
    def grids_phy(self):
        """Mapped evaluation grids of the fields; None for a run without FEEC fields."""
        if self._grids_phy is None:
            self._grids_phy = self._load_grids("grids_phy")
        return self._grids_phy

    def _load_grids(self, name):
        self._ensure_processed()
        path = self.path_pproc / "fields_data" / f"{name}.bin"
        if not path.exists():
            return None
        with path.open("rb") as stream:
            return pickle.load(stream)

    @property
    def scalars(self) -> xr.Dataset:
        """Scalar time series, read from the raw output; needs no post-processing."""
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
                    variables[name] = self._stamp(data_array(np.asarray(dataset), ("t",), {"t": time}, name=name,
                                                             label=name.replace("_", " "),
                                                             coord_units={"t": self.time_unit}))
                self._scalars = xr.Dataset(variables)
        return self._scalars

    @property
    def label(self) -> str:
        """Short description of the numerical parameters, for figure titles."""
        if self._label is None:
            try:
                sim = self.sim
            except FileNotFoundError:  # an output folder without its configuration
                self._label = self.path_out.name
                return self._label
            values = []
            for holder, attr, name in ((sim.time_opts, "dt", "dt"),
                                       (sim.time_opts, "split_algo", "algo"),
                                       (sim.grid, "num_elements", "Nel"),
                                       (sim.derham_opts, "degree", "p")):
                value = getattr(holder, attr, None) if holder is not None else None
                if value is not None:
                    values.append(f"{name}={value}")
            self._label = ", ".join(values) or self.path_out.name
        return self._label

    def save_scalars(self, path=None, **kwargs) -> str:
        """Write the scalar time series as CSV (or NPZ); ``post_processing/scalars.csv`` by default."""
        path = Path(path) if path else self.path_pproc / "scalars.csv"
        return save_scalars(self.scalars, str(path), **kwargs)

    def save_report(self, directory=None, **kwargs) -> list[str]:
        """Write the standard report: a scalar table, the scalar overview and one figure per scalar.

        Files go to ``post_processing/report/`` by default; returns their paths.
        """
        from struphy.diagnostics.plotting import save_all_scalars

        directory = Path(directory) if directory else self.path_pproc / "report"
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
        # binned slices are named after their dimensions (e1_v1_density); SPH views (view_0) are not
        dims = tuple(part for part in slice_name.split("_") if part in grids) or tuple(sorted(grids))
        values = np.load(path, mmap_mode="r")
        expected = (len(self.time), *(len(grids[dim]) for dim in dims))
        if values.shape != expected:
            raise ValueError(f"{path} has shape {values.shape}; expected {expected} from its coordinates")
        coords = {"t": self.time, **{dim: grids[dim] for dim in dims}}
        logical_dims = tuple(dim for dim in dims if dim in {"e1", "e2", "e3"})
        try:
            if len(logical_dims) == 2:
                mesh = np.meshgrid(*(np.asarray(grids[dim]) for dim in logical_dims), indexing="ij")
                arguments = {"e1": 0.5, "e2": 0.0, "e3": 0.0}
                arguments.update(dict(zip(logical_dims, mesh)))
                physical = self.sim.domain(arguments["e1"], arguments["e2"], arguments["e3"], squeeze_out=True)
            elif len(logical_dims) == 3:
                physical = self.sim.domain(*(np.asarray(grids[dim]) for dim in ("e1", "e2", "e3")))
            else:
                physical = ()
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
        # one small file per saved step: read them instead of keeping thousands of memory maps open
        values = np.stack([np.load(path) for path in paths])
        return wrap_orbits(values, self.time[:len(paths)], time_unit=self.time_unit)


def open_output(path_out, *, time_units: str = "physical") -> Output:
    """Open the output folder of a finished simulation.

    Nothing is allocated and no MPI is needed; products are read on first access.

    Parameters
    ----------
    path_out:
        The simulation output folder (``sim.env.path_out`` of the run).
    time_units:
        ``"physical"`` (seconds) or ``"normalized"`` time coordinates.
    """
    path = Path(path_out)
    if not (path / "data").is_dir():
        raise FileNotFoundError(f"{path.resolve()} is not a Struphy output folder (it has no data/ directory)")
    return Output(path, time_units=time_units)
