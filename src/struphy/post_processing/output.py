"""Discover and lazily load the products of one Struphy run."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from struphy.post_processing import store
from struphy.post_processing.arrays import data_array, save_scalars
from struphy.post_processing.output_accessors import OutputPlots

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

    def __contains__(self, key: object) -> bool:
        """Check the catalog without loading a product."""
        return key in self._loaders

    def clear_cache(self):
        """Drop loaded arrays while keeping product discovery information."""
        self._cache.clear()


class ProductCatalog(Mapping[str, xr.DataArray]):
    """A relative, lazy view of a :class:`ProductMapping` subtree."""

    def __init__(self, mapping: ProductMapping, prefix: str = ""):
        self._mapping, self._prefix = mapping, prefix

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def __getitem__(self, key: str) -> xr.DataArray:
        return self._mapping[self._full_key(key)]

    def __iter__(self) -> Iterator[str]:
        prefix = f"{self._prefix}/" if self._prefix else ""
        return iter(sorted(key[len(prefix) :] for key in self._mapping if key.startswith(prefix)))

    def __len__(self) -> int:
        prefix = f"{self._prefix}/" if self._prefix else ""
        return sum(key.startswith(prefix) for key in self._mapping)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._full_key(key) in self._mapping

    def clear_cache(self):
        """Drop cached arrays in the underlying catalog."""
        self._mapping.clear_cache()


class ProductNamespace:
    """Hierarchical, discoverable attribute view over product names.

    A product named ``species/slice/value`` is exposed as
    ``namespace.species.slice.value``. Product names are simulation-dependent, so
    ``__dir__`` is populated from the on-disk catalog for interactive completion.
    Use :attr:`catalog` when generic iteration over arbitrary products is needed.
    """

    def __init__(self, mapping, prefix=""):
        self._mapping, self._prefix = mapping, prefix

    def __repr__(self):
        location = self._prefix or "products"
        products = tuple(self.catalog)
        return f"{type(self).__name__}({location!r}, products={products!r})"

    def __getattr__(self, name):
        key = f"{self._prefix}/{name}" if self._prefix else name
        if key in self._mapping:
            return self._mapping[key]
        prefix = key + "/"
        if any(product.startswith(prefix) for product in self._mapping):
            return type(self)(self._mapping, key)
        location = self._prefix or "products"
        raise AttributeError(f"{name!r}; available names under {location!r}: {tuple(self)}")

    def __getitem__(self, key):
        if "/" in key:
            key = f"{self._prefix}/{key}" if self._prefix else key
            return self._mapping[key]
        return getattr(self, key)

    def __iter__(self):
        prefix = f"{self._prefix}/" if self._prefix else ""
        children = {key[len(prefix) :].split("/", 1)[0] for key in self._mapping if key.startswith(prefix)}
        return iter(sorted(children))

    def __len__(self):
        return sum(1 for _ in self)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self))

    @property
    def catalog(self):
        """Flat lazy catalog below this namespace, with names relative to it."""
        return ProductCatalog(self._mapping, self._prefix)


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
    * Products plot themselves, e.g. ``out["en_phi"].struphy.plot.timeseries(fit=(0, 40))``;
      :attr:`plot` holds the plots that need the whole run.
    * Every array carries the run in ``attrs["run"]`` (:attr:`label`) and ``attrs["run_name"]``.

    Parameters
    ----------
    path_out:
        The simulation output folder, ``sim.env.path_out``.
    sim:
        The simulation that wrote ``path_out``, if it is at hand.
    time_units:
        ``"normalized"`` (the default) keeps Struphy time units, in which the analytic
        results of the models are expressed; every product then also carries seconds as the
        coordinate ``t_seconds``. ``"physical"`` makes ``t`` itself seconds.
    """

    def __init__(self, path_out, *, sim=None, time_units: str = "normalized"):
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
        if getattr(self, "_tree", None) is not None:
            self._tree.close()  # an open store would block the next process() from writing it
        self._time = self._grids_log = self._grids_phy = self._scalars = self._products = self._label = None
        self._tree = None
        self._species = None
        self._seconds = None

    def __getitem__(self, name: str) -> xr.DataArray:
        """Any product by name: a scalar (``"en_tot"``), a field (``"em_fields/phi_log"``), a binned
        distribution or SPH density (``"kinetic_ions/e1_v1_density/f_binned"``) or orbits (``"kinetic_ions"``).
        """
        if name in self.scalars.data_vars:
            return self.scalars[name]
        for catalog in (self.field_catalog, self.distribution_catalog, self.density_catalog, self.orbit_catalog):
            if name in catalog:
                return catalog[name]
        available = (
            *self.scalars.data_vars,
            *self.field_catalog,
            *self.distribution_catalog,
            *self.density_catalog,
            *self.orbit_catalog,
        )
        raise KeyError(f"{name!r} not found; available products: {available}")

    def _stamp(self, array: xr.DataArray) -> xr.DataArray:
        array.attrs.update(run=self.label, run_name=self.path_out.name)
        if self.time_units == "normalized" and "t" in array.dims and self.seconds_per_time is not None:
            seconds = np.asarray(array.coords["t"]) * self.seconds_per_time
            array = array.assign_coords(t_seconds=("t", seconds))
            array.coords["t_seconds"].attrs.update(long_name="$t$", units="s")
        return array

    @property
    def seconds_per_time(self) -> float | None:
        """One Struphy time unit in seconds; None when the configuration is missing."""
        if self._seconds is None:
            try:
                self._seconds = float(self.sim.model.units.t)
            except FileNotFoundError:
                self._seconds = False
        return self._seconds or None

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

        options = dict(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
            force=force,
        )
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
        logger.warning(
            "\nNo post-processed data in %s, processing with default options (call out.process(...) to choose them)",
            self.path_out,
        )
        self.process()

    def _product_mappings(self) -> dict[str, ProductMapping]:
        if self._products is None:
            self._ensure_processed()
            discovered = {kind: self._discover(kind) for kind in ("fields", "distributions", "densities", "orbits")}
            self._products = {
                kind: ProductMapping({key: (lambda load=load: self._stamp(load())) for key, load in loaders.items()})
                for kind, loaders in discovered.items()
            }
            shadowed = {key.split("/")[0] for loaders in discovered.values() for key in loaders} & set(dir(type(self)))
            if shadowed:
                logger.warning(
                    "species %s shadow attributes of Output; reach them as out.fields, "
                    "out.distributions, out.densities or out.orbits",
                    sorted(shadowed),
                )
        return self._products

    @property
    def species_catalog(self) -> ProductMapping:
        """Every product of every species, keyed ``<species>/<product>``; see :meth:`__getattr__`."""
        catalogs = (self.field_catalog, self.distribution_catalog, self.density_catalog)
        loaders = {key: (lambda catalog=catalog, key=key: catalog[key]) for catalog in catalogs for key in catalog}
        loaders.update(
            {
                f"{species}/orbits": (lambda species=species: self.orbit_catalog[species])
                for species in self.orbit_catalog
            }
        )
        return ProductMapping(loaders)

    def __getattr__(self, name: str) -> ProductNamespace:
        """Products of one species or field group, as ``out.<species>.<product>``.

        ``out.kinetic_ions.e1_v1_density.f_binned`` and ``out.kinetic_ions.orbits`` are the
        products of that species, whatever kind they are; the grouped views :attr:`fields`,
        :attr:`distributions`, :attr:`densities` and :attr:`orbits` show them by kind.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        attribute = getattr(type(self), name, None)
        if isinstance(attribute, property):
            attribute.fget(self)  # the property raised AttributeError itself; show its own error
        # the raw output names the species, so an unknown name never starts post-processing
        if name not in self._raw_species():
            raise AttributeError(f"{name!r}; available species: {tuple(sorted(self._raw_species()))}")
        return ProductNamespace(self.species_catalog, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._raw_species())

    def _raw_species(self) -> set[str]:
        """Species and field groups of this run; cheap, and never starts post-processing.

        They are named in the raw output, and in the products of a run that is already processed.
        """
        if self._species is None:
            names = set()
            path = self.path_out / "data" / "data_proc0.hdf5"
            if path.exists():
                with h5py.File(path) as file:
                    for group in ("feec", "kinetic"):
                        if group in file:
                            names.update(file[group])
            if self.is_processed:
                names.update(key.split("/")[0] for key in self.species_catalog)
            self._species = names
        return self._species

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
        """Plots of the whole run: ``out.plot.scalars()`` and ``out.plot.equilibrium()``.

        A single product plots itself, e.g. ``out.em_fields.phi_log.struphy.plot.slice(...)``.
        """
        return OutputPlots(self)

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
        """Factor from Struphy time units to :attr:`time_units`."""
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

    def _first_field(self) -> xr.Dataset | None:
        """The dataset of a field species, which carries the evaluation grids."""
        for group, dataset in self._groups().items():
            if "/" not in group and any("e1" in dataset[name].dims for name in dataset.data_vars):
                return dataset
        return None

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
        dataset = self._first_field()
        if dataset is None:
            return None
        if name == "grids_log":
            return [np.asarray(dataset[dim]) for dim in ("e1", "e2", "e3")]
        if not all(coordinate in dataset.coords for coordinate in ("X", "Y", "Z")):
            return None
        return [np.asarray(dataset[coordinate]) for coordinate in ("X", "Y", "Z")]

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
                    variables[name] = self._stamp(
                        data_array(
                            np.asarray(dataset),
                            ("t",),
                            {"t": time},
                            name=name,
                            label=name.replace("_", " "),
                            coord_units={"t": self.time_unit},
                        )
                    )
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
            for holder, attr, name in (
                (sim.time_opts, "dt", "dt"),
                (sim.time_opts, "split_algo", "algo"),
                (sim.grid, "num_elements", "Nel"),
                (sim.derham_opts, "degree", "p"),
            ):
                value = getattr(holder, attr, None) if holder is not None else None
                if value is not None:
                    values.append(f"{name}={value}")
            self._label = ", ".join(values) or self.path_out.name
        return self._label

    def info(self) -> str:
        """A table of everything this output holds, printed by ``print(out.info())``.

        Names are listed as they are reached, e.g. ``out.kinetic_ions.e1_v1_density.f_binned``
        and ``out["kinetic_ions/e1_v1_density/f_binned"]``. Nothing is loaded.
        """
        lines = [f"Output of {self.path_out}", f"  {self.label}", ""]
        scalars = tuple(self.scalars.data_vars)
        lines += ["scalars (no post-processing needed)"]
        lines += [f"  out.scalars.{name}" for name in scalars] or ["  (none)"]
        if not self.is_processed:
            lines += ["", "products (not post-processed yet; run out.process(...) to choose options)"]
            lines += [f"  out.{name}.*" for name in sorted(self._raw_species())]
            return "\n".join(lines)
        for kind, catalog in (
            ("fields", self.field_catalog),
            ("distributions", self.distribution_catalog),
            ("densities", self.density_catalog),
            ("orbits", self.orbit_catalog),
        ):
            lines += ["", kind]
            entries = [f"  out.{key.replace('/', '.')}" for key in catalog]
            if kind == "orbits":
                entries = [f"  out.{key}.orbits" for key in catalog]
            lines += entries or ["  (none)"]
        return "\n".join(lines)

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

    @property
    def tree(self) -> xr.DataTree:
        """The product store as an :class:`xarray.DataTree`, read lazily."""
        if self._tree is None:
            self._ensure_processed()
            self._tree = store.open_tree(store.store_path(self.path_pproc))
        return self._tree

    def _groups(self) -> dict[str, xr.Dataset]:
        """Every group of the store that holds products, by path without the leading slash."""
        return {path.lstrip("/"): node.ds for path, node in self.tree.subtree_with_keys if node.ds.data_vars}

    def _discover(self, kind: str) -> dict:
        """Loaders for one kind of product, keyed as ``<species>[/<slice>]/<variable>``."""
        loaders = {}
        for group, dataset in self._groups().items():
            for name in dataset.data_vars:
                if self._kind(group, name) != kind:
                    continue
                key = group if name == "orbits" else f"{group}/{name}"
                loaders[key] = lambda group=group, name=name: self._load(group, name)
        return loaders

    @staticmethod
    def _kind(group: str, name: str) -> str:
        """Which catalog a variable belongs to; the store keeps them apart by shape and name."""
        if name == "orbits":
            return "orbits"
        if "/" not in group:
            return "fields"
        return "densities" if name == "n" else "distributions"

    def _load(self, group: str, name: str) -> xr.DataArray:
        array = self.tree[group].ds[name]
        if self.time_units == "physical" and "t" in array.dims:
            array = array.assign_coords(t=array.t * self.time_scale)
            array.coords["t"].attrs["units"] = "s"
        return array

def open_output(path_out, *, time_units: str = "normalized") -> Output:
    """Open the output folder of a finished simulation.

    Nothing is allocated and no MPI is needed; products are read on first access.

    Parameters
    ----------
    path_out:
        The simulation output folder (``sim.env.path_out`` of the run).
        ``"normalized"`` (the default) or ``"physical"`` (seconds) time coordinates.
    """
    path = Path(path_out)
    if not (path / "data").is_dir():
        raise FileNotFoundError(f"{path.resolve()} is not a Struphy output folder (it has no data/ directory)")
    return Output(path, time_units=time_units)
