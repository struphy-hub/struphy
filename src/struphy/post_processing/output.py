"""Discover and lazily load the products of one Struphy run."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Iterator, Mapping
from functools import cached_property
from html import escape
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr
from feectools.ddm.mpi import mpi as MPI

from struphy.post_processing import store
from struphy.post_processing.arrays import BINNED_LABELS, data_array, save_scalars
from struphy.post_processing.output_accessors import OutputPlots
from struphy.post_processing.profiling import Profile
from struphy.post_processing.si import to_si

logger = logging.getLogger("struphy")


def mpi_comm_world():
    """The communicator used by output post-processing."""
    return MPI.COMM_WORLD


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

    def __init__(self, mapping: "ProductMapping | ProductCatalog", prefix: str = ""):
        self._mapping, self._prefix = mapping, prefix

    def __repr__(self):
        location = self._prefix or "products"
        products = tuple(self.catalog)
        return f"{type(self).__name__}({location!r}, products={products!r})"

    def __getattr__(self, name: str) -> Any:
        """A leaf product (``xr.DataArray``) or a nested :class:`ProductNamespace`; only known at
        run time, so callers get ``Any`` here rather than a static type a type checker cannot verify.
        """
        key = f"{self._prefix}/{name}" if self._prefix else name
        if key in self._mapping:
            return self._mapping[key]
        prefix = key + "/"
        if any(product.startswith(prefix) for product in self._mapping):
            return type(self)(self._mapping, key)
        location = self._prefix or "products"
        raise AttributeError(f"{name!r}; available names under {location!r}: {tuple(self)}")

    def __getitem__(self, key: str) -> Any:
        """A leaf product or a nested namespace, see :meth:`__getattr__`."""
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
    """A lightweight, lazy handle to the output of one Struphy simulation.

    Obtain it from :attr:`Simulation.output` (or the return value of :meth:`Simulation.run`)
    or construct ``Output(path_out)`` in a separate process. Construction reads saved metadata
    when it is available, but does not load field or particle data.

    Call :meth:`evaluate` to obtain one product as an :class:`xarray.DataArray`. It materializes
    post-processing products on demand; call :meth:`pproc` explicitly to choose its options.
    The :attr:`xarray` property exposes the complete post-processed product tree.
    Render products through this object too, e.g. ``out.viewer("em_fields/E", x="e1", y="e2")``.

    * :attr:`scalars` are read directly from the raw HDF5 output.
    * :attr:`fields`, :attr:`distributions`, :attr:`densities` and :attr:`orbits` are retained
      as compatibility views over the post-processed products.
    * :attr:`model`, :attr:`domain` and numerical options are reconstructed lazily
      from saved metadata. No simulation object is created or retained.
    * Every array carries the run in ``attrs["run"]`` (:attr:`label`) and ``attrs["run_name"]``.

    Parameters
    ----------
    path_out:
        The simulation output folder, ``sim.env.path_out``.
    time_units:
        ``"normalized"`` (the default) keeps Struphy time units, in which the analytic
        results of the models are expressed; every product then also carries seconds as the
        coordinate ``t_seconds``. ``"physical"`` makes ``t`` itself seconds.
    """

    def __init__(self, path_out, *, time_units: str = "normalized"):
        if time_units not in {"physical", "normalized"}:
            raise ValueError("time_units must be 'physical' or 'normalized'")
        self.path_out = Path(path_out).resolve()
        self.time_units = time_units
        self.comm = mpi_comm_world()
        self._reset()
        # A Simulation can expose its Output before it has written metadata. In that case,
        # keep the handle usable and let metadata raise its normal error when requested.
        try:
            self.metadata
        except FileNotFoundError:
            pass

    def __repr__(self):
        return f"{type(self).__name__}({str(self.path_out)!r}, processed={self.is_processed})"

    def with_time_units(self, time_units: str) -> "Output":
        """The same output with time coordinates in ``"physical"`` or ``"normalized"`` units."""
        return type(self)(self.path_out, time_units=time_units)

    def clear_cache(self):
        """Close lazy product files and discard loaded arrays while retaining metadata."""
        self._reset()

    close = clear_cache

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def compare(first: "Output", second: "Output", product: str, *, method: str = "linear") -> xr.Dataset:
        """Align one product from two runs and return both values, their difference and ratio."""
        left = first.evaluate(product)
        right = second.evaluate(product)
        right = right.interp_like(left, method=method)
        difference = left - right
        ratio = xr.where(right != 0, left / right, np.nan)
        return xr.Dataset({"first": left, "second": right, "difference": difference, "ratio": ratio})

    def _reset(self):
        if getattr(self, "_tree", None) is not None:
            self._tree.close()  # an open store would block the next process() from writing it
        self._time = self._grids_log = self._grids_phy = self._scalars = self._products = self._label = None
        self._profile = None
        self._tree = None
        self._species = None
        self._seconds = None

    def __getitem__(self, name: str) -> xr.DataArray:
        """Compatibility shorthand for :meth:`evaluate`."""
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

    def evaluate(
        self,
        name: str,
        *,
        sel: Mapping[str, Any] | None = None,
        isel: Mapping[str, Any] | None = None,
        method: str | None = None,
        drop: bool = False,
        as_numpy: bool = False,
        physical: Mapping[str, float] | None = None,
    ) -> xr.DataArray | np.ndarray:
        """Return a named simulation product as an :class:`xarray.DataArray`.

        Scalars are read directly from raw output. Other products are materialized with
        :meth:`pproc` on first use when no complete post-processing output exists. The returned
        array is an ordinary xarray object, so use xarray for selection, arithmetic and further
        analysis. ``isel`` selects positions (for example ``{"t": -1}``) and ``sel`` selects
        dimension-coordinate values (for example ``{"e3": 0.5}``). Positional selection is
        applied first, followed by coordinate selection. ``method`` and ``drop`` have xarray's
        usual ``.sel``/``.isel`` meanings. Set ``as_numpy=True`` to return only the selected
        values as a :class:`numpy.ndarray`.

        ``physical={"X": x, "Y": y, "Z": z}`` evaluates a field at a physical point when
        its domain supplies an analytical ``inverse_map``. It converts the point to logical
        coordinates and uses xarray interpolation.
        """
        physical_sel = None
        if physical:
            required = {"X", "Y", "Z"}
            if set(physical) != required:
                raise ValueError("physical selection requires exactly X, Y and Z")
            inverse = getattr(self.domain, "inverse_map", None)
            if inverse is None:
                raise NotImplementedError(f"{type(self.domain).__name__} has no inverse_map for physical evaluation")
            eta = inverse(*(float(physical[axis]) for axis in ("X", "Y", "Z")))
            physical_sel = dict(zip(("e1", "e2", "e3"), map(float, eta)))
        array = self[name]
        if isel:
            array = array.isel(isel, drop=drop)
        if physical_sel:
            array = array.interp(physical_sel, method=method or "linear")
        if sel:
            options = {"drop": drop}
            if method is not None:
                options["method"] = method
            array = array.sel(sel, **options)
        elif method is not None:
            raise ValueError("method requires a coordinate selection through sel")
        return array.to_numpy() if as_numpy else array

    def growth_rate(self, product: str | xr.DataArray, *, window=(None, None), amplitude: bool = False):
        """Fit exponential growth of a scalar product and return a ``FitResult``."""
        from struphy.diagnostics.analysis import GrowthFit, growth_rate

        return growth_rate(self._array(product), GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def damping_rate(self, product: str | xr.DataArray, *, window=(None, None), amplitude: bool = False):
        """Fit exponential decay to the envelope of an oscillating scalar; returns a ``FitResult``."""
        from struphy.diagnostics.analysis import GrowthFit, damping_rate

        return damping_rate(self._array(product), GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def envelope(self, product: str | xr.DataArray) -> xr.DataArray:
        """Return the local maxima of a time series, e.g. to overlay on the signal."""
        from struphy.diagnostics.analysis import envelope

        return envelope(self._array(product))

    def norm(self, product: str | xr.DataArray, *, dims=None, squared: bool = False) -> xr.DataArray:
        """Return the L2 norm over ``dims`` (default: all but ``t``), as a function of time."""
        from struphy.diagnostics.analysis import norm

        return norm(self._array(product), dims=dims, squared=squared)

    def spatial_average(self, product: str | xr.DataArray, *, dims=None) -> xr.DataArray:
        """Mean of a product over ``e1``, ``e2``, ``e3`` (or ``dims``), e.g. f(t, v1) from f(t, e1, v1)."""
        from struphy.diagnostics.analysis import spatial_average

        return spatial_average(self._array(product), dims=dims)

    def velocity_moments(self, product: str | xr.DataArray, *, dims=None) -> xr.Dataset:
        """Density, mean velocity and variance of a binned distribution, as functions of the other dimensions.

        See :func:`struphy.diagnostics.analysis.velocity_moments`.
        """
        from struphy.diagnostics.analysis import velocity_moments

        return velocity_moments(self._array(product), dims=dims)

    def with_physical_coords(self, product: str | xr.DataArray) -> xr.DataArray:
        """Attach mapped ``X``, ``Y``, ``Z`` coordinates to a product on a logical grid.

        Fields already carry them; this is for products that do not, such as binned densities.
        The coordinates are evaluated with the run's domain on the array's ``e1``, ``e2``, ``e3``
        grid; a missing logical dimension is evaluated at ``0.5``.
        """
        array = self._array(product)
        if all(name in array.coords for name in ("X", "Y", "Z")):
            return array
        dims = tuple(dim for dim in ("e1", "e2", "e3") if dim in array.dims)
        if not dims:
            raise ValueError(f"{array.name!r} has no logical dimensions e1, e2, e3; its dimensions are {array.dims}")
        missing = [dim for dim in dims if dim not in array.coords]
        if missing:
            raise ValueError(f"{array.name!r} has no coordinate values for {missing}")
        grids = [np.asarray(array.coords[dim]) if dim in dims else np.array([0.5]) for dim in ("e1", "e2", "e3")]
        mapped = self.domain(*grids)
        shape = tuple(len(grid) for grid in grids)
        for name, values in zip(("X", "Y", "Z"), mapped):
            values = np.asarray(values).reshape(shape)
            keep = tuple(slice(None) if dim in dims else 0 for dim in ("e1", "e2", "e3"))
            array = array.assign_coords({name: (dims, values[keep])})
        return array

    def drift(self, product: str | xr.DataArray, *, ref=None) -> xr.DataArray:
        """Return the deviation of a time series from a reference or its initial value."""
        from struphy.diagnostics.analysis import drift

        return drift(self._array(product), ref=ref)

    def relative_error(self, product: str | xr.DataArray, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Return the absolute relative deviation of a time series."""
        from struphy.diagnostics.analysis import relative_error

        return relative_error(self._array(product), ref=ref, skip_first=skip_first)

    def keys(self) -> tuple[str, ...]:
        """Return the names accepted by :meth:`evaluate`, without loading their arrays.

        As with evaluating a non-scalar product, this materializes default post-processing when
        needed. Call :meth:`pproc` first when its options should be chosen explicitly.
        """
        catalogs = (self.field_catalog, self.distribution_catalog, self.density_catalog, self.orbit_catalog)
        return tuple(sorted((*self.scalars.data_vars, *(key for catalog in catalogs for key in catalog))))

    def catalog(self, *, details: bool = False) -> xr.Dataset:
        """Return a structured catalog of evaluable products.

        Set ``details=True`` to include dimensions and units; this opens each product but leaves
        its numerical values lazily backed by the product store.
        """
        keys = self.keys()
        data = {
            "kind": ("product", [self._product_kind(key) for key in keys]),
            "description": ("product", [self._product_description(key) for key in keys]),
        }
        if details:
            arrays = [self.evaluate(key) for key in keys]
            data["dimensions"] = ("product", [", ".join(array.dims) for array in arrays])
            data["units"] = ("product", [str(array.attrs.get("units", "")) for array in arrays])
        return xr.Dataset(data, coords={"product": list(keys)})

    def provenance(self, product: str | None = None) -> dict:
        """Return stored post-processing provenance and current raw-output freshness."""
        from struphy.post_processing.post_processing_tools import source_fingerprint

        path = self.path_pproc / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest["current"] = manifest.get("source_fingerprint") == source_fingerprint(str(self.path_out))
        if product is not None:
            manifest["product"] = product
            manifest["available"] = product in self.keys()
        return manifest

    def _array(self, product: str | xr.DataArray) -> xr.DataArray:
        """Resolve a saved product name or accept an already-derived xarray array."""
        if isinstance(product, str):
            return self.evaluate(product)
        if isinstance(product, xr.DataArray):
            return product
        raise TypeError(f"product must be a product name or xarray.DataArray, got {type(product).__name__}")

    def timeseries(self, product: str | xr.DataArray, *others: str | xr.DataArray, **kwargs):
        """Plot one or more scalar products; see :meth:`ArrayPlots.timeseries`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        result = ArrayPlots(self._array(product)).timeseries(*(self._array(other) for other in others), **kwargs)
        return result.fig, result.ax

    def view(self, product: str | xr.DataArray, **kwargs):
        """Configure a reusable slice view of one product; see :meth:`ArrayPlots.view`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        return ArrayPlots(self._array(product)).view(**kwargs)

    def slice(self, product: str | xr.DataArray, *, ax=None, **kwargs):
        """Render one two-dimensional slice; see :meth:`ArrayPlots.slice`."""
        result = self.view(product, **kwargs).slice(ax=ax)
        return result.fig, result.ax

    def panels(self, product: str | xr.DataArray, **kwargs):
        """Render evenly spaced snapshots; see :meth:`ArrayPlots.panels`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        result = ArrayPlots(self._array(product)).panels(**kwargs)
        return result.fig, result.ax

    def viewer(self, product: str | xr.DataArray, **kwargs):
        """Create an interactive slice viewer; see :meth:`ArrayPlots.viewer`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        viewer = ArrayPlots(self._array(product)).viewer(**kwargs)
        result = viewer.draw()
        result.fig._struphy_viewer = viewer
        return result.fig, result.ax

    def animation(self, product: str | xr.DataArray, **kwargs):
        """Create a slice animation; see :meth:`ArrayPlots.animation`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        animation = ArrayPlots(self._array(product)).animation(**kwargs)
        animation._fig._struphy_animation = animation
        return animation._fig, animation._fig.axes[0]

    def frames(self, product: str | xr.DataArray, directory, **kwargs):
        """Export slice frames; see :meth:`ArrayPlots.frames`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        return ArrayPlots(self._array(product)).frames(directory, **kwargs)

    def trajectories(self, product: str | xr.DataArray, **kwargs):
        """Plot saved marker trajectories; see :meth:`ArrayPlots.trajectories`."""
        from struphy.post_processing.xarray_accessors import ArrayPlots

        result = ArrayPlots(self._array(product)).trajectories(**kwargs)
        return result.fig, result.ax

    def plot_scalars(self, names=None, *, relative_to: str | None = None, logy: bool = False):
        """Plot an overview of the scalar time series of this run."""
        result = OutputPlots(self).scalars(names=names, relative_to=relative_to, logy=logy)
        return result.fig, result.ax

    def equilibrium(self, ax=None):
        """Plot the radial equilibrium profiles saved with this run."""
        result = OutputPlots(self).equilibrium(ax=ax)
        return result.fig, result.ax

    @property
    def units(self):
        """The units of the run's normalization, in SI; see :class:`struphy.physics.physics.Units`."""
        return self.model.units

    def to_si(
        self, product: str | xr.DataArray, unit: str | float | None = None, *, label: str | None = None
    ) -> xr.DataArray:
        """A product in SI units: coordinates always, values when ``unit`` names their normalization.

        ``unit`` is one of ``x``, ``B``, ``n``, ``v``, ``t``, ``p``, ``rho``, ``j``, ``kBT``, or a
        number for a composite unit, described by ``label``. See :func:`struphy.post_processing.si.to_si`.
        """
        return to_si(self._array(product), self.units, unit, label=label)

    @property
    def profile(self) -> Profile:
        """The timing regions of this run; needs ``sim.run(profiling_activated=True)``."""
        if self._profile is None:
            path = self.path_out / "profiling_data.h5"
            if not path.is_file():
                raise FileNotFoundError(
                    f"no profiling data in {self.path_out}; run with sim.run(profiling_activated=True)"
                )
            self._profile = Profile(path, label=self.label)
        return self._profile

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
                self._seconds = float(self.model.units.t)
            except FileNotFoundError:
                self._seconds = False
        return self._seconds or None

    @property
    def path_pproc(self) -> Path:
        return self.path_out / "post_processing"

    @cached_property
    def metadata(self) -> dict:
        """Saved run metadata, with legacy ``config.json`` supported as a fallback."""
        for name in ("run_metadata.json", "config.json"):
            path = self.path_out / name
            if path.is_file():
                with path.open() as stream:
                    return json.load(stream)
        raise FileNotFoundError(f"Neither run_metadata.json nor config.json exists in {self.path_out}")

    def _restore(self, key, cls):
        # Constructors expect tuples where JSON encodes sequences as lists.
        def tuples(value):
            if isinstance(value, dict):
                return {name: tuples(item) for name, item in value.items()}
            if isinstance(value, list):
                return tuple(tuples(item) for item in value)
            return value

        value = self.metadata[key]
        return cls.from_dict(tuples(value)) if value is not None else None

    @cached_property
    def model(self):
        """Model reconstructed from its saved constructor arguments."""
        from struphy.models.base import StruphyModel

        return self._restore("model", StruphyModel)

    @cached_property
    def domain(self):
        """Computational domain of the saved run."""
        from struphy.geometry.base import Domain

        return self._restore("domain", Domain)

    @cached_property
    def equil(self):
        """Saved equilibrium, if present."""
        from struphy.fields_background.base import FluidEquilibrium

        return self._restore("equil", FluidEquilibrium)

    @cached_property
    def grid(self):
        """Saved spatial grid, if present."""
        from struphy.topology.grids import TensorProductGrid

        return self._restore("grid", TensorProductGrid)

    @cached_property
    def derham_opts(self):
        """Saved finite element options, if present."""
        from struphy.io.options import DerhamOptions

        return self._restore("derham_opts", DerhamOptions)

    @cached_property
    def time_opts(self):
        """Time-stepping options of the saved run."""
        from struphy.io.options import Time

        return self._restore("time_opts", Time)

    @cached_property
    def mpi_ranks(self) -> int:
        """Number of ranks that wrote the raw output (not the current communicator)."""
        if "mpi_ranks" in self.metadata:
            return int(self.metadata["mpi_ranks"])
        import yaml

        with (self.path_out / "meta.yml").open() as stream:
            return int(yaml.safe_load(stream)["MPI processes"])

    @property
    def is_processed(self) -> bool:
        """Whether complete post-processing of the current raw output exists."""
        from struphy.post_processing.post_processing_tools import is_processed

        return is_processed(str(self.path_out))

    def pproc(
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
        """Materialize post-processed products; reuse matching existing products.

        Call this on every MPI rank. Serial processing (the default) runs on rank 0 while
        the other ranks wait. Parallel processing reconstructs the field decomposition
        and requires the same number of ranks as the saved run.

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
            Evaluate fields on all ranks of this output's communicator.
        force:
            Reprocess even when matching products exist.

        Returns
        -------
        Output
            This run, so that ``run = Output(path).pproc(physical=True)`` reads naturally.
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
        if parallel:
            PostProcessor(self, parallel_pproc=True).process(**options)
        else:
            if self.comm.Get_rank() == 0:
                PostProcessor(self).process(**options)
            self.comm.Barrier()
        self._reset()
        return self

    def process(self, **options) -> "Output":
        """Compatibility alias for :meth:`pproc`."""
        return self.pproc(**options)

    def _ensure_processed(self):
        if self.is_processed:
            return
        if self.comm.Get_size() > 1:
            raise RuntimeError(f"{self.path_out} has no post-processed data; call out.pproc() on all ranks first")
        logger.warning(
            "\nNo post-processed data in %s, processing with default options (call out.pproc(...) to choose them)",
            self.path_out,
        )
        self.pproc()

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

        ``out.kinetic_ions.e1_v1_density.f`` and ``out.kinetic_ions.orbits`` are the
        products of that species, whatever kind they are; the grouped views :attr:`fields`,
        :attr:`distributions`, :attr:`densities` and :attr:`orbits` show them by kind.
        """
        if name.startswith("_") or name == "sim":
            raise AttributeError(name)
        attribute = getattr(type(self), name, None)
        if isinstance(attribute, cached_property):
            attribute.func(self)
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
        """Compatibility namespace for whole-run plots.

        Prefer :meth:`plot_scalars` and :meth:`equilibrium`; product plots are direct methods
        of :class:`Output`, such as :meth:`viewer` and :meth:`timeseries`.
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
        return float(self.model.units.t) if self.time_units == "physical" else 1.0

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
                self.metadata
            except FileNotFoundError:  # an output folder without its configuration
                self._label = self.path_out.name
                return self._label
            values = []
            for holder, attr, name in (
                (self.time_opts, "dt", "dt"),
                (self.time_opts, "split_algo", "algo"),
                (self.grid, "num_elements", "Nel"),
                (self.derham_opts, "degree", "p"),
            ):
                value = getattr(holder, attr, None) if holder is not None else None
                if value is not None:
                    values.append(f"{name}={value}")
            self._label = ", ".join(values) or self.path_out.name
        return self._label

    def info(self) -> str:
        """A table of every key accepted by :meth:`evaluate`, with a short description.

        Use ``print(out.info())`` interactively. As with :meth:`keys`, this materializes default
        post-processing when needed; call :meth:`pproc` first to choose its options.
        """
        rows = [(key, self._product_description(key)) for key in self.keys()]
        key_width = max((len(key) for key, _ in rows), default=3)
        lines = [
            f"Output: {self.path_out}",
            self.label,
            "",
            f"{'Key':<{key_width}}  Description",
            f"{'-' * key_width}  -----------",
        ]
        lines.extend(f"{key:<{key_width}}  {description}" for key, description in rows)
        return "\n".join(lines)

    def _product_description(self, key: str) -> str:
        """A stable description for a key, without loading its data array."""
        if key in self.scalars.data_vars:
            return f"scalar time series ({key.replace('_', ' ')})"
        if key in self.field_catalog:
            return f"field ({key.rsplit('/', 1)[-1]})"
        if key in self.distribution_catalog:
            label = BINNED_LABELS.get(key.rsplit("/", 1)[-1], key.rsplit("/", 1)[-1])
            return f"particle distribution ({label})"
        if key in self.density_catalog:
            label = BINNED_LABELS.get(key.rsplit("/", 1)[-1], key.rsplit("/", 1)[-1])
            return f"SPH density ({label})"
        return "marker trajectories"

    def _product_kind(self, key: str) -> str:
        if key in self.scalars.data_vars:
            return "scalar"
        if key in self.field_catalog:
            return "field"
        if key in self.distribution_catalog:
            return "distribution"
        if key in self.density_catalog:
            return "density"
        return "orbits"

    @staticmethod
    def _quantity_hint(key: str) -> str:
        """Static label for a catalog key, e.g. distinguishing ``f`` from ``delta_f``."""
        label = BINNED_LABELS.get(key.rsplit("/", 1)[-1])
        return f"  ({label})" if label else ""

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

    def report(self, directory=None, *, products=(), format: str = "markdown", max_scalar_rows: int = 200) -> str:
        """Write a compact, reproducible data report and return its path.

        The report records run metadata, the full product catalog, and dimensions/units of any
        explicitly requested products. ``format`` is ``"markdown"`` or ``"html"``. The full
        scalar history is written as ``scalars.csv``; HTML embeds up to ``max_scalar_rows`` rows.
        """
        if format not in {"markdown", "html"}:
            raise ValueError("format must be 'markdown' or 'html'")
        catalog = self.catalog(details=False)
        requested = [self.evaluate(key) for key in products]
        directory = Path(directory) if directory else self.path_pproc / "report"
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = save_scalars(self.scalars, str(directory / "scalars.csv"))
        rows = [
            (str(key), str(kind), str(description))
            for key, kind, description in zip(catalog.product.values, catalog.kind.values, catalog.description.values)
        ]
        scalar_names = tuple(self.scalars.data_vars)
        scalar_time = np.asarray(self.scalars.coords["t"]) if "t" in self.scalars.coords else np.empty(0)
        scalar_values = (
            np.column_stack([np.asarray(self.scalars[name]) for name in scalar_names])
            if scalar_names
            else np.empty((len(scalar_time), 0))
        )
        scalar_rows = len(scalar_time)
        indices = np.linspace(0, scalar_rows - 1, min(scalar_rows, max_scalar_rows), dtype=int) if scalar_rows else []
        summaries = []
        for name, values in zip(scalar_names, scalar_values.T):
            finite = values[np.isfinite(values)]
            summaries.append(
                (
                    name,
                    float(finite[0]) if finite.size else np.nan,
                    float(finite[-1]) if finite.size else np.nan,
                    float(finite.min()) if finite.size else np.nan,
                    float(finite.max()) if finite.size else np.nan,
                )
            )
        requested_summary = []
        for array in requested:
            values = np.asarray(array)
            finite = values[np.isfinite(values)]
            requested_summary.append(
                (
                    array.name,
                    ", ".join(array.dims),
                    str(array.attrs.get("units", "")),
                    int(values.size),
                    float(finite.min()) if finite.size else np.nan,
                    float(finite.max()) if finite.size else np.nan,
                )
            )
        if format == "markdown":
            lines = [
                f"# Struphy output report",
                "",
                f"- Path: `{self.path_out}`",
                f"- Run: {self.label}",
                "",
                "## Products",
                "",
                "| Key | Kind | Description |",
                "| --- | --- | --- |",
            ]
            lines += [f"| `{key}` | {kind} | {description} |" for key, kind, description in rows]
            lines += [
                "",
                "## Scalar summary",
                "",
                "| Scalar | Initial | Final | Min | Max |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            lines += [
                f"| `{name}` | {initial:.6g} | {final:.6g} | {minimum:.6g} | {maximum:.6g} |"
                for name, initial, final, minimum, maximum in summaries
            ]
            lines += ["", f"Full scalar values: `{Path(csv_path).name}`"]
            if requested:
                lines += [
                    "",
                    "## Requested data",
                    "",
                    "| Key | Dimensions | Units | Values | Min | Max |",
                    "| --- | --- | --- | ---: | ---: | ---: |",
                ]
                lines += [
                    f"| `{name}` | {dims} | {units} | {size} | {minimum:.6g} | {maximum:.6g} |"
                    for name, dims, units, size, minimum, maximum in requested_summary
                ]
            path = directory / "report.md"
            path.write_text("\n".join(lines) + "\n")
        else:
            product_body = "".join(
                f"<tr><td><code>{escape(key)}</code></td><td>{escape(kind)}</td><td>{escape(description)}</td></tr>"
                for key, kind, description in rows
            )
            summary_body = "".join(
                f"<tr><td><code>{escape(name)}</code></td><td>{initial:.6g}</td><td>{final:.6g}</td><td>{minimum:.6g}</td><td>{maximum:.6g}</td></tr>"
                for name, initial, final, minimum, maximum in summaries
            )
            values_body = "".join(
                f"<tr><td>{float(scalar_time[index]):.6g}</td>"
                + "".join(f"<td>{value:.6g}</td>" for value in scalar_values[index])
                + "</tr>"
                for index in indices
            )
            requested_body = "".join(
                f"<tr><td><code>{escape(str(name))}</code></td><td>{escape(dims)}</td><td>{escape(units)}</td><td>{size}</td><td>{minimum:.6g}</td><td>{maximum:.6g}</td></tr>"
                for name, dims, units, size, minimum, maximum in requested_summary
            )
            path = directory / "report.html"
            style = "body{max-width:1200px;margin:2rem auto;padding:0 1rem;background:#f7f8fa;color:#1f2937;font:15px system-ui,sans-serif}h1,h2{color:#123b5d}.cards{display:flex;gap:1rem;flex-wrap:wrap}.card{background:white;padding:1rem;border-radius:8px;box-shadow:0 1px 3px #0002;min-width:220px}table{border-collapse:collapse;width:100%;background:white;margin:1rem 0}th{background:#123b5d;color:white;text-align:left}th,td{padding:.55rem;border-bottom:1px solid #dbe1e8}tr:nth-child(even){background:#f3f6f9}code{color:#8a2558}.scroll{overflow:auto}.muted{color:#52606d}a{color:#075985}"
            path.write_text(
                f"<!doctype html><meta charset=utf-8><title>Struphy output report</title><style>{style}</style><h1>Struphy output report</h1><div class=cards><div class=card><b>Run</b><br>{escape(self.label)}</div><div class=card><b>Output directory</b><br><code>{escape(str(self.path_out))}</code></div><div class=card><b>Products</b><br>{len(rows)}</div><div class=card><b>Scalar samples</b><br>{scalar_rows}</div></div><h2>Scalar summary</h2><div class=scroll><table><tr><th>Scalar</th><th>Initial</th><th>Final</th><th>Min</th><th>Max</th></tr>{summary_body}</table></div><p class=muted>Showing {len(indices)} of {scalar_rows} rows. <a href='{Path(csv_path).name}'>Download all scalar values (CSV)</a>.</p><div class=scroll><table><tr><th>t</th>{''.join(f'<th>{escape(name)}</th>' for name in scalar_names)}</tr>{values_body}</table></div><h2>Products</h2><div class=scroll><table><tr><th>Key</th><th>Kind</th><th>Description</th></tr>{product_body}</table></div>{'<h2>Requested data</h2><div class=scroll><table><tr><th>Key</th><th>Dimensions</th><th>Units</th><th>Values</th><th>Min</th><th>Max</th></tr>' + requested_body + '</table></div>' if requested_body else ''}"
            )
        return str(path)

    @property
    def tree(self) -> xr.DataTree:
        """The product store as an :class:`xarray.DataTree`, read lazily."""
        if self._tree is None:
            self._ensure_processed()
            self._tree = store.open_tree(store.store_path(self.path_pproc))
        return self._tree

    @property
    def xarray(self) -> xr.DataTree:
        """The complete post-processed product tree as an xarray :class:`DataTree`.

        Prefer :meth:`evaluate` when requesting one named product. Accessing this property
        materializes post-processing output if it does not exist, but preserves xarray's lazy
        backing arrays.
        """
        return self.tree

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

    Saved metadata is read immediately; products are materialized and opened on demand.

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
