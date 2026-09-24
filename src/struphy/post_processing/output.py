"""Discover and lazily load the products of one Struphy run."""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from functools import cached_property
from html import escape
from pathlib import Path
from typing import Any, Literal

import h5py
import cunumpy as xp
import numpy as np
import xarray as xr
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from pyevtk.hl import gridToVTK

from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.base import Particles
from struphy.post_processing import store
from struphy.post_processing.arrays import (
    BINNED_LABELS, data_array, save_scalars, wrap_binned_data, wrap_field_data, wrap_orbits,
)
from struphy.post_processing.orbits import orbits_tools
from struphy.post_processing.manifest import MANIFEST_SCHEMA_VERSION, is_processed, normalize_options, source_fingerprint
from struphy.post_processing.profiling import Profile
from struphy.post_processing.si import to_si
from struphy.utils.progress import tqdm

logger = logging.getLogger("struphy")

# Push-forward of each de Rham space to Cartesian components, see Domain.push.
PUSH_KINDS = {"H1": "0", "Hcurl": "1", "Hdiv": "2", "L2": "3", "H1vec": "v"}
Representation = Literal["0", "1", "2", "3", "v", "norm"]


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
    Optional plotting is provided by the separate ``struphy-plots`` package.

    * :attr:`scalars` are read directly from the raw HDF5 output.
    * :attr:`fields`, :attr:`distributions`, :attr:`densities` and :attr:`orbits` are retained
      as compatibility views over the post-processed products.
    * :attr:`model`, :attr:`initial_conditions`, :attr:`domain` and numerical options are
      reconstructed lazily from saved metadata. No simulation object is created or retained.
    * Every array carries the run in ``attrs["run"]`` (:attr:`label`) and ``attrs["run_name"]``.

    Parameters
    ----------
    path_out:
        The simulation output folder, ``sim.env.path_out``.
    """

    def __init__(self, path_out):
        self.path_out = Path(path_out).resolve()
        self._time_units = "normalized"
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

    @property
    def time_units(self) -> str:
        """Time coordinates returned by this view: ``normalized`` or ``physical``."""
        return self._time_units

    def with_time_units(self, time_units: str) -> "Output":
        """Open an independent view with ``t`` in normalized units or seconds.

        Normalized arrays also carry a ``t_seconds`` coordinate. This choice changes
        only data returned by the view; saved products remain in normalized units.
        """
        if time_units not in {"physical", "normalized"}:
            raise ValueError("time_units must be 'physical' or 'normalized'")
        view = type(self)(self.path_out)
        view._time_units = time_units
        return view

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
        self._spline_derham = None
        self._spline_snapshots = {}

    def evaluate(
        self,
        name: str,
        *,
        method: str | None = None,
        drop: bool = False,
        as_numpy: bool = False,
        t: int | float | slice | Sequence[int] | None = None,
        eta1: Any | None = None,
        eta2: Any | None = None,
        eta3: Any | None = None,
        representation: Representation | None = None,
        **coordinates: Any,
    ) -> xr.DataArray | np.ndarray:
        """Return a named simulation product as an :class:`xarray.DataArray`.

        Scalars are read directly from raw output. Other products are materialized with
        :meth:`pproc` on first use when no complete post-processing output exists. The returned
        array is an ordinary xarray object, so use xarray for selection, arithmetic and further
        analysis. Set ``as_numpy=True`` to return only the selected values as a
        :class:`numpy.ndarray`.

        Common selections can be passed directly: ``t`` selects saved snapshots
        by index (an integer, list of integers, or slice); omit it for every
        saved timestep. The returned array always retains its ``t`` dimension.
        A float ``t`` selects a time coordinate. Other keyword arguments select
        named coordinates, for example ``component=2`` or ``e1=0.5``.

        Supplying all of ``eta1``, ``eta2`` and ``eta3`` instead evaluates a raw FEEC
        spline field directly on that logical grid. Each eta can be a scalar, a list,
        a one-dimensional array, or a ``range``; mixed inputs form their tensor-product
        mesh internally. This reads coefficients one saved snapshot at a time and does
        not materialize a spatial post-processing product. Use a raw field name such as
        ``"em_fields/e_field"``. ``representation`` selects the output representation
        after spline evaluation: one of ``"0"``, ``"1"``, ``"2"``, ``"3"``, ``"v"``, or
        ``"norm"``. The input representation is inferred from the field's FEEC space.
        Scalars default to ``"0"`` and vectors to ``"norm"``.
        """
        selectors = dict(coordinates)
        if "physical" in selectors:
            raise TypeError("physical is no longer supported; use eta1, eta2, eta3 and representation")

        eta = (eta1, eta2, eta3)
        has_eta = any(value is not None for value in eta)
        if has_eta:
            if any(value is None for value in eta):
                raise ValueError("eta1, eta2, and eta3 must be supplied together")
            array = self._evaluate_spline_field(name, *eta, t=t, method=method, representation=representation)
            t = None
            method = None
        if not has_eta:
            if representation is not None:
                raise ValueError("representation requires eta1, eta2, and eta3")
            array = self._product(name)
        if t is not None:
            if isinstance(t, (int, np.integer)):
                array = array.isel(t=[int(t)], drop=drop)
            elif isinstance(t, slice):
                array = array.isel(t=t, drop=drop)
            elif isinstance(t, (list, tuple, np.ndarray)):
                if not all(isinstance(index, (int, np.integer)) for index in t):
                    raise TypeError("t sequences must contain saved-snapshot indices")
                array = array.isel(t=list(t), drop=drop)
            elif isinstance(t, (float, np.floating)):
                selectors["t"] = [float(t)]
            else:
                raise TypeError("t must be a saved-snapshot index, index sequence, slice, or float time coordinate")
        if selectors:
            options = {"drop": drop}
            if method is not None:
                options["method"] = method
            array = array.sel(selectors, **options)
        elif method is not None:
            raise ValueError("method requires a direct coordinate selector")
        return array.to_numpy() if as_numpy else array

    def _evaluate_spline_field(
        self, name: str, eta1: Any, eta2: Any, eta3: Any, *, t: int | float | slice | Sequence[int] | None,
        method: str | None, representation: Representation | None,
    ) -> xr.DataArray:
        """Evaluate one raw FEEC field on a tensor-product logical grid."""
        try:
            species, variable = name.split("/")
        except ValueError as error:
            raise ValueError("raw spline fields use a 'species/variable' name") from error

        path = self.path_out / "data" / "data_proc0.hdf5"
        with h5py.File(path) as file:
            if "feec" not in file:
                raise ValueError("This output contains no saved FEEC fields")
            times = np.asarray(file["time/value"]) * self.time_scale
            indices = self._snapshot_indices(t, times, method=method)

        etas, grid_dims, grid_coords = self._logical_grid(eta1, eta2, eta3)
        grid_shape = tuple(len(grid_coords[dim]) for dim in grid_dims)

        values = []
        for snapshot in indices:
            fields = self.spline_fields(t=int(snapshot))
            try:
                field = fields[species][variable]
            except KeyError as error:
                available = tuple(f"{group}/{key}" for group, entries in fields.items() for key in entries)
                raise KeyError(f"{name!r} is not a saved raw FEEC field; available fields: {available}") from error
            value = field(*etas, squeeze_out=False)
            value = self._apply_representation(value, etas, PUSH_KINDS[field.space_id], representation)
            if isinstance(value, (list, tuple)):
                value = [self._reshape_spline_value(component, grid_shape) for component in value]
            else:
                value = self._reshape_spline_value(value, grid_shape)
            values.append(value)

        is_vector = bool(values and isinstance(values[0], list))
        data = np.asarray(values) if values else np.empty((0, *grid_shape))
        dims = ("t",) + (("component",) if is_vector else ()) + tuple(grid_dims)
        coords: dict[str, Any] = {"t": times[indices], **grid_coords}
        if is_vector:
            coords["component"] = np.arange(data.shape[1])
        return self._stamp(xr.DataArray(data, dims=dims, coords=coords, name=variable))

    def _apply_representation(
        self, value: Any, etas: tuple[Any, Any, Any], source: str, representation: Representation | None,
    ) -> Any:
        """Transform a field from its FEEC-space representation to the requested target."""
        target = representation or ("norm" if source in {"1", "2", "v"} else "0")
        if target == source:
            return value
        transformation = f"{source}_to_{target}"
        try:
            transformed = self.domain.transform(value, *etas, kind=transformation, squeeze_out=True)
        except KeyError as error:
            raise ValueError(f"cannot transform {source!r} fields to representation {target!r}") from error
        if target not in {"0", "3"} and not isinstance(transformed, (list, tuple)):
            transformed = [transformed[component] for component in range(3)]
        return transformed

    @staticmethod
    def _logical_grid(*etas: Any) -> tuple[tuple[Any, Any, Any], tuple[str, ...], dict[str, np.ndarray]]:
        """Normalize mixed logical-coordinate inputs for spline tensor-product evaluation."""
        arguments = []
        dims = []
        coords = {}
        for dimension, eta in zip(("e1", "e2", "e3"), etas):
            array = np.asarray(eta, dtype=float)
            if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(f"{dimension} values must be finite and lie in the logical unit interval [0, 1]")
            if array.ndim == 0:
                arguments.append(float(array))
            elif array.ndim == 1:
                if not array.size:
                    raise ValueError(f"{dimension} must contain at least one coordinate")
                arguments.append(xp.asarray(array))
                dims.append(dimension)
                coords[dimension] = array
            else:
                raise ValueError(f"{dimension} must be a scalar or one-dimensional coordinate sequence")
        return tuple(arguments), tuple(dims), coords

    @staticmethod
    def _reshape_spline_value(value: Any, shape: tuple[int, ...]) -> Any:
        """Convert one squeezed spline result to the requested logical-grid shape."""
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        if not shape:
            return array.item()
        return array.reshape(shape)

    @staticmethod
    def _snapshot_indices(
        selection: int | float | slice | Sequence[int] | None, times: np.ndarray, *, method: str | None,
    ) -> np.ndarray:
        """Turn the public ``t`` selector into non-negative saved-snapshot indices."""
        count = len(times)
        if selection is None:
            return np.arange(count)
        if isinstance(selection, (int, np.integer)):
            indices = np.array([int(selection)])
        elif isinstance(selection, slice):
            return np.arange(count)[selection]
        elif isinstance(selection, (list, tuple, np.ndarray)):
            if not all(isinstance(index, (int, np.integer)) for index in selection):
                raise TypeError("t sequences must contain saved-snapshot indices")
            indices = np.asarray(selection, dtype=int)
        elif isinstance(selection, (float, np.floating)):
            matches = np.flatnonzero(np.isclose(times, float(selection)))
            if matches.size:
                return matches[:1]
            if method == "nearest" and count:
                return np.array([np.abs(times - float(selection)).argmin()])
            raise KeyError(f"time coordinate {selection} is not saved")
        else:
            raise TypeError("t must be a saved-snapshot index, index sequence, slice, or float time coordinate")
        indices = np.where(indices < 0, indices + count, indices)
        if np.any((indices < 0) | (indices >= count)):
            raise IndexError("t is outside the saved snapshot range")
        return indices

    def _product(self, name: str) -> xr.DataArray:
        """Resolve one saved product for :meth:`evaluate`."""
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
        """Model reconstructed from saved metadata, including initial conditions."""
        from struphy.models.base import StruphyModel
        from struphy.models.variables import PICVariable

        model = self._restore("model", StruphyModel)
        for species_name, variables in self.initial_conditions.items():
            species = model.species.get(species_name)
            if species is None:
                continue
            for variable_name, definition in variables.items():
                variable = species.variables.get(variable_name)
                if variable is None:
                    continue
                if "backgrounds" in definition:
                    variable._backgrounds = definition["backgrounds"]
                if "perturbations" in definition:
                    variable._perturbations = definition["perturbations"]
                if isinstance(variable, PICVariable) and "initial_condition" in definition:
                    variable._initial_condition = definition["initial_condition"]
        return model

    @cached_property
    def initial_conditions(self) -> dict:
        """Initial-condition definitions reconstructed from run metadata.

        This reconstructs backgrounds, perturbations, distributions, and saved
        Python functions/classes without creating a
        :class:`~struphy.simulation.sim.Simulation` instance. Definitions that
        cannot be deserialized are omitted; their serialized form remains in
        :attr:`metadata`.
        """
        from struphy.simulation.sim import Simulation

        version = self.metadata.get("model", {}).get(
            "initial_conditions_schema_version", self.metadata.get("initial_conditions_schema_version", 1)
        )
        if version != 1:
            raise ValueError(f"Unsupported initial-conditions metadata schema version: {version}.")
        result = {}
        for species_name, variables in self._initial_condition_metadata().items():
            result[species_name] = {}
            for variable_name, definition in variables.items():
                restored = {}
                for key, value in definition.items():
                    try:
                        restored[key] = Simulation._deserialize_initial_condition(value)
                    except ValueError as error:
                        logger.warning("Skipping %s.%s %s: %s", species_name, variable_name, key, error)
                result[species_name][variable_name] = restored
        return result

    def _initial_condition_metadata(self) -> dict:
        """Read variable definitions, including the layout of older output folders."""
        legacy = self.metadata.get("initial_conditions")
        if legacy is not None:
            return legacy
        return {
            species_name: {
                name: variable["initial_conditions"]
                for name, variable in species.get("variables", {}).items()
                if "initial_conditions" in variable
            }
            for species_name, species in self.metadata.get("model", {}).get("species", {}).items()
        }

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
        options = dict(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
            force=force,
        )
        try:
            if parallel or self.comm.Get_rank() == 0:
                self._setup_processing(parallel)
                self._process_raw(**options)
            if not parallel:
                self.comm.Barrier()
        finally:
            self._reset()
            for name in (
                "_pproc_derham", "_pproc_comm", "_pproc_rank", "_pproc_ranks", "_pproc_parallel",
                "_pproc_t_grid", "_pproc_exist_fields", "_pproc_exist_particles",
                "_pproc_kinetic_species", "_pproc_kinetic_kinds", "_collect_recv_bufs",
            ):
                self.__dict__.pop(name, None)
        return self

    def _setup_processing(self, parallel: bool):
        """Prepare the communicator and FEEC reconstruction for this processing run."""
        self._pproc_parallel = parallel
        self._pproc_comm = self.comm if parallel else MockComm()
        self._pproc_rank = self._pproc_comm.Get_rank()
        if parallel and self._pproc_comm.Get_size() != self.mpi_ranks:
            raise ValueError("Parallel post-processing requires the same number of MPI ranks as the saved run.")
        self._pproc_ranks = range(self._pproc_rank, self._pproc_rank + 1) if parallel else range(self.mpi_ranks)
        self._pproc_derham = None
        if self.grid is not None and self.derham_opts is not None:
            self._pproc_derham = Derham(
                self.grid, self.derham_opts, comm=self._pproc_comm if parallel else None, domain=self.domain,
            )

    def process(self, **options) -> "Output":
        """Compatibility alias for :meth:`pproc`."""
        return self.pproc(**options)

    def _write_manifest(self, status, *, options=None, error=None):
        if self._pproc_rank != 0:
            return
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": status,
            "source_fingerprint": source_fingerprint(str(self.path_out)),
            "options": options or {},
        }
        if error is not None:
            manifest["error"] = str(error)
        if status == "complete":
            manifest["products"] = sorted(
                os.path.relpath(os.path.join(root, name), self.path_pproc)
                for root, _, files in os.walk(self.path_pproc)
                for name in files
                if name != "manifest.json"
            )
        path = os.path.join(self.path_pproc, "manifest.json")
        temporary = path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    def _reset_pproc_dir(self):
        if self._pproc_rank == 0:
            if os.path.exists(self.path_pproc):
                shutil.rmtree(self.path_pproc)
            os.mkdir(self.path_pproc)
        self._pproc_comm.Barrier()

    def _process_raw(
        self,
        step: int = 1,
        celldivide: int | Sequence[int] = (1, 1, 1),
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        force: bool = False,
    ):
        """Output post-processing for fields and particle data in ``self.path_out``.

        Parameters
        ----------
        step : int
            Interval of saved time steps to post-process (1 = every step, 2 = every second step, ...).
        celldivide : int or sequence of int
            Grid refinement factor when evaluating FEM fields (e.g. ``celldivide=(2, 2, 2)`` evaluates two
            points per cell in each logical direction). A single int is applied to all three directions.
        physical : bool
            If True, also compute push-forwarded physical (x,y,z) components of fields.
        guiding_center : bool
            If True, compute guiding-center coordinates for particle orbits (requires
            Particles6D marker data).
        classify : bool
            If True, run orbit classification (passing, trapped, lost) after computing orbits.
        create_vtk : bool
            If True, create VTK files for visualisation.
        force : bool
            Reprocess even when output already exists. Set False to reuse a previous
            run's results, so a plotting script can be re-run cheaply.

        Returns
        -------
        bool
            Whether post-processing actually ran.
        """
        options = normalize_options(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
        )
        if not force and is_processed(str(self.path_out), options):
            logger.warning(f"\nReusing existing post-processing in {self.path_pproc}")
            return False

        self._reset_pproc_dir()
        if self._pproc_rank == 0:
            store.create(store.store_path(self.path_pproc), options=json.dumps(options))
        self._pproc_comm.Barrier()
        self._write_manifest("processing", options=options)
        logger.warning(f"\nPost-processing path {self.path_out}")

        # check for fields and kinetic data in hdf5 file that need post processing
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            if self._pproc_rank == 0:
                # save time grid at which post-processing data is created
                xp.save(os.path.join(self.path_pproc, "t_grid.npy"), file["time/value"][::step].copy())
            self._pproc_t_grid = xp.asarray(file["time/value"][::step])

            if "feec" in file.keys():
                self._pproc_exist_fields = True
            else:
                self._pproc_exist_fields = False

            if "kinetic" in file.keys():
                self._pproc_exist_particles = {"markers": False, "f": False, "n_sph": False}
                self._pproc_kinetic_species = []
                self._pproc_kinetic_kinds = []
                for name in file["kinetic"].keys():
                    self._pproc_kinetic_species += [name]
                    self._pproc_kinetic_kinds += [next(iter(self.model.species[name].variables.values())).space]

                    # check for saved markers
                    if "markers" in file["kinetic"][name]:
                        self._pproc_exist_particles["markers"] = True
                    # check for saved distribution function
                    if "f" in file["kinetic"][name]:
                        self._pproc_exist_particles["f"] = True
                    # check for saved sph density
                    if "n_sph" in file["kinetic"][name]:
                        self._pproc_exist_particles["n_sph"] = True
            else:
                self._pproc_exist_particles = None

        # feec variables
        try:
            self._process_fields(step=step, celldivide=celldivide, physical=physical, create_vtk=create_vtk)
            self._process_particles(step=step, guiding_center=guiding_center, classify=classify)
        except Exception as error:
            self._write_manifest("failed", options=options, error=error)
            raise

        self._write_manifest("complete", options=options)

        return True

    def _process_fields(
        self,
        step: int = 1,
        celldivide: int | Sequence[int] = (1, 1, 1),
        physical: bool = False,
        create_vtk: bool = True,
    ):
        """Evaluate the FEEC fields of all saved time steps and write them to disk.

        The time steps are processed one after another: only the spline coefficients of a
        single snapshot are held in memory, and each rank evaluates only those points of the
        evaluation grid that lie in its own MPI domain. Arrays of the size of the global
        evaluation grid therefore only ever exist on rank 0, where they are needed for output.

        Parameters
        ----------
        step : int
            Interval of saved time steps to post-process (1 = every step, 2 = every second step, ...).
        celldivide : int or sequence of int
            Grid refinement factor when evaluating FEM fields. A single int is applied to all
            three directions.
        physical : bool
            If True, also compute push-forwarded physical (x,y,z) components of fields.
        create_vtk : bool
            If True, create VTK files for visualisation.
        """
        if not self._pproc_exist_fields:
            logger.warning("\nNo feec fields found in hdf5 file, skipping post-processing of fields.")
            return

        # one set of spline functions, re-used for every time step
        fields, t_grid = self._create_femfields(step=step)

        # evaluation grid; each rank only ever evaluates the points of its own domain
        grids_log, grid_slices = self._create_eval_grids(celldivide=celldivide)
        grids_log_loc = [grid[sl] for grid, sl in zip(grids_log, grid_slices[self._pproc_rank])]
        glob_shape = tuple(grid.size for grid in grids_log)

        # the physical grid is only needed for output, hence it is only built on rank 0
        if self._pproc_rank == 0:
            grids_phy = list(self.domain(*grids_log))
        else:
            grids_phy = None

        # point_data[species][var][t] stays an empty list on all ranks except rank 0
        point_data = {species: {name: {} for name in vars} for species, vars in fields.items()}
        point_data_phy = {species: {name: {} for name in vars} for species, vars in fields.items()}

        logger.warning("\nEvaluating fields ...")
        with ExitStack() as stack:
            # hdf5 files of the simulation ranks whose data is read by this rank
            files = [
                stack.enter_context(
                    h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r"),
                )
                for rank in self._pproc_ranks
            ]

            for n, t in enumerate(tqdm(t_grid)):
                self._load_femfields(fields, files, n, step=step)

                vals, vals_phy = self._eval_femfields(
                    fields,
                    grids_log_loc,
                    grid_slices,
                    glob_shape,
                    physical=physical,
                )

                if self._pproc_rank == 0:
                    for species, vars in vals.items():
                        for name, val in vars.items():
                            point_data[species][name][t] = val
                            point_data_phy[species][name][t] = vals_phy[species][name]

        # directory for the vtk files
        path_fields = os.path.join(self.path_pproc, "fields_data")

        if self._pproc_rank == 0:
            # one group per species in the product store, with the mapped grids as coordinates
            for species, vars in point_data.items():
                variables = {}
                for name, val in vars.items():
                    variables[name] = wrap_field_data(val, grids_log, grids_phy=grids_phy, name=name)
                    if physical:
                        variables[name + "_xyz"] = wrap_field_data(
                            point_data_phy[species][name], grids_log, grids_phy=grids_phy, name=name + "_xyz"
                        )
                store.write_group(store.store_path(self.path_pproc), f"/{species}", xr.Dataset(variables))

            if create_vtk:
                try:
                    os.mkdir(path_fields)
                except FileExistsError:
                    shutil.rmtree(path_fields)
                    os.mkdir(path_fields)
                self._create_vtk(path_fields, t_grid, grids_phy, point_data)
                if physical:
                    self._create_vtk(path_fields, t_grid, grids_phy, point_data_phy, physical=True)
        self._pproc_comm.Barrier()

    def _process_particles(
        self,
        step: int = 1,
        guiding_center: bool = False,
        classify: bool = False,
    ):

        if self._pproc_exist_particles is None:
            logger.warning("\nNo kinetic data found in hdf5 file, skipping post-processing of kinetic data.")
            return

        # directory for kinetic data
        path_kinetics = os.path.join(self.path_pproc, "kinetic_data")

        if self._pproc_rank == 0:
            try:
                os.mkdir(path_kinetics)
            except:
                shutil.rmtree(path_kinetics)
                os.mkdir(path_kinetics)
        self._pproc_comm.Barrier()

        # kinetic post-processing for each species
        for n, species in enumerate(self._pproc_kinetic_species):
            # directory for each species
            path_kinetics_species = os.path.join(path_kinetics, species)

            if self._pproc_rank == 0:
                try:
                    os.mkdir(path_kinetics_species)
                except:
                    shutil.rmtree(path_kinetics_species)
                    os.mkdir(path_kinetics_species)
            self._pproc_comm.Barrier()

            # markers
            if self._pproc_exist_particles["markers"]:
                self._post_process_markers(
                    path_kinetics_species,
                    step,
                )

                if guiding_center:
                    assert self._pproc_kinetic_kinds[n] == "Particles6D"
                    orbits_tools.post_process_orbit_guiding_center(
                        self.domain, self.equil, path_kinetics_species, species
                    )

                if classify:
                    orbits_tools.post_process_orbit_classification(path_kinetics_species, species)

            # distribution function
            if self._pproc_exist_particles["f"]:
                if self._pproc_kinetic_kinds[n] == "DeltaFParticles6D":
                    compute_bckgr = True
                else:
                    compute_bckgr = False

                self._post_process_f(
                    path_kinetics_species,
                    step,
                    compute_bckgr=compute_bckgr,
                )

            # sph density
            if self._pproc_exist_particles["n_sph"]:
                self._post_process_n_sph(
                    path_kinetics_species,
                    step,
                )

    def _create_femfields(self, step: int = 1):
        """Allocate one FEEC spline field object per saved variable.

        Only a single set of fields is allocated, no matter how many time steps are
        post-processed; the coefficients of the individual snapshots are read into it one
        after another by :meth:`_load_femfields`.

        Parameters
        ----------
        step : int
            Time-step stride when reading saved snapshots (default 1).

        Returns
        -------
        fields : dict
            Nested dictionary mapping species -> variable -> ``SplineFunction``.
        t_grid : xp.ndarray
            Array of times at which the fields were saved.
        """
        # get fields names, space IDs and time grid from 0-th rank hdf5 file
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            space_ids = {}
            logger.warning("\nReading hdf5 data of following species:")
            for species, dset in file["feec"].items():
                space_ids[species] = {}
                logger.warning(f"{species}:")
                for var, ddset in dset.items():
                    space_ids[species][var] = ddset.attrs["space_id"]
                    logger.warning(f"  {var}: {ddset}")

            t_grid = file["time/value"][::step].copy()

        # create one FemField for each variable, re-used for all snapshots
        fields = {}
        for species, vars in space_ids.items():
            fields[species] = {}
            for var, id in vars.items():
                fields[species][var] = self._pproc_derham.create_spline_function(
                    var,
                    id,
                )

        logger.warning("Creation of Struphy Fields done.")

        return fields, t_grid

    def _load_femfields(self, fields: dict, files: list, n: int, step: int = 1):
        """Read the spline coefficients of one snapshot into ``fields`` (in-place).

        Parameters
        ----------
        fields : dict
            Nested dictionary species -> variable -> ``SplineFunction``, as returned
            by :meth:`_create_femfields`.
        files : list
            Open hdf5 files, one for each simulation rank processed by this rank.
        n : int
            Index of the snapshot in the (strided) time grid.
        step : int
            Time-step stride of the saved snapshots.
        """
        for file in files:
            for species, dset in file["feec"].items():
                for var, ddset in dset.items():
                    # get global start indices, end indices and pads
                    gl_s = ddset.attrs["starts"]
                    gl_e = ddset.attrs["ends"]
                    pads = ddset.attrs["pads"]

                    assert gl_s.shape == (3,) or gl_s.shape == (3, 3)
                    assert gl_e.shape == (3,) or gl_e.shape == (3, 3)
                    assert pads.shape == (3,) or pads.shape == (3, 3)

                    vector = fields[species][var].vector

                    # scalar field
                    if gl_s.shape == (3,):
                        s1, s2, s3 = gl_s
                        e1, e2, e3 = gl_e
                        p1, p2, p3 = pads

                        vector[
                            s1 : e1 + 1,
                            s2 : e2 + 1,
                            s3 : e3 + 1,
                        ] = ddset[n * step, p1:-p1, p2:-p2, p3:-p3]

                    # vector-valued field
                    else:
                        for comp in range(3):
                            s1, s2, s3 = gl_s[comp]
                            e1, e2, e3 = gl_e[comp]
                            p1, p2, p3 = pads[comp]

                            vector[comp][
                                s1 : e1 + 1,
                                s2 : e2 + 1,
                                s3 : e3 + 1,
                            ] = ddset[str(comp + 1)][n * step, p1:-p1, p2:-p2, p3:-p3]

                    vector.update_ghost_regions()

    def _create_eval_grids(self, celldivide: int | Sequence[int] = (1, 1, 1)):
        """Build the logical evaluation grids and distribute them over the MPI ranks.

        The grid points are split among the ranks exactly as
        :meth:`~struphy.feec.psydac_derham.SplineFunction._flag_pts_not_on_proc` does,
        such that every point is evaluated by exactly one rank. This allows each rank
        to allocate only its own part of the evaluation grid.

        Parameters
        ----------
        celldivide : int or sequence of int
            Refinement factor in each logical direction; a single int is applied to all
            three directions, a sequence must have length three.

        Returns
        -------
        grids_log : list
            The three global logical 1d grids.
        grid_slices : list
            One entry per rank, holding the three slices of ``grids_log`` owned by that rank.
            The slices of all ranks tile the global grid exactly.
        """
        if isinstance(celldivide, int):
            celldivide = (celldivide,) * 3

        assert isinstance(celldivide, Sequence)
        assert len(celldivide) == 3

        num_elements = self._pproc_derham.num_elements

        grids_log = [
            xp.linspace(0.0, 1.0, num_elements_i * n_i + 1) for num_elements_i, n_i in zip(num_elements, celldivide)
        ]

        # domain decomposition of the pproc communicator (one row per rank), see Derham.domain_array
        dom_arr = self._pproc_derham.domain_array

        grid_slices = []
        for rank in range(dom_arr.shape[0]):
            slices = []
            for n, grid in enumerate(grids_log):
                left = dom_arr[rank, 3 * n + 0]
                right = dom_arr[rank, 3 * n + 1]

                # points on an interior boundary are shifted into the process to the right of it
                shifted = grid.copy()
                if left != 0.0:
                    shifted[shifted == left] += 1e-8
                if right != 1.0:
                    shifted[shifted == right] += 1e-8

                inds = xp.nonzero(xp.logical_and(shifted >= left, shifted <= right))[0]
                assert inds.size > 0, f"Rank {rank} has no evaluation point in direction {n + 1}."
                assert inds.size == inds[-1] - inds[0] + 1, "Evaluation points of a rank must be contiguous."

                slices += [slice(int(inds[0]), int(inds[-1]) + 1)]

            grid_slices += [tuple(slices)]

        # the local grids must tile the global evaluation grid exactly
        n_points = sum(
            (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) * (sl[2].stop - sl[2].start) for sl in grid_slices
        )
        assert n_points == grids_log[0].size * grids_log[1].size * grids_log[2].size, (
            "The MPI domains do not tile the evaluation grid exactly."
        )

        return grids_log, grid_slices

    def _collect_on_root(self, loc_val: xp.ndarray, grid_slices: list, glob_shape: tuple):
        """Assemble the local parts of an evaluation-grid array on rank 0.

        Only rank 0 allocates an array of the size of the global evaluation grid;
        all other ranks just send the points they own.

        Parameters
        ----------
        loc_val : xp.ndarray
            Values on the evaluation points owned by this rank.
        grid_slices : list
            Slices of the global grid owned by each rank, see :meth:`_create_eval_grids`.
        glob_shape : tuple
            Number of points of the global evaluation grid in each direction.

        Returns
        -------
        xp.ndarray or None
            The global array on rank 0, None on all other ranks.
        """
        if not self._pproc_parallel:
            return loc_val

        if self._pproc_rank == 0:
            glob_val = xp.empty(glob_shape, dtype=loc_val.dtype)
            glob_val[grid_slices[0]] = loc_val

            # cache receive buffers to avoid repeated allocations in tight loops
            if not hasattr(self, "_collect_recv_bufs"):
                self._collect_recv_bufs = {}

            for rank in range(1, len(grid_slices)):
                sl = grid_slices[rank]
                shape = tuple(sl_i.stop - sl_i.start for sl_i in sl)
                buf = self._collect_recv_bufs.get((rank, shape, loc_val.dtype))
                if buf is None:
                    buf = xp.empty(shape, dtype=loc_val.dtype)
                    self._collect_recv_bufs[(rank, shape, loc_val.dtype)] = buf
                self._pproc_comm.Recv(buf, source=rank, tag=rank)
                glob_val[sl] = buf

            return glob_val

        else:
            self._pproc_comm.Send(xp.ascontiguousarray(loc_val), dest=0, tag=self._pproc_rank)
            return None

    def _eval_femfields(
        self,
        fields: dict,
        grids_log_loc: list,
        grid_slices: list,
        glob_shape: tuple,
        *,
        physical: bool = False,
    ):
        """Evaluate the spline fields of one snapshot on the evaluation grid.

        Each rank evaluates only the grid points of its own MPI domain, the values are
        then collected on rank 0.

        Parameters
        ----------
        fields : dict
            Nested dictionary species -> var -> ``SplineFunction`` holding the coefficients
            of one snapshot, see :meth:`_load_femfields`.
        grids_log_loc : list
            The three logical 1d grids restricted to the domain of this rank.
        grid_slices : list
            Slices of the global grid owned by each rank, see :meth:`_create_eval_grids`.
        glob_shape : tuple
            Number of points of the global evaluation grid in each direction.
        physical : bool, optional
            If True, also compute the push-forwarded physical (x,y,z) components.

        Returns
        -------
        vals, vals_phy : dict
            Nested dictionaries species -> var -> list of arrays (one entry for scalar-valued
            and three entries for vector-valued spaces). The arrays are only assembled on
            rank 0, the lists stay empty on all other ranks. ``vals_phy`` holds empty lists
            if ``physical`` is False.
        """
        vals = {}
        vals_phy = {}
        for species, vars in fields.items():
            vals[species] = {}
            vals_phy[species] = {}
            for name, field in vars.items():
                assert isinstance(field, SplineFunction)

                vals[species][name] = []
                vals_phy[species][name] = []

                # evaluate the field on the grid points of this rank only
                loc_val = field(*grids_log_loc, local=True)

                if physical:
                    # push-forward
                    loc_val_phy = self.domain.push(
                        loc_val,
                        *grids_log_loc,
                        kind=PUSH_KINDS[field.space_id],
                    )

                # scalar spaces
                if isinstance(loc_val, xp.ndarray):
                    comps = [loc_val]
                    comps_phy = [loc_val_phy] if physical else []
                # vector-valued spaces
                else:
                    comps = [loc_val[j] for j in range(3)]
                    comps_phy = [loc_val_phy[j] for j in range(3)] if physical else []

                # collect the values of all ranks on rank 0
                for comp in comps:
                    glob_val = self._collect_on_root(comp, grid_slices, glob_shape)
                    if self._pproc_rank == 0:
                        vals[species][name] += [glob_val]

                for comp in comps_phy:
                    glob_val = self._collect_on_root(comp, grid_slices, glob_shape)
                    if self._pproc_rank == 0:
                        vals_phy[species][name] += [glob_val]

        return vals, vals_phy

    def _create_vtk(
        self,
        path: str,
        t_grid: xp.ndarray,
        grids_phy: list,
        point_data: dict,
        *,
        physical: bool = False,
    ):
        """Write evaluated field arrays to VTK (.vts) files for visualization.

        Parameters
        ----------
        path : str
            Directory where species subfolders and their `vtk` folders will be created.
        t_grid : xp.ndarray
            Time grid corresponding to entries in ``point_data``.
        grids_phy : list
            Physical coordinate arrays returned by :meth:`_eval_femfields`.
        point_data : dict
            Evaluated field values as returned by :meth:`_eval_femfields`.
        physical : bool, optional
            If True, writes files for push-forwarded physical components (folder suffix "_phy").
        """
        for species, vars in point_data.items():
            species_path = os.path.join(path, species, "vtk" + physical * "_phy")
            if os.path.exists(species_path):
                shutil.rmtree(species_path)
            os.makedirs(species_path)

        # time loop
        nt = max(len(t_grid) - 1, 1)
        log_nt = int(xp.log10(nt)) + 1

        logger.warning(f"\nCreating vtk in {path} ...")
        for n, t in enumerate(tqdm(t_grid)):
            point_data_n = {}

            for species, vars in point_data.items():
                species_path = os.path.join(path, species, "vtk" + physical * "_phy")
                point_data_n[species] = {}
                for name, data in vars.items():
                    points_list = data[t]

                    # scalar
                    if len(points_list) == 1:
                        point_data_n[species][name] = points_list[0]

                    # vectorpoint_data[name]
                    else:
                        for j in range(3):
                            point_data_n[species][name + f"_{j + 1}"] = points_list[j]

                gridToVTK(
                    os.path.join(species_path, "step_{0:0{1}d}".format(n, log_nt)),
                    *grids_phy,
                    pointData=point_data_n[species],
                )

    def _post_process_markers(
        self,
        path_kinetic_species: str,
        step: int = 1,
    ):
        """Compute Cartesian marker positions and write them to .npy and .txt files.

        For each saved time step this function collects marker datasets from all MPI ranks,
        reconstructs full marker arrays (positions, velocities, weights, ids), maps logical
        coordinates to physical coordinates via ``self.domain`` and writes per-step
        ``.npy`` (binary) and ``.txt`` (ASCII) files suitable for quick inspection or
        import into visualization tools.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory where results will be written.
        step : int, optional
            Time-step stride to process (default 1).
        """

        species = path_kinetic_species.split("/")[-1]
        species_obj: ParticleSpecies = self.model.particle_species[species]

        # open hdf5 files and get names and number of saved markers of kinetic species
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            # get number of time steps and markers
            nt, n_markers, n_cols = file_0["kinetic/" + species + "/markers"].shape

        # get velocity dimension from one of the variables of the species
        for _, var in species_obj.variables.items():
            assert isinstance(var, PICVariable | SPHVariable)
            cls: Particles = var.particles_class
            vdim = cls.vdim
            break

        log_nt = int(xp.log10(int(((nt - 1) / step)))) + 1

        # directory for .txt files and marker index which will be saved
        path_orbits = os.path.join(path_kinetic_species, "orbits")

        if vdim == 2:
            save_index = list(range(0, 6)) + [10] + [-1]
        elif vdim == 3:
            save_index = list(range(0, 7)) + [-1]
        else:
            save_index = list(range(0, 4)) + [-1]

        if self._pproc_rank == 0:
            try:
                os.mkdir(path_orbits)
            except:
                shutil.rmtree(path_orbits)
                os.mkdir(path_orbits)
        self._pproc_comm.Barrier()

        # temporary array, plus every step of it for the product store
        temp = xp.empty((n_markers, len(save_index)), order="C")
        orbits = []
        lost_particles_mask = xp.empty(n_markers, dtype=bool)

        logger.warning(f"Evaluation of {n_markers} marker orbits for {species}")

        # loop over time grid
        for n in tqdm(range(int((nt - 1) / step) + 1)):
            # clear buffer
            temp[:, :] = 0.0

            # create text file for this time step and this species
            file_npy = os.path.join(
                path_orbits,
                species + "_{0:0{1}d}.npy".format(n, log_nt),
            )
            file_txt = os.path.join(
                path_orbits,
                species + "_{0:0{1}d}.txt".format(n, log_nt),
            )

            for rank in self._pproc_ranks:
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    markers = file["kinetic/" + species + "/markers"]
                    ids = markers[n * step, :, -1].astype("int")
                    ids = ids[ids != -1]  # exclude holes
                    temp[ids] = markers[n * step, : ids.size, save_index]

            if self._pproc_parallel:
                if self._pproc_rank == 0:
                    self._pproc_comm.Reduce(MPI.IN_PLACE, temp, op=MPI.SUM, root=0)
                else:
                    self._pproc_comm.Reduce(temp, None, op=MPI.SUM, root=0)

            # sorting out lost particles
            ids = temp[:, -1].astype("int")
            ids_lost_particles = xp.setdiff1d(xp.arange(n_markers), ids)
            ids_removed_particles = xp.nonzero(temp[:, 0] == -1.0)[0]
            ids_lost_particles = xp.array(list(set(ids_lost_particles) | set(ids_removed_particles)), dtype=int)
            lost_particles_mask[:] = False
            lost_particles_mask[ids_lost_particles] = True

            if len(ids_lost_particles) > 0:
                # lost markers are saved as [0, ..., 0, ids]
                temp[lost_particles_mask, -1] = ids_lost_particles
                ids = xp.unique(xp.append(ids, ids_lost_particles))

            assert xp.all(sorted(ids) == xp.arange(n_markers))

            # compute physical positions (x, y, z)
            pos_phys = self.domain(xp.array(temp[~lost_particles_mask, :3]), change_out_order=True)
            temp[~lost_particles_mask, :3] = pos_phys

            if self._pproc_rank == 0:
                orbits.append(temp.copy())
                # save numpy
                xp.save(file_npy, temp)
                # move ids to first column and save txt
                temp = xp.roll(temp, 1, axis=1)
                xp.savetxt(file_txt, temp[:, (0, 1, 2, 3, -1)], fmt="%12.6f", delimiter=", ")
            self._pproc_comm.Barrier()

        if self._pproc_rank == 0:
            values = wrap_orbits(xp.stack(orbits), self._pproc_t_grid[: len(orbits)])
            store.write_group(store.store_path(self.path_pproc), f"/{species}", xr.Dataset({"orbits": values}))

    def _post_process_f(
        self,
        path_kinetic_species,
        step=1,
        compute_bckgr=False,
    ):
        """Assemble and save distribution functions from per-rank binned data.

        This reads the binned full-f and delta-f arrays produced by the simulation across
        MPI ranks, sums them to global arrays, and stores the results under
        ``<path_kinetic_species>/distribution_function/<slice>``. When ``compute_bckgr`` is
        True, an analytic kinetic background is evaluated on the same grids and added.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory.
        step : int, optional
            Time-step stride to process (default 1).
        compute_bckgr : bool, optional
            If True, add the background stored by the simulation to the binned delta f.
        """
        print(f"{self._pproc_rank} starting post-processing of distribution functions for {path_kinetic_species} ...")

        species = path_kinetic_species.split("/")[-1]

        logger.warning("Evaluation of distribution functions for " + str(species))

        # the bin centers of every slice, as saved by the simulation
        slice_grids = {}
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            for slice_name in tqdm(file_0["kinetic/" + species + "/f"]):
                dims = [part for part in slice_name.split("_")]
                centers = [grid[:] for _, grid in file_0["kinetic/" + species + "/f/" + slice_name].attrs.items()]
                slice_grids[slice_name] = dict(zip(dims, centers))
        slice_names = list(slice_grids)

        # compute distribution function
        for slice_name in tqdm(slice_names):
            logger.info(f"Processing slice {slice_name} for species {species}")
            grids = slice_grids[slice_name]

            for rank in self._pproc_ranks:
                print(f"{rank = } ----------------------------")
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    if self._pproc_parallel:
                        data = file["kinetic/" + species + "/f/" + slice_name][::step]
                        data_df = file["kinetic/" + species + "/df/" + slice_name][::step]
                    else:
                        if rank == 0:
                            data = file["kinetic/" + species + "/f/" + slice_name][::step].copy()
                            data_df = file["kinetic/" + species + "/df/" + slice_name][::step].copy()
                        else:
                            data += file["kinetic/" + species + "/f/" + slice_name][::step]
                            data_df += file["kinetic/" + species + "/df/" + slice_name][::step]

            print(f"{self._pproc_rank =} with {xp.sum(data) =} and {xp.sum(data_df) =}")

            if self._pproc_parallel:
                if self._pproc_rank == 0:
                    self._pproc_comm.Reduce(
                        MPI.IN_PLACE,
                        data,
                        op=MPI.SUM,
                        root=0,
                    )
                    self._pproc_comm.Reduce(
                        MPI.IN_PLACE,
                        data_df,
                        op=MPI.SUM,
                        root=0,
                    )
                else:
                    self._pproc_comm.Reduce(
                        data,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )
                    self._pproc_comm.Reduce(
                        data_df,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )

            print(f"{self._pproc_rank =} with {xp.sum(data) =} and {xp.sum(data_df) =}")

            print(f"{self._pproc_rank =} done.")
            if self._pproc_rank == 0:
                full_f = data
                if compute_bckgr:
                    # the background of a delta-f species is stored by the simulation on the bin centers
                    key_background = f"kinetic/{species}/f_background/{slice_name}"
                    with h5py.File(os.path.join(self.path_out, "data", "data_proc0.hdf5"), "r") as file:
                        if key_background not in file:
                            raise ValueError(
                                f"{key_background} is missing from the raw output; outputs of older versions "
                                "do not store the background of delta-f species."
                            )
                        data_bckgr = file[key_background][()]

                    # add extra axis for data_bckgr since data_df has axis for time series
                    full_f = data_df + data_bckgr[None]

                store.write_group(
                    store.store_path(self.path_pproc),
                    f"/{species}/{slice_name}",
                    self._binned_dataset(grids, {"f": full_f, "delta_f": data_df}),
                )

    def _binned_dataset(self, grids: dict, variables: dict) -> xr.Dataset:
        """One binned product per variable, with time, bin centers and mapped coordinates."""
        dims = tuple(dim for dim in grids)
        coords = {"t": self._pproc_t_grid, **grids}
        coords.update(self._mapped_coords(grids))
        return xr.Dataset(
            {name: wrap_binned_data(values, dims, coords, name=name) for name, values in variables.items()}
        )

    def _mapped_coords(self, grids: dict) -> dict:
        """``X``, ``Y``, ``Z`` on the logical directions of ``grids``, when there are two or three."""
        logical = tuple(dim for dim in grids if dim in ("e1", "e2", "e3"))
        if len(logical) not in (2, 3) or self.domain is None:
            return {}
        try:
            if len(logical) == 2:
                mesh = xp.meshgrid(*(xp.asarray(grids[dim]) for dim in logical), indexing="ij")
                arguments = {"e1": 0.5, "e2": 0.0, "e3": 0.0}
                arguments.update(dict(zip(logical, mesh)))
                mapped = self.domain(arguments["e1"], arguments["e2"], arguments["e3"], squeeze_out=True)
            else:
                mapped = self.domain(*(xp.asarray(grids[dim]) for dim in logical))
        except (TypeError, ValueError):
            logger.debug("Could not map the coordinates of %s", logical, exc_info=True)
            return {}
        return {name: (logical, xp.asarray(grid)) for name, grid in zip(("X", "Y", "Z"), mapped)}

    def _post_process_n_sph(
        self,
        path_kinetic_species,
        step=1,
    ):
        """Compute and save SPH density fields from per-rank outputs.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory where results will be written.
        step : int, optional
            Time-step stride to process (default 1).
        """
        species = path_kinetic_species.split("/")[-1]

        logger.warning("Evaluation of sph density for " + str(species))

        # the evaluation points of every view, as saved by the simulation
        view_grids = {}
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            for view in file_0["kinetic/" + species + "/n_sph"]:
                attrs = file_0["kinetic/" + species + "/n_sph/" + view].attrs
                view_grids[view] = {f"e{direction}": attrs["eta" + direction][:] for direction in ("1", "2", "3")}
        views = list(view_grids)

        # compute sph density
        for view in tqdm(views):
            for rank in self._pproc_ranks:
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    if self._pproc_parallel:
                        data = file["kinetic/" + species + "/n_sph/" + view][::step]
                    else:
                        if rank == 0:
                            data = file["kinetic/" + species + "/n_sph/" + view][::step].copy()
                        else:
                            data += file["kinetic/" + species + "/n_sph/" + view][::step]

            if self._pproc_parallel:
                if self._pproc_rank == 0:
                    self._pproc_comm.Reduce(
                        MPI.IN_PLACE,
                        data,
                        op=MPI.SUM,
                        root=0,
                    )
                else:
                    self._pproc_comm.Reduce(
                        data,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )

            if self._pproc_rank == 0:
                store.write_group(
                    store.store_path(self.path_pproc),
                    f"/{species}/{view}",
                    self._binned_dataset(view_grids[view], {"n": data}),
                )

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

    def iter_spline_coefficients(self, *, stride: int = 1, rank: int = 0):
        """Yield saved FEEC spline coefficients one snapshot at a time.

        This reads only the raw ``data_proc<rank>.hdf5`` datasets. It does not
        allocate spline functions, evaluate fields, create post-processing files,
        or load particle data. Scalar variables are arrays; vector variables are
        tuples of component arrays.

        ``rank`` selects one MPI rank's local coefficients. Global assembly remains
        part of the explicit post-processing workflow.
        """
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")

        path = self.path_out / "data" / f"data_proc{rank}.hdf5"
        with h5py.File(path) as file:
            if "feec" not in file:
                return
            times = file["time/value"]
            for snapshot in range(0, len(times), stride):
                coefficients = {}
                for species_name, species in file["feec"].items():
                    variables = {}
                    for variable_name, variable in species.items():
                        if isinstance(variable, h5py.Dataset):
                            variables[variable_name] = np.asarray(variable[snapshot])
                        else:
                            variables[variable_name] = tuple(
                                np.asarray(variable[component][snapshot]) for component in sorted(variable, key=int)
                            )
                    coefficients[species_name] = variables
                yield float(times[snapshot]) * self.time_scale, coefficients

    def spline_fields(self, *, t: int) -> dict:
        """Return FEEC ``SplineFunction`` objects loaded at saved index ``t``.

        Every requested snapshot is retained in an in-memory cache. Requesting
        an already loaded ``t`` performs no HDF5 reads; requesting a new one
        allocates and fills one additional set of spline functions without
        altering earlier snapshots. The cache therefore grows with the number
        of requested snapshots. As with :meth:`evaluate`, ``t=0`` is the first
        saved snapshot and ``t=-1`` is the last.

        The returned mapping is ``species -> variable -> SplineFunction``. It is
        intentionally separate from :meth:`evaluate`, which serves persisted
        post-processed xarray products.
        """
        if not isinstance(t, int):
            raise TypeError("t must be an integer saved-snapshot index")
        if self.grid is None or self.derham_opts is None:
            raise ValueError("Spline fields require saved grid and derham options")

        data_path = self.path_out / "data" / "data_proc0.hdf5"
        with h5py.File(data_path) as file:
            if "feec" not in file:
                raise ValueError("This output contains no saved FEEC fields")
            n_snapshots = len(file["time/value"])
            if t < 0:
                t += n_snapshots
            if not 0 <= t < n_snapshots:
                raise IndexError(f"t={t} is outside the saved snapshot range")

            if self._spline_derham is None:
                self._spline_derham = Derham(self.grid, self.derham_opts, comm=None, domain=self.domain)
            if t not in self._spline_snapshots:
                fields = {
                    species_name: {
                        variable_name: self._spline_derham.create_spline_function(
                            variable_name, variable.attrs["space_id"]
                        )
                        for variable_name, variable in species.items()
                    }
                    for species_name, species in file["feec"].items()
                }

        if t not in self._spline_snapshots:
            with ExitStack() as stack:
                files = [
                    stack.enter_context(h5py.File(self.path_out / "data" / f"data_proc{rank}.hdf5"))
                    for rank in range(self.mpi_ranks)
                ]
                self._load_femfields(fields, files, t)
            self._spline_snapshots[t] = fields
        return self._spline_snapshots[t]

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

    def info(self) -> None:
        """Print a concise run summary, configuration reference, and product catalog.

        Use ``out.info()`` interactively. The summary includes model parameters,
        species variables, propagator options, and initial-condition definitions saved in
        metadata. As with :meth:`keys`, the product catalog materializes default
        post-processing when needed; call :meth:`pproc` first to choose its options.
        """
        rows = [(key, self._product_description(key)) for key in self.keys()]
        key_width = max((len(key) for key, _ in rows), default=3)
        model = self.metadata.get("model", {})
        lines = [
            f"Output: {self.path_out}",
            self.label,
            "",
            "Configuration",
            "-------------",
            f"Model: {model.get('model', self.metadata.get('model_name', 'unknown'))}",
            f"Model parameters: {json.dumps(model.get('params', {}), sort_keys=True)}",
            "Species and variables:",
        ]
        for species_name, species in model.get("species", {}).items():
            parameters = {
                key: value
                for key, value in species.items()
                if key
                not in {"class", "variables", "loading_params", "weights_params", "boundary_params", "sorting_params", "saving_params"}
                and value is not None
            }
            lines.append(f"  {species_name} ({species.get('class', 'Species')}): {json.dumps(parameters, sort_keys=True)}")
            for variable_name, variable in species.get("variables", {}).items():
                lines.append(
                    f"    {variable_name}: {variable.get('class', 'Variable')} "
                    f"[{variable.get('space', 'unknown')}], save_data={variable.get('save_data', True)}"
                )
        lines.append("Propagator options:")
        for name, options in model.get("propagator_options", {}).items():
            lines.append(f"  {name}: {json.dumps(options, sort_keys=True)}")
        lines.append("Initial conditions:")
        for species_name, variables in self._initial_condition_metadata().items():
            for variable_name, definition in variables.items():
                parts = ", ".join(f"{key}={self._initial_condition_description(value)}" for key, value in definition.items())
                lines.append(f"  {species_name}.{variable_name}: {parts}")
        lines = [
            *lines,
            "",
            "Help",
            "----",
            "- Use out.model for the reconstructed model and its variables.",
            "- Use out.initial_conditions for reconstructed backgrounds, perturbations, and distributions.",
            "- Saved Python initial conditions are reconstructed from source; unsupported definitions remain in out.metadata.",
            "- Use out.keys(), out.fields, out.distributions, out.densities, and out.orbits to discover products.",
            "- Use out.evaluate(key) and out.pproc(...) to load and process products.",
            "",
            f"{'Key':<{key_width}}  Description",
            f"{'-' * key_width}  -----------",
        ]
        lines.extend(f"{key:<{key_width}}  {description}" for key, description in rows)
        print("\n".join(lines))

    @staticmethod
    def _initial_condition_description(value) -> str:
        """Short, source-free description of one serialized initial condition."""
        if value is None:
            return "none"
        if isinstance(value, list):
            return "[" + ", ".join(Output._initial_condition_description(item) for item in value) + "]"
        if not isinstance(value, dict):
            return repr(value)
        kind = value.get("type")
        if kind is None:
            return "mapping"
        if kind in {"python_function", "python_class"}:
            return f"{kind}({value.get('name', value.get('serialization', 'unknown'))})"
        if kind == "callable":
            return f"callable({value.get('serialization', 'unknown')})"
        return kind

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
                "# Struphy output report",
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


def open_output(path_out) -> Output:
    """Open the output folder of a finished simulation.

    Saved metadata is read immediately; products are materialized and opened on demand.

    Parameters
    ----------
    path_out:
        The simulation output folder (``sim.env.path_out`` of the run).
    """
    path = Path(path_out)
    if not (path / "data").is_dir():
        raise FileNotFoundError(f"{path.resolve()} is not a Struphy output folder (it has no data/ directory)")
    return Output(path)
