"""Views of an :class:`~struphy.post_processing.output.Output` in the shapes of earlier versions.

They back the deprecated :meth:`Simulation.load_plotting_data`, so that code written for
``sim.orbits``, ``sim.f``, ``sim.spline_values`` and ``sim.n_sph`` keeps working. New code should use
the :class:`xarray.DataArray` products of :class:`~struphy.post_processing.output.Output` directly.
"""

from types import SimpleNamespace

import numpy as np


def _set(namespace: SimpleNamespace, path: tuple[str, ...], value):
    """Set ``namespace.<path[0]>.<path[1]>...`` to ``value``, creating the levels in between."""
    for name in path[:-1]:
        if not hasattr(namespace, name):
            setattr(namespace, name, SimpleNamespace())
        namespace = getattr(namespace, name)
    setattr(namespace, path[-1], value)


def _field_data(array) -> dict[float, list[np.ndarray]]:
    """Time -> list of components, each a 3d array, as stored for the fields of earlier versions."""
    values = np.asarray(array)
    if "component" not in array.dims:
        values = values[:, None]
    return {float(t): list(comps) for t, comps in zip(np.asarray(array["t"]), values)}


def legacy_views(output) -> SimpleNamespace:
    """The products of ``output`` as ``orbits``, ``f``, ``spline_values`` and ``n_sph``.

    * ``orbits.<species>``: array of shape ``(time, marker, quantity)``.
    * ``f.<species>.<slice>``: ``f_binned``, ``delta_f_binned`` and one ``grid_<dim>`` per bin axis.
    * ``spline_values.<species>.<name>_log`` (``_phy``): ``.data``, see :func:`_field_data`.
    * ``n_sph.<species>.<view>``: ``n_sph`` and its meshgrid ``grid_n_sph``.
    """
    views = SimpleNamespace(
        orbits=SimpleNamespace(),
        f=SimpleNamespace(),
        spline_values=SimpleNamespace(),
        n_sph=SimpleNamespace(),
    )

    for species, array in output.orbit_catalog.items():
        setattr(views.orbits, species, np.asarray(array))

    for key, array in output.field_catalog.items():
        species, name = key.split("/", 1)
        # the fields in logical coordinates were saved as ``<name>_log``, the pushed-forward ones as ``<name>_phy``
        name = f"{name.removesuffix('_xyz')}_phy" if name.endswith("_xyz") else f"{name}_log"
        _set(views.spline_values, (species, name), SimpleNamespace(data=_field_data(array)))

    for key, array in output.distribution_catalog.items():
        species, slice_name, name = key.split("/")
        _set(views.f, (species, slice_name, f"{name}_binned"), np.asarray(array))
        for dim in array.dims:
            if dim != "t":
                _set(views.f, (species, slice_name, f"grid_{dim}"), np.asarray(array[dim]))

    for key, array in output.density_catalog.items():
        species, view, _ = key.split("/")
        grid = np.meshgrid(*(np.asarray(array[dim]) for dim in ("e1", "e2", "e3")), indexing="ij")
        _set(views.n_sph, (species, view, "n_sph"), np.asarray(array))
        _set(views.n_sph, (species, view, "grid_n_sph"), tuple(grid))

    return views
