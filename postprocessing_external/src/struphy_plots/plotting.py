"""Small, composable plotting functions for labeled Struphy output.

They remain importable for plotting arbitrary labeled arrays. The optional xarray
accessor exposes them as ``array.struphy.plot.*``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Slider

from .analysis import (
    FitResult,
    GrowthFit,
    drift,
    growth_rate,
    relative_error,
)
from .arrays import (
    SCALARS_EXCLUDE,
    axis_label,
    save_scalars,
    scalar_names,
    validate_array,
    value_label,
)

logger = logging.getLogger("struphy")

STRUPHY_STYLE = {
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": "medium",
    "legend.frameon": False,
    "image.cmap": "viridis",
}

PLANES = {
    "XY": ("X", "Y", "X", "Y"),
    "XZ": ("X", "Z", "X", "Z"),
    "YZ": ("Y", "Z", "Y", "Z"),
    "RZ": ("R", "Z", "R", "Z"),
}


@dataclass(frozen=True)
class View:
    """A reusable selection and rendering recipe for an N-dimensional product."""

    x: str | None = None
    y: str | None = None
    sweep: str = "t"
    select: dict[str, float] = field(default_factory=dict)
    isel: dict[str, int] = field(default_factory=dict)
    coordinates: Literal["logical", "physical"] = "logical"
    plane: Literal["XY", "XZ", "YZ", "RZ"] = "XY"


@dataclass
class PlotResult:
    """Already-rendered Matplotlib objects; saving never redraws them.

    As the last expression of a notebook cell it displays its figure once; there is no need
    to write ``.fig``.
    """

    fig: object
    ax: object
    artists: list = field(default_factory=list)
    fit_results: list[FitResult | None] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    _shown: bool = field(default=False, init=False, repr=False, compare=False)

    def save(self, path, *, close=False, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        if close:
            plt.close(self.fig)
        return str(path)

    def show(self):
        plt.show()
        self._shown = True
        return self

    def __repr__(self):
        return f"{type(self).__name__}(fig={self.fig!r})"

    def _ipython_display_(self):
        if not self._shown:
            _display_figure(self.fig)


def _detach_figure(fig):
    """Take a figure out of pyplot under the inline backend, which would show it as a still image."""
    import matplotlib

    if "inline" in matplotlib.get_backend():
        plt.close(fig)


def _display_figure(fig):
    """Display a figure as a notebook cell result, exactly once.

    The inline backend shows every open figure again at the end of the cell, so the displayed
    figure is closed. Interactive backends (e.g. ipympl) already show the figure when it is
    created, so nothing is displayed twice there either.
    """
    import matplotlib

    if "inline" not in matplotlib.get_backend():
        return
    from IPython.display import display

    display(fig)
    plt.close(fig)


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


def _items(data):
    return [data] if isinstance(data, (xr.DataArray, xr.Dataset)) else list(data)


def shared_run_label(data, default="") -> str:
    """The run description shared by all arrays (``attrs["run"]``), or ``default``.

    Arrays loaded from a :class:`~struphy.Output` carry it; arrays from different runs share none.
    """
    runs = {item.attrs.get("run") for item in _items(data)}
    if len(runs - {None, ""}) > 1:
        return ""
    runs.discard(None)
    runs.discard("")
    return runs.pop() if runs else default


def _finish(fig, *, run_label="", tight=True):
    if run_label:
        fig.suptitle(run_label, fontsize="small")
    if tight:
        fig.tight_layout()


def _select(data: xr.DataArray, view: View, *, keep_sweep=True):
    validate_array(data)
    overlap = set(view.select) & set(view.isel)
    if overlap:
        raise ValueError(f"dimensions cannot appear in both select and isel: {sorted(overlap)}")
    selected = data
    if view.select:
        selected = selected.sel(view.select, method="nearest")
    if view.isel:
        selected = selected.isel(view.isel)
    if not keep_sweep and view.sweep in selected.dims:
        selected = selected.isel({view.sweep: 0})
    return selected


def logical_grids(data: xr.DataArray, *, x=None, y=None):
    """Return 2-D logical coordinate grids and their labels."""
    if x is None or y is None:
        if data.ndim != 2:
            raise ValueError(f"x and y are required unless data is two-dimensional; got {data.dims}")
        x, y = data.dims
    if set(data.dims) != {x, y}:
        raise ValueError(f"selected data must contain exactly {x!r} and {y!r}; got {data.dims}")
    xgrid, ygrid = np.meshgrid(np.asarray(data.coords[x]), np.asarray(data.coords[y]), indexing="ij")
    return xgrid, ygrid, axis_label(data, x), axis_label(data, y)


def physical_grids(data: xr.DataArray, *, plane="XY"):
    """Return physical auxiliary coordinates already attached to a selected field."""
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; expected one of {tuple(PLANES)}")
    xname, yname, xlabel, ylabel = PLANES[plane]
    missing = [name for name in ("X", "Y", "Z") if name not in data.coords]
    if missing:
        raise ValueError(f"physical coordinates are not attached to {data.name!r}: missing {missing}")
    xcoord = np.sqrt(data.X**2 + data.Y**2) if xname == "R" else data.coords[xname]
    ycoord = data.coords[yname]
    if xcoord.ndim != 2 or ycoord.ndim != 2:
        raise ValueError("select all but two spatial dimensions before requesting a physical grid")
    return np.asarray(xcoord), np.asarray(ycoord), xlabel, ylabel


def _slice_data(data, view):
    selected = _select(data, view)
    if view.sweep in selected.dims and view.sweep not in (view.x, view.y):
        raise ValueError(f"select one {view.sweep!r} value before drawing a static slice, or display it as x or y")
    if view.x is None or view.y is None:
        if selected.ndim != 2:
            raise ValueError(f"view.x and view.y are required for remaining dims {selected.dims}")
        x, y = selected.dims
    else:
        x, y = view.x, view.y
    if set(selected.dims) != {x, y}:
        raise ValueError(f"selection leaves dimensions {selected.dims}; expected only {x!r}, {y!r}")
    selected = selected.transpose(x, y)
    grids = (
        physical_grids(selected, plane=view.plane)
        if view.coordinates == "physical"
        else logical_grids(selected, x=x, y=y)
    )
    return selected, grids


def plot_timeseries(data, *, ax=None, logy=True, fit: GrowthFit | None = None, title=None, run_label=None):
    """Plot one or more time series, each on its own time grid; series of different runs are labeled by run."""
    series = _items(data)
    if not series:
        raise ValueError("at least one time series is required")
    for item in series:
        validate_array(item, required_dims=("t",))
        if item.dims != ("t",):
            raise ValueError(f"time series must have dims ('t',), got {item.dims}")
    label_of = _label
    if len({item.attrs.get("run_name") for item in series}) > 1:

        def label_of(item):
            return " ".join(
                filter(None, (_label(item), f"({item.attrs['run_name']})" if item.attrs.get("run_name") else ""))
            )

    run_label = shared_run_label(series) if run_label is None else run_label
    own_figure = ax is None
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        artists, fits = [], []
        for item in series:
            (line,) = ax.plot(item.t, item, label=label_of(item) or None)
            artists.append(line)
            result = growth_rate(item, fit) if fit is not None else None
            fits.append(result)
            if result is not None:
                (fitted,) = ax.plot(
                    result.time,
                    result.fitted,
                    "--",
                    color=line.get_color(),
                    label=rf"fit: $\gamma$ = {result.rate:.4e}",
                )
                ax.axvspan(result.time[0], result.time[-1], alpha=0.12, color="grey")
                artists.append(fitted)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(series[0], "t"))
        ax.set_ylabel(value_label(series[0]))
        ax.set_title(title if title is not None else _label(series[0]))
        if any(label_of(item) for item in series) or fit is not None:
            ax.legend()
        _finish(fig, run_label=run_label if own_figure else "", tight=own_figure)
    return PlotResult(fig, ax, artists, fits)


def plot_lineout(data: xr.DataArray, *, x: str | None = None, ax=None, title=None):
    """Plot a selected one-dimensional profile using one named coordinate."""
    validate_array(data)
    if data.ndim != 1:
        raise ValueError(f"lineout needs exactly one remaining dimension, got {data.dims}")
    x = data.dims[0] if x is None else x
    if x != data.dims[0]:
        raise ValueError(f"lineout coordinate {x!r} is not the remaining dimension {data.dims[0]!r}")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    (line,) = ax.plot(data[x], data)
    ax.set(xlabel=axis_label(data, x), ylabel=value_label(data), title=_label(data) if title is None else title)
    _finish(fig, run_label=shared_run_label(data) if line.axes.figure is fig else "")
    return PlotResult(fig, ax, [line])


def plot_vector(
    data: xr.DataArray, *, x: str, y: str, components: tuple[int, int] = (0, 1), component_dim: str = "component", ax=None,
    stride: int = 1, coordinates: Literal["logical", "physical"] = "logical",
):
    """Render two components of a selected vector field with Matplotlib quivers."""
    validate_array(data, required_dims=(component_dim, x, y))
    if set(data.dims) != {component_dim, x, y}:
        raise ValueError(f"select every dimension except {component_dim!r}, {x!r}, and {y!r}; got {data.dims}")
    if stride < 1:
        raise ValueError("stride must be positive")
    vector = data.transpose(component_dim, x, y).isel({component_dim: list(components), x: slice(None, None, stride), y: slice(None, None, stride)})
    if coordinates == "physical":
        planes = {frozenset(("e1", "e2")): "XY", frozenset(("e1", "e3")): "XZ", frozenset(("e2", "e3")): "YZ"}
        plane = planes.get(frozenset((x, y)))
        if plane is None:
            raise ValueError("physical vector plots require two logical spatial dimensions")
        xg, yg, xlabel, ylabel = physical_grids(vector.isel({component_dim: 0}), plane=plane)
    else:
        xg, yg, xlabel, ylabel = logical_grids(vector.isel({component_dim: 0}), x=x, y=y)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    quiver = ax.quiver(xg, yg, vector.isel({component_dim: 0}), vector.isel({component_dim: 1}))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=_label(data), aspect="equal" if coordinates == "physical" else "auto")
    _finish(fig, run_label=shared_run_label(data))
    return PlotResult(fig, ax, [quiver])


def plot_volume_slices(data: xr.DataArray, *, indices: dict[str, int] | None = None, cmap=None):
    """Show three orthogonal midpoint slices of a selected scalar volume."""
    validate_array(data, required_dims=("e1", "e2", "e3"))
    if set(data.dims) != {"e1", "e2", "e3"}:
        raise ValueError(f"select every non-spatial dimension before volume_slices(); got {data.dims}")
    indices = {dim: data.sizes[dim] // 2 for dim in data.dims} | (indices or {})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    artists = []
    for ax, normal, x, y in zip(axes, ("e3", "e2", "e1"), ("e1", "e1", "e2"), ("e2", "e3", "e3")):
        plane = data.isel({normal: indices[normal]}).transpose(x, y)
        mesh = ax.pcolormesh(plane[x], plane[y], np.asarray(plane).T, shading="auto", cmap=cmap)
        ax.set(xlabel=axis_label(plane, x), ylabel=axis_label(plane, y), title=f"{normal} index {indices[normal]}")
        fig.colorbar(mesh, ax=ax, label=value_label(data))
        artists.append(mesh)
    fig.suptitle(" — ".join(filter(None, (_label(data), shared_run_label(data)))) )
    return PlotResult(fig, axes, artists)


def plot_compare(first: xr.DataArray, second: xr.DataArray, *, mode: Literal["difference", "ratio"] = "difference", ax=None):
    """Plot a one-dimensional aligned difference or ratio of two arrays."""
    first, second = xr.align(first, second, join="inner")
    result = first - second if mode == "difference" else xr.where(second != 0, first / second, np.nan)
    result.name = f"{_label(first)} {mode}"
    return plot_lineout(result, ax=ax)


def pyvista_volume(data: xr.DataArray, *, name: str | None = None, cmap="viridis", opacity="linear"):
    """Create a PyVista volume view from a selected scalar field with ``X/Y/Z`` coordinates.

    The returned plotter is not shown automatically; call ``plotter.show()`` in an
    interactive session or use PyVista's off-screen rendering options in batch jobs.
    """
    import pyvista as pv

    validate_array(data, required_dims=("e1", "e2", "e3"))
    if set(data.dims) != {"e1", "e2", "e3"}:
        raise ValueError(f"select every non-spatial dimension before pyvista_volume(); got {data.dims}")
    if any(coord not in data.coords for coord in ("X", "Y", "Z")):
        raise ValueError("pyvista_volume() requires mapped X, Y, and Z coordinates")
    grid = pv.StructuredGrid(
        np.asarray(data.X, dtype=float), np.asarray(data.Y, dtype=float), np.asarray(data.Z, dtype=float)
    )
    name = name or _label(data) or "value"
    grid.point_data[name] = np.asarray(data).ravel(order="F")
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars=name, cmap=cmap, opacity=opacity)
    plotter.show_axes()
    return plotter


def show_equilibrium(path_out, *, scalars: str = "p0", cmap="viridis"):
    """Create a PyVista view of ``geometry.vts`` and its equilibrium scalar field."""
    import pyvista as pv

    grid = pv.read(str(Path(path_out) / "geometry.vts"))
    if scalars not in grid.point_data:
        raise KeyError(f"{scalars!r} is not available; choices: {tuple(grid.point_data)}")
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=scalars, cmap=cmap, show_edges=False)
    plotter.show_axes()
    return plotter


class _SliceRenderer:
    """Shared selection, color limits and mesh rendering for every slice presentation."""

    def __init__(self, data, view, *, vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None):
        self.data = _select(data, view)
        self.view = View(x=view.x, y=view.y, sweep=view.sweep, coordinates=view.coordinates, plane=view.plane)
        self.vmin, self.vmax = vmin, vmax
        self.shared_clim = shared_clim
        self.cmap = cmap or STRUPHY_STYLE["image.cmap"]
        self.equal_aspect = view.coordinates == "physical" if equal_aspect is None else equal_aspect
        self.title = _label(data) if title is None else title
        self.limits = self._limits(self.data) if shared_clim else None

    def _limits(self, data):
        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax
        values = np.asarray(data)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError("cannot determine color limits from data without finite values; provide vmin and vmax")
        return (
            float(finite.min()) if self.vmin is None else self.vmin,
            float(finite.max()) if self.vmax is None else self.vmax,
        )

    def draw(self, ax, data):
        values, (xg, yg, xlabel, ylabel) = _slice_data(data, self.view)
        lo, hi = self.limits if self.shared_clim else self._limits(values)
        mesh = ax.pcolormesh(xg, yg, values, shading="auto", vmin=lo, vmax=hi, cmap=self.cmap)
        ax.set(xlabel=xlabel, ylabel=ylabel, aspect="equal" if self.equal_aspect else "auto")
        ax.grid(False)
        return mesh

    def frame_title(self, index):
        return f"{self.title} at {self.view.sweep} = {float(self.data[self.view.sweep][index]):.3e}"

    def indices(self, step):
        if not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("step must be a positive integer")
        validate_array(self.data, required_dims=(self.view.sweep,))
        if not self.data.sizes[self.view.sweep]:
            raise ValueError("cannot render an empty sweep")
        return range(0, self.data.sizes[self.view.sweep], step)


def plot_slice(
    data: xr.DataArray,
    *,
    view=None,
    ax=None,
    vmin=None,
    vmax=None,
    equal_aspect=None,
    title=None,
    run_label=None,
    cmap=None,
    shared_clim=True,
):
    """Render one selected two-dimensional slice."""
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
        shared_clim=shared_clim,
    )
    run_label = shared_run_label(data) if run_label is None else run_label
    own_figure = ax is None
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        mesh = renderer.draw(ax, renderer.data)
        fig.colorbar(mesh, ax=ax, label=value_label(data))
        ax.set_title(renderer.title)
        _finish(fig, run_label=run_label if own_figure else "", tight=own_figure)
    return PlotResult(fig, ax, [mesh])


def plot_panels(
    data: xr.DataArray,
    *,
    view=None,
    nrows=3,
    ncols=4,
    shared_clim=True,
    title=None,
    run_label=None,
    vmin=None,
    vmax=None,
    cmap=None,
    equal_aspect=None,
):
    """Plot snapshots with common color limits over the entire selected sweep by default."""
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
    )
    renderer.indices(1)
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    sweep = renderer.view.sweep
    indices = np.linspace(0, renderer.data.sizes[sweep] - 1, nrows * ncols).astype(int)
    run_label = shared_run_label(data) if run_label is None else run_label
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 2.8 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        meshes = []
        for ax, index in zip(axes.ravel(), indices):
            mesh = renderer.draw(ax, renderer.data.isel({sweep: int(index)}))
            meshes.append(mesh)
            ax.set_title(f"{sweep} = {float(renderer.data[sweep][index]):.3e}")
            if not shared_clim:
                fig.colorbar(mesh, ax=ax, label=value_label(data))
        if shared_clim:
            fig.colorbar(meshes[-1], ax=list(axes.ravel()), label=value_label(data))
        fig.suptitle(" — ".join(filter(None, (renderer.title, run_label))))
    return PlotResult(fig, axes, meshes)


class InteractiveSliceViewer:
    """Slider view with the same rendering options as static and exported slices."""

    def __init__(
        self,
        data: xr.DataArray,
        *,
        view=None,
        vmin=None,
        vmax=None,
        run_label=None,
        shared_clim=True,
        cmap=None,
        equal_aspect=None,
        title=None,
    ):
        self.data = validate_array(data)
        self.view = view or View()
        self.options = dict(
            vmin=vmin, vmax=vmax, shared_clim=shared_clim, cmap=cmap, equal_aspect=equal_aspect, title=title
        )
        self.run_label = shared_run_label(data) if run_label is None else run_label
        self.result = None
        self.sliders = {}

    def show(self):
        (self.result or self.draw()).show()
        return self

    def _ipython_display_(self):
        (self.result or self.draw())._ipython_display_()

    def draw(self):
        if self.result is not None:
            return self.result
        renderer = _SliceRenderer(self.data, self.view, **self.options)
        base = renderer.data
        x, y = self.view.x, self.view.y
        if x is None or y is None:
            candidates = [dim for dim in base.dims if dim != self.view.sweep]
            if len(candidates) < 2:
                raise ValueError("viewer needs two display dimensions")
            x, y = candidates[:2]
        renderer.view = View(x=x, y=y, coordinates=self.view.coordinates, plane=self.view.plane)
        controls = [dim for dim in base.dims if dim not in {x, y}]
        indices = {dim: 0 for dim in controls}
        with plt.rc_context(STRUPHY_STYLE):
            fig, ax = plt.subplots()
            fig.subplots_adjust(bottom=0.13 + 0.05 * len(controls))
            mesh = renderer.draw(ax, base.isel(indices))
            colorbar = fig.colorbar(mesh, ax=ax, label=value_label(self.data))
            self.result = PlotResult(fig, ax, [mesh])

            def update(_=None):
                for dim, slider in self.sliders.items():
                    indices[dim] = int(slider.val)
                self.result.artists[0].remove()
                mesh = renderer.draw(ax, base.isel(indices))
                self.result.artists[:] = [mesh]
                colorbar.update_normal(mesh)
                values = ", ".join(f"{dim}={float(base[dim][index]):.3e}" for dim, index in indices.items())
                ax.set_title(" at ".join(filter(None, (renderer.title, values))))
                fig.canvas.draw_idle()

            for row, dim in enumerate(controls):
                if base.sizes[dim] == 1:
                    continue
                slider_ax = fig.add_axes([0.20, 0.05 + 0.05 * row, 0.60, 0.025])
                slider = Slider(slider_ax, dim, 0, base.sizes[dim] - 1, valstep=1)
                slider.on_changed(update)
                self.sliders[dim] = slider
            update()
            _finish(fig, run_label=self.run_label, tight=False)
            # Keep widget callbacks alive even if only the PlotResult is retained.
            self.result.data["viewer"] = self
        return self.result


def animate_slices(
    data: xr.DataArray,
    *,
    view=None,
    interval=100,
    step=1,
    vmin=None,
    vmax=None,
    shared_clim=True,
    cmap=None,
    equal_aspect=None,
    title=None,
):
    """Animate slices with fixed color limits over the selected sweep by default."""
    from matplotlib.animation import FuncAnimation

    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
    )
    frames = renderer.indices(step)
    sweep = renderer.view.sweep
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        mesh = renderer.draw(ax, renderer.data.isel({sweep: 0}))
        colorbar = fig.colorbar(mesh, ax=ax, label=value_label(data))
        _finish(fig, run_label=shared_run_label(data))

    def update(index):
        nonlocal mesh
        mesh.remove()
        mesh = renderer.draw(ax, renderer.data.isel({sweep: index}))
        colorbar.update_normal(mesh)
        ax.set_title(renderer.frame_title(index))
        return (mesh,)

    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _detach_figure(fig)
    return animation


def save_frames(
    data: xr.DataArray,
    directory,
    *,
    view=None,
    step=1,
    prefix="frame",
    dpi=110,
    vmin=None,
    vmax=None,
    shared_clim=True,
    cmap=None,
    equal_aspect=None,
    title=None,
):
    """Export the configured sweep as PNGs, sharing color limits by default."""
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
    )
    frames = renderer.indices(step)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        try:
            sweep = renderer.view.sweep
            mesh = renderer.draw(ax, renderer.data.isel({sweep: 0}))
            colorbar = fig.colorbar(mesh, ax=ax, label=value_label(data))
            _finish(fig, run_label=shared_run_label(data))
            for frame, index in enumerate(frames):
                mesh.remove()
                mesh = renderer.draw(ax, renderer.data.isel({sweep: index}))
                colorbar.update_normal(mesh)
                ax.set_title(renderer.frame_title(index))
                path = directory / f"{prefix}_{frame:04d}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                paths.append(str(path))
        finally:
            plt.close(fig)
    return paths


def plot_scalars(scalars, *, names=None, exclude=SCALARS_EXCLUDE, relative_to=None, logy=False, run_label=None):
    """Plot every scalar time series in one axes."""
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        raise ValueError("no scalars to plot")
    run_label = shared_run_label([scalars[name] for name in selected]) if run_label is None else run_label
    fig, ax = plt.subplots(layout="constrained")
    for name in selected:
        values = scalars[name] / scalars[relative_to] if relative_to else scalars[name]
        ax.plot(values.t, values, label=name)
    if logy:
        ax.set_yscale("log")
    units = {scalars[name].attrs.get("units", "") for name in selected}
    ylabel = f"quantity / {relative_to}" if relative_to else (f"[{units.pop()}]" if len(units) == 1 else "[a.u.]")
    ax.set(xlabel=axis_label(scalars[selected[0]], "t"), ylabel=ylabel, title="Scalars")
    ax.legend(fontsize="small")
    if run_label:
        fig.suptitle(run_label, fontsize="small")
    return PlotResult(fig, ax, list(ax.lines))


def save_all_scalars(
    scalars,
    directory,
    *,
    names=None,
    exclude=SCALARS_EXCLUDE,
    logy=False,
    run_label=None,
    table="csv",
    file_format="png",
    dpi=110,
):
    """Write a table, scalar overview and one figure per scalar."""
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    if table:
        paths.append(save_scalars(scalars, str(directory / f"scalars.{table}"), names=selected, fmt=table))
    overview = plot_scalars(scalars, names=selected, logy=logy, run_label=run_label)
    path = directory / f"scalars.{file_format}"
    overview.save(path, dpi=dpi, close=True)
    paths.append(str(path))
    for name in selected:
        result = plot_timeseries(scalars[name], logy=logy, title=name, run_label=run_label)
        path = directory / f"{name}.{file_format}"
        result.save(path, dpi=dpi, close=True)
        paths.append(str(path))
    return paths


def plot_marker_trajectories(orbits: xr.DataArray, *, ax=None, max_markers=200, show_paths=None):
    """Plot a static 3-D trajectory overview; interactive marker UI is intentionally separate."""
    validate_array(orbits, required_dims=("t", "marker", "quantity"))
    count = min(orbits.sizes["marker"], max_markers)
    positions = np.asarray(orbits.isel(marker=slice(0, count)).sel(quantity=["x", "y", "z"]))
    fig = plt.figure() if ax is None else ax.figure
    ax = fig.add_subplot(111, projection="3d") if ax is None else ax
    show_paths = count <= 200 if show_paths is None else show_paths
    artists = []
    if show_paths:
        for marker in range(count):
            artists.extend(ax.plot(*positions[:, marker].T, lw=0.8, alpha=0.5))
    artists.append(ax.scatter(*positions[-1].T, s=8))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Marker trajectories")
    return PlotResult(fig, ax, artists)


def plot_equilibrium_profile(path_out, *, ax=None):
    """Plot radial equilibrium profiles from ``geometry.vts``."""
    import pyvista as pv

    equilibrium = pv.read(str(Path(path_out) / "geometry.vts"))
    shape = equilibrium.dimensions
    grid = np.reshape(equilibrium.points, shape + (3,))
    radius = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2)
    pressure = np.reshape(equilibrium.point_data["p0"], shape)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.plot(radius[0, 0], pressure[0, 0], label=r"$p_0$")
    if "n0" in equilibrium.point_data:
        density = np.reshape(equilibrium.point_data["n0"], shape)
        ax.plot(radius[0, 0], density[0, 0], label=r"$n_0$")
        ax.plot(radius[0, 0], pressure[0, 0] / density[0, 0], label=r"$T_0$")
    ax.set(xlabel=r"$R$", title="Radial equilibrium profiles")
    ax.legend()
    return PlotResult(fig, ax, list(ax.lines))
