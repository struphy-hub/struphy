"""Small, composable plotting functions for labeled Struphy output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Slider

from struphy.post_processing.arrays import (
    SCALARS_EXCLUDE,
    axis_label,
    orbit_columns,
    save_scalars,
    scalar_names,
    validate_array,
    value_label,
)

logger = logging.getLogger("struphy")

STRUPHY_STYLE = {
    "figure.figsize": (8.0, 5.0), "figure.dpi": 110, "axes.grid": True,
    "grid.alpha": 0.3, "axes.titlesize": "medium", "legend.frameon": False,
    "image.cmap": "viridis",
}

PLANES = {
    "XY": ("X", "Y", "X", "Y"),
    "XZ": ("X", "Z", "X", "Z"),
    "YZ": ("Y", "Z", "Y", "Z"),
    "RZ": ("R", "Z", "R", "Z"),
}


@dataclass(frozen=True)
class GrowthFit:
    """Configuration for an exponential growth-rate fit."""

    window: tuple[float | None, float | None] = (None, None)
    amplitude_from_quadratic: bool = False


@dataclass(frozen=True)
class FitResult:
    rate: float
    intercept: float
    time: np.ndarray
    fitted: np.ndarray


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
    """Already-rendered Matplotlib objects; saving never redraws them."""

    fig: object
    ax: object
    artists: list = field(default_factory=list)
    fit_results: list[FitResult | None] = field(default_factory=list)

    def save(self, path, *, close=False, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        if close:
            plt.close(self.fig)
        return str(path)

    def show(self):
        plt.show()
        return self


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


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


def growth_rate(data: xr.DataArray, fit: GrowthFit | None = None) -> FitResult | None:
    """Fit ``exp(rate*t + intercept)`` using only finite, positive samples."""
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"growth-rate input must have dims ('t',), got {data.dims}")
    fit = fit or GrowthFit()
    time, values = np.asarray(data.t), np.asarray(data)
    lo = time[0] if fit.window[0] is None else fit.window[0]
    hi = time[-1] if fit.window[1] is None else fit.window[1]
    lo, hi = sorted((lo, hi))
    valid = (time >= lo) & (time <= hi) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(valid) < 2:
        return None
    selected_time = time[valid]
    signal = np.log(np.sqrt(values[valid])) if fit.amplitude_from_quadratic else np.log(values[valid])
    rate, intercept = np.polyfit(selected_time, signal, 1)
    scale = 2.0 if fit.amplitude_from_quadratic else 1.0
    fitted = np.exp(scale * (rate * selected_time + intercept))
    return FitResult(float(rate), float(intercept), selected_time, fitted)


def drift(data: xr.DataArray, *, ref=None) -> xr.DataArray:
    """Signed deviation from an explicit reference or the first time sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    out = data - reference
    out.attrs = dict(data.attrs)
    out.attrs["label"] = f"{_label(data)} drift".strip()
    return out


def relative_error(data: xr.DataArray, *, ref=None, skip_first=True) -> xr.DataArray:
    """Absolute relative deviation from an explicit reference or first sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    if np.any(np.asarray(reference) == 0):
        raise ValueError("cannot take a relative error against a reference of zero")
    out = abs(data - reference) / abs(reference)
    out.attrs = {"label": f"relative error of {_label(data)}".strip(), "units": ""}
    return out.isel(t=slice(1, None)) if skip_first else out


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
    if view.sweep in selected.dims:
        raise ValueError(f"select one {view.sweep!r} value before drawing a static slice")
    if view.x is None or view.y is None:
        if selected.ndim != 2:
            raise ValueError(f"view.x and view.y are required for remaining dims {selected.dims}")
        x, y = selected.dims
    else:
        x, y = view.x, view.y
    if set(selected.dims) != {x, y}:
        raise ValueError(f"selection leaves dimensions {selected.dims}; expected only {x!r}, {y!r}")
    selected = selected.transpose(x, y)
    grids = physical_grids(selected, plane=view.plane) if view.coordinates == "physical" else logical_grids(selected, x=x, y=y)
    return selected, grids


def plot_timeseries(data, *, ax=None, logy=True, fit: GrowthFit | None = None, title=None, run_label=""):
    """Plot one or more aligned time series."""
    series = [data] if isinstance(data, xr.DataArray) else list(data)
    if not series:
        raise ValueError("at least one time series is required")
    for item in series:
        validate_array(item, required_dims=("t",))
        if item.dims != ("t",):
            raise ValueError(f"time series must have dims ('t',), got {item.dims}")
    if len(series) > 1:
        series = list(xr.align(*series, join="exact"))
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        artists, fits = [], []
        for item in series:
            line, = ax.plot(item.t, item, label=_label(item) or None)
            artists.append(line)
            result = growth_rate(item, fit) if fit is not None else None
            fits.append(result)
            if result is not None:
                fitted, = ax.plot(result.time, result.fitted, "--", color=line.get_color(),
                                  label=rf"fit: $\gamma$ = {result.rate:.4e}")
                ax.axvspan(result.time[0], result.time[-1], alpha=0.12, color="grey")
                artists.append(fitted)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(series[0], "t"))
        ax.set_ylabel(value_label(series[0]))
        ax.set_title(title if title is not None else _label(series[0]))
        if any(_label(item) for item in series) or fit is not None:
            ax.legend()
        _finish(fig, run_label=run_label, tight=ax is not None)
    return PlotResult(fig, ax, artists, fits)


def plot_slice(data: xr.DataArray, *, view=None, ax=None, vmin=None, vmax=None,
               equal_aspect=None, title=None, run_label=""):
    """Render one selected two-dimensional slice."""
    view = view or View()
    selected, (xgrid, ygrid, xlabel, ylabel) = _slice_data(data, view)
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        mesh = ax.pcolormesh(xgrid, ygrid, np.asarray(selected), shading="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label=value_label(data))
        if equal_aspect if equal_aspect is not None else view.coordinates == "physical":
            ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title if title is not None else _label(data))
        ax.grid(False)
        _finish(fig, run_label=run_label)
    return PlotResult(fig, ax, [mesh])


def plot_panels(data: xr.DataArray, *, view=None, nrows=3, ncols=4, shared_clim=True,
                title=None, run_label=""):
    """Plot snapshots spread across a sweep coordinate."""
    view = view or View()
    selected = _select(data, view)
    validate_array(selected, required_dims=(view.sweep,))
    count = nrows * ncols
    indices = np.linspace(0, selected.sizes[view.sweep] - 1, count).astype(int)
    snapshots = [selected.isel({view.sweep: int(index)}) for index in indices]
    limits = (None, None)
    if shared_clim:
        limits = (min(float(item.min()) for item in snapshots), max(float(item.max()) for item in snapshots))
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.8*nrows), sharex=True,
                                 sharey=True, squeeze=False, layout="constrained")
        meshes = []
        for ax, index, snapshot in zip(axes.ravel(), indices, snapshots):
            local_view = View(x=view.x, y=view.y, coordinates=view.coordinates, plane=view.plane)
            values, (xg, yg, xlabel, ylabel) = _slice_data(snapshot, local_view)
            mesh = ax.pcolormesh(xg, yg, values, shading="auto", vmin=limits[0], vmax=limits[1])
            meshes.append(mesh)
            ax.set_title(f"{view.sweep} = {float(selected[view.sweep][index]):.3e}")
            ax.grid(False)
            if not shared_clim:
                fig.colorbar(mesh, ax=ax)
        for ax in axes[-1]: ax.set_xlabel(xlabel)
        for row in axes: row[0].set_ylabel(ylabel)
        if shared_clim: fig.colorbar(meshes[-1], ax=list(axes.ravel()), label=value_label(data))
        heading = title if title is not None else _label(data)
        fig.suptitle(" — ".join(filter(None, (heading, run_label))))
    return PlotResult(fig, axes, meshes)


class InteractiveSliceViewer:
    """Stateful viewer using one recipe for the sweep and all remaining dimensions."""

    def __init__(self, data: xr.DataArray, *, view=None, vmin=None, vmax=None, run_label=""):
        self.data = validate_array(data)
        self.view = view or View()
        self.vmin, self.vmax, self.run_label = vmin, vmax, run_label
        self.result = None
        self.sliders = {}

    def show(self):
        return self.draw().show()

    def draw(self):
        base = _select(self.data, self.view)
        x, y = self.view.x, self.view.y
        if x is None or y is None:
            candidates = [dim for dim in base.dims if dim != self.view.sweep]
            if len(candidates) < 2:
                raise ValueError("viewer needs two display dimensions")
            x, y = candidates[:2]
        controls = [dim for dim in base.dims if dim not in {x, y}]
        indices = {dim: 0 for dim in controls}

        def frame():
            return base.isel(indices), View(x=x, y=y, coordinates=self.view.coordinates, plane=self.view.plane)

        selected, frame_view = frame()
        selected, (xg, yg, xlabel, ylabel) = _slice_data(selected, frame_view)
        with plt.rc_context(STRUPHY_STYLE):
            fig, ax = plt.subplots()
            fig.subplots_adjust(bottom=0.13 + 0.05*len(controls))
            mesh = ax.pcolormesh(xg, yg, selected, shading="auto", vmin=self.vmin, vmax=self.vmax)
            colorbar = fig.colorbar(mesh, ax=ax, label=value_label(self.data))
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.grid(False)
            if self.view.coordinates == "physical": ax.set_aspect("equal", adjustable="box")
            state = {"mesh": mesh}

            def update(_=None):
                for dim, slider in self.sliders.items(): indices[dim] = int(slider.val)
                item, item_view = frame()
                item, grids = _slice_data(item, item_view)
                state["mesh"].remove()
                state["mesh"] = ax.pcolormesh(grids[0], grids[1], item, shading="auto",
                                              vmin=self.vmin, vmax=self.vmax)
                if self.vmin is None and self.vmax is None:
                    state["mesh"].set_clim(float(item.min()), float(item.max()))
                colorbar.update_normal(state["mesh"])
                values = ", ".join(f"{dim}={float(base[dim][index]):.3e}" for dim, index in indices.items())
                ax.set_title(" at ".join(filter(None, (_label(self.data), values))))
                fig.canvas.draw_idle()

            for row, dim in enumerate(controls):
                slider_ax = fig.add_axes([0.20, 0.05 + 0.05*row, 0.60, 0.025])
                slider = Slider(slider_ax, dim, 0, base.sizes[dim]-1, valstep=1)
                slider.on_changed(update)
                self.sliders[dim] = slider
            update()
            _finish(fig, run_label=self.run_label, tight=False)
        self.result = PlotResult(fig, ax, [state["mesh"]])
        return self.result


def animate_slices(data: xr.DataArray, *, view=None, interval=100, step=1, vmin=None, vmax=None):
    """Create an animation using the same :class:`View` as static slices."""
    from matplotlib.animation import FuncAnimation
    view = view or View()
    selected = _select(data, view)
    frames = range(0, selected.sizes[view.sweep], step)
    first = selected.isel({view.sweep: 0})
    local = View(x=view.x, y=view.y, coordinates=view.coordinates, plane=view.plane)
    values, grids = _slice_data(first, local)
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(grids[0], grids[1], values, shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=value_label(data))
    ax.set(xlabel=grids[2], ylabel=grids[3])

    def update(index):
        item = selected.isel({view.sweep: index})
        item, item_grids = _slice_data(item, local)
        mesh.set_array(np.asarray(item).ravel())
        ax.set_title(f"{_label(data)} at {view.sweep} = {float(selected[view.sweep][index]):.3e}")
        return mesh,
    return FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)


def save_frames(data: xr.DataArray, directory, *, view=None, step=1, prefix="frame", dpi=110):
    """Write a sweep as PNG frames without retaining figures."""
    view = view or View()
    selected = _select(data, view)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame, index in enumerate(range(0, selected.sizes[view.sweep], step)):
        item = selected.isel({view.sweep: index})
        local = View(x=view.x, y=view.y, coordinates=view.coordinates, plane=view.plane)
        result = plot_slice(item, view=local,
                            title=f"{_label(data)} at {view.sweep} = {float(selected[view.sweep][index]):.3e}")
        path = directory / f"{prefix}_{frame:04d}.png"
        result.save(path, dpi=dpi, close=True)
        paths.append(str(path))
    return paths


def plot_scalars(scalars, *, names=None, exclude=SCALARS_EXCLUDE, relative_to=None,
                 error_panel="en_tot", logy=False, run_label=""):
    """Plot a scalar overview and optional conservation-error panel."""
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected: raise ValueError("no scalars to plot")
    has_error = error_panel is not None and error_panel in scalars
    fig, axes = plt.subplots(2 if has_error else 1, 1, sharex=has_error,
                             figsize=(8, 6.5) if has_error else None,
                             height_ratios=(2, 1) if has_error else None,
                             layout="constrained")
    ax = axes[0] if has_error else axes
    for name in selected:
        values = scalars[name] / scalars[relative_to] if relative_to else scalars[name]
        ax.plot(values.t, values, label=name)
    if logy: ax.set_yscale("log")
    units = {scalars[name].attrs.get("units", "") for name in selected}
    ylabel = f"quantity / {relative_to}" if relative_to else (f"[{units.pop()}]" if len(units) == 1 else "[a.u.]")
    ax.set(ylabel=ylabel, title="Scalars")
    ax.legend(fontsize="small")
    artists, error = list(ax.lines), None
    if has_error:
        error = relative_error(scalars[error_panel])
        axes[1].plot(error.t, error)
        if np.any(np.asarray(error) > 0): axes[1].set_yscale("log")
        axes[1].set(xlabel=axis_label(error, "t"), ylabel=f"relative error of {error_panel}")
        artists.extend(axes[1].lines)
    else:
        ax.set_xlabel(axis_label(scalars[selected[0]], "t"))
    if run_label: fig.suptitle(run_label, fontsize="small")
    return PlotResult(fig, axes, artists), error


def save_all_scalars(scalars, directory, *, names=None, exclude=SCALARS_EXCLUDE, logy=False,
                     run_label="", table="csv", file_format="png", dpi=110):
    """Write a table, scalar overview and one figure per scalar."""
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected: return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    if table:
        paths.append(save_scalars(scalars, str(directory / f"scalars.{table}"), names=selected, fmt=table))
    overview, _ = plot_scalars(scalars, names=selected, logy=logy, run_label=run_label)
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
    validate_array(orbits, required_dims=("t", "marker", "attribute"))
    columns = orbits.attrs.get("columns", orbit_columns(orbits.sizes["attribute"]))
    count = min(orbits.sizes["marker"], max_markers)
    values = np.asarray(orbits.isel(marker=slice(0, count)))
    fig = plt.figure() if ax is None else ax.figure
    ax = fig.add_subplot(111, projection="3d") if ax is None else ax
    positions = values[..., columns["position"]]
    show_paths = count <= 200 if show_paths is None else show_paths
    artists = []
    if show_paths:
        for marker in range(count):
            artists.extend(ax.plot(*positions[:, marker].T, lw=.8, alpha=.5))
    artists.append(ax.scatter(*positions[-1].T, s=8))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Marker trajectories")
    return PlotResult(fig, ax, artists)


def plot_equilibrium_profile(path_out, *, ax=None):
    """Plot radial equilibrium profiles from ``geometry.vts``."""
    import pyvista as pv

    equilibrium = pv.read(str(Path(path_out) / "geometry.vts"))
    shape = equilibrium.dimensions
    grid = np.reshape(equilibrium.points, shape + (3,))
    radius = np.sqrt(grid[..., 0]**2 + grid[..., 1]**2)
    pressure = np.reshape(equilibrium.point_data["p0"], shape)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.plot(radius[0, 0], pressure[0, 0], label=r"$p_0$")
    if "n0" in equilibrium.point_data:
        density = np.reshape(equilibrium.point_data["n0"], shape)
        ax.plot(radius[0, 0], density[0, 0], label=r"$n_0$")
        ax.plot(radius[0, 0], pressure[0, 0]/density[0, 0], label=r"$T_0$")
    ax.set(xlabel=r"$R$", title="Radial equilibrium profiles")
    ax.legend()
    return PlotResult(fig, ax, list(ax.lines))
