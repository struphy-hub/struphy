"""Optional accessors for plots and diagnostics of a single labeled array.

Every product of an :class:`~struphy.Output` carries this accessor after importing
``struphy_plots``, and so does every array derived from one.

Dimensions that are neither displayed nor swept are selected by naming them: an integer is a
position (``t=-1``), ``"first"`` and ``"last"`` are the ends, and a float is the nearest
coordinate value (``t=0.35``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

Coordinates = Literal["logical", "physical"]
Plane = Literal["XY", "XZ", "YZ", "RZ"]


@xr.register_dataarray_accessor("struphy")
class StruphyAccessor:
    """Struphy diagnostics of one array: ``array.struphy.plot`` and ``array.struphy.analysis``."""

    def __init__(self, array: xr.DataArray):
        self._array = array

    @property
    def plot(self) -> "ArrayPlots":
        """Plots of this array, e.g. ``array.struphy.plot.slice(x="e1", y="v1", t="last")``."""
        return ArrayPlots(self._array)

    @property
    def analysis(self) -> "ArrayAnalysis":
        """Diagnostics of this array, e.g. ``array.struphy.analysis.growth_rate()``."""
        return ArrayAnalysis(self._array)


class _ArrayAccessor:
    def __init__(self, array: xr.DataArray):
        self._array = array


class ArrayPlots(_ArrayAccessor):
    """Plots of one array, as ``array.struphy.plot.<kind>(...)``.

    Dimensions that are neither displayed nor swept are selected by naming them: an integer is a
    position (``t=-1``), ``"first"`` and ``"last"`` are the ends, and a float is the nearest
    coordinate value (``t=0.35``).
    """

    def _view(self, x, y, sweep, coords, plane, selection):
        from .plotting import View

        select, index = {}, {}
        for dim, value in selection.items():
            if dim not in self._array.dims:
                raise TypeError(
                    f"{dim!r} is not a dimension of {self._array.name!r}; its dimensions are {self._array.dims}"
                )
            if value == "first":
                index[dim] = 0
            elif value == "last":
                index[dim] = -1
            elif isinstance(value, (bool, str)):
                raise TypeError(f'cannot select {dim}={value!r}; use a number, or "first"/"last"')
            elif isinstance(value, (int, np.integer)):
                index[dim] = int(value)
            else:
                select[dim] = float(value)
        return View(x=x, y=y, sweep=sweep, select=select, isel=index, coordinates=coords, plane=plane)

    def timeseries(
        self, *others, logy: bool = True, fit=None, fit_amplitude: bool = False, title: str | None = None, ax=None
    ):
        """This time series, and any others given, in one axes.

        Parameters
        ----------
        *others:
            Further arrays with the single dimension ``t``; they may come from other runs and
            need not share this array's time grid.
        logy:
            Logarithmic value axis.
        fit:
            Time window ``(t0, t1)`` of an exponential fit per series (``None`` for an open end),
            or ``True`` for the whole series. Rates are in ``result.fit_results``.
        fit_amplitude:
            The series is quadratic in an amplitude (e.g. an energy); fit the amplitude's rate.
        """
        from .analysis import GrowthFit
        from .plotting import plot_timeseries

        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries([self._array, *others], ax=ax, logy=logy, fit=growth, title=title)

    def lineout(self, *, x: str | None = None, ax=None, title: str | None = None, **selection):
        """Plot a one-dimensional profile after selecting every other dimension."""
        from .plotting import _select, plot_lineout

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_lineout(_select(self._array, view), x=x, ax=ax, title=title)

    def vector(
        self,
        *,
        x: str,
        y: str,
        components: tuple[int, int] = (0, 1),
        stride: int = 1,
        coordinates: Coordinates = "logical",
        ax=None,
        **selection,
    ):
        """Plot two vector components after selecting time and remaining dimensions."""
        from .plotting import _select, plot_vector

        view = self._view(None, None, "t", coordinates, "XY", selection)
        return plot_vector(
            _select(self._array, view), x=x, y=y, components=components, stride=stride, coordinates=coordinates, ax=ax
        )

    def volume_slices(self, *, indices: dict[str, int] | None = None, cmap=None, **selection):
        """Render three orthogonal slices of a selected scalar volume."""
        from .plotting import _select, plot_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_volume_slices(_select(self._array, view), indices=indices, cmap=cmap)

    def volume(self, *, name: str | None = None, cmap="viridis", opacity="linear", **selection):
        """Create a PyVista volume plotter for a selected scalar field."""
        from .plotting import _select, pyvista_volume

        view = self._view(None, None, "t", "logical", "XY", selection)
        return pyvista_volume(_select(self._array, view), name=name, cmap=cmap, opacity=opacity)

    def compare(self, other: xr.DataArray, *, mode: Literal["difference", "ratio"] = "difference", ax=None):
        """Plot a one-dimensional aligned difference or ratio against another array."""
        from .plotting import plot_compare

        return plot_compare(self._array, other, mode=mode, ax=ax)

    def view(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        **selection,
    ) -> "SliceView":
        """Configure a reusable slice view without rendering a figure.

        Use xarray's ``.sel()``/``.isel()`` for general selection, or pass remaining
        dimensions here (integers are positions, floats nearest coordinates,
        ``"first"``/``"last"`` select an end).

        ``shared_clim=True`` fixes color limits over all selected data, including
        frames omitted by a panel layout or export step. False rescales each frame.
        Explicit ``vmin``/``vmax`` override either limit in both modes. ``cmap``,
        ``equal_aspect`` and ``title`` apply to every presentation of this view.

        Examples
        --------
        >>> view = f.struphy.plot.view(x="e1", y="v1", cmap="RdBu_r")
        >>> view.slice(t="last")
        >>> view.panels(nrows=2, ncols=3)
        >>> view.save_frames("frames")
        """
        self._view(x, y, sweep, coords, plane, selection)  # validate selections now
        return SliceView(
            self._array,
            dict(x=x, y=y, sweep=sweep, coords=coords, plane=plane),
            selection,
            dict(vmin=vmin, vmax=vmax, shared_clim=shared_clim, cmap=cmap, equal_aspect=equal_aspect, title=title),
        )

    def slice(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        ax=None,
        **selection,
    ):
        """Render one 2-D slice; see :meth:`view` for shared options."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).slice(ax=ax)

    def panels(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        nrows: int = 3,
        ncols: int = 4,
        **selection,
    ):
        """Render evenly spaced snapshots; see :meth:`view` for shared options."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).panels(nrows=nrows, ncols=ncols)

    def viewer(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        **selection,
    ):
        """Create an interactive slider view; retain the returned viewer."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).viewer()

    def animation(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        interval: int = 100,
        step: int = 1,
        **selection,
    ):
        """Animate the sweep; retain the returned Matplotlib animation."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).animation(interval=interval, step=step)

    def frames(
        self,
        directory,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        step: int = 1,
        prefix: str = "frame",
        dpi: int = 110,
        **selection,
    ):
        """Export PNGs; equivalent to ``plot.view(...).save_frames(directory)``."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).save_frames(directory, step=step, prefix=prefix, dpi=dpi)

    def trajectories(self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None):
        """Three-dimensional paths of saved markers; for an orbit product."""
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._array, ax=ax, max_markers=max_markers, show_paths=show_paths)


class SliceView:
    """A configured array view, shared by static, interactive and exported plots.

    Construct with ``array.struphy.plot.view(...)``. Configuration does not create
    figures or copy the underlying array.
    """

    def __init__(self, array, coordinates, selection, options):
        self._array = array
        self._coordinates = dict(coordinates)
        self._selection = dict(selection)
        self._options = dict(options)

    def _view(self, **selection):
        return ArrayPlots(self._array)._view(**self._coordinates, selection={**self._selection, **selection})

    def slice(self, *, ax=None, **selection):
        """Draw a snapshot, e.g. ``view.slice(t="last")``; return a PlotResult."""
        # Resolve shared limits before selecting a single snapshot, so it uses
        # the same scale as panels, animation and export of this configured view.
        from .plotting import _SliceRenderer, plot_slice

        options = dict(self._options)
        if options["shared_clim"]:
            renderer = _SliceRenderer(self._array, self._view(), **options)
            options.update(zip(("vmin", "vmax"), renderer.limits))
        return plot_slice(self._array, view=self._view(**selection), ax=ax, **options)

    def panels(self, *, nrows=3, ncols=4):
        """Draw snapshots spread along the sweep; return a PlotResult."""
        from .plotting import plot_panels

        return plot_panels(self._array, view=self._view(), nrows=nrows, ncols=ncols, **self._options)

    def viewer(self):
        """Create a viewer with sliders for unselected dimensions."""
        from .plotting import InteractiveSliceViewer

        return InteractiveSliceViewer(self._array, view=self._view(), **self._options)

    def animation(self, *, interval=100, step=1):
        """Create a Matplotlib animation using this view's rendering options."""
        from .plotting import animate_slices

        return animate_slices(self._array, view=self._view(), interval=interval, step=step, **self._options)

    def save_frames(self, directory, *, step=1, prefix="frame", dpi=110):
        """Export PNG frames using this view's rendering options; return paths."""
        from .plotting import save_frames

        return save_frames(
            self._array, directory, view=self._view(), step=step, prefix=prefix, dpi=dpi, **self._options
        )


class ArrayAnalysis(_ArrayAccessor):
    """Quantitative diagnostics of one array, as ``array.struphy.analysis.<quantity>(...)``."""

    def growth_rate(self, *, window: tuple[float | None, float | None] = (None, None), amplitude: bool = False):
        """Fit ``exp(rate * t + intercept)`` to this time series within ``window``.

        With ``amplitude=True`` the series is quadratic in an amplitude (e.g. an energy) and the
        amplitude's rate is returned. Returns a ``FitResult`` (``.rate``, ``.intercept``,
        ``.time``, ``.fitted``), or ``None`` with fewer than two valid samples.
        """
        from .analysis import GrowthFit, growth_rate

        return growth_rate(self._array, GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def damping_rate(self, *, window: tuple[float | None, float | None] = (None, None), amplitude: bool = False):
        """Fit exponential decay to the envelope of this oscillating time series; see ``growth_rate``."""
        from .analysis import GrowthFit, damping_rate

        return damping_rate(self._array, GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def envelope(self) -> xr.DataArray:
        """Local maxima of this time series."""
        from .analysis import envelope

        return envelope(self._array)

    def norm(self, *, dims=None, squared: bool = False) -> xr.DataArray:
        """L2 norm over ``dims`` (default: every dimension except ``t``)."""
        from .analysis import norm

        return norm(self._array, dims=dims, squared=squared)

    def drift(self, *, ref=None) -> xr.DataArray:
        """Signed deviation of this time series from ``ref`` or from its first sample."""
        from .analysis import drift

        return drift(self._array, ref=ref)

    def relative_error(self, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Absolute relative deviation from ``ref`` or from this series' first sample."""
        from .analysis import relative_error

        return relative_error(self._array, ref=ref, skip_first=skip_first)

    def spatial_average(self, *, dims=None) -> xr.DataArray:
        """Mean over the logical space dimensions ``e1``, ``e2``, ``e3`` (or ``dims``).

        For a binned ``e1_v1`` distribution this is f(v1, t) averaged over space; see
        :func:`struphy_plots.analysis.spatial_average`.
        """
        from .analysis import spatial_average

        return spatial_average(self._array, dims=dims)

    def velocity_moments(self, *, dims=None) -> xr.Dataset:
        """Density, mean velocity and variance of a binned distribution over its velocity dimensions.

        See :func:`struphy_plots.analysis.velocity_moments` for the definitions.
        """
        from .analysis import velocity_moments

        return velocity_moments(self._array, dims=dims)

    def dispersion(self, *, component: int = 0, slice_at: tuple = (None, 0, 0), physical: bool = False, **kwargs):
        """Space-time power spectrum of this field and fitted dispersion branches.

        The time coordinate must be normalized, see :meth:`struphy.Output.with_time_units`. See
        :func:`struphy.post_processing.spectral.compute_dispersion` for ``slice_at`` and fit options.
        Returns an xarray Dataset with ``power(omega, k)``.
        """
        from struphy.post_processing.spectral import compute_dispersion

        if self._array.t.attrs.get("units") == "s":
            raise ValueError(
                "the spectrum needs normalized time; take the field from out.with_time_units('normalized')"
            )
        return compute_dispersion(self._array, component=component, slice_at=slice_at, physical=physical, **kwargs)
