"""``array.struphy.<kind>(...)``: plots and diagnostics of a single labeled array.

Every product of an :class:`~struphy.Output` carries this accessor, and so does every array
derived from one, e.g. ``out.ions.eta1_v1.f.isel(v1=0).struphy.plot.timeseries()``. Plots that
need the whole run (the scalar overview, the equilibrium profiles) live on ``out.plot``.

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
        from struphy.diagnostics.plotting import View

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
        from struphy.diagnostics.plotting import GrowthFit, plot_timeseries

        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries([self._array, *others], ax=ax, logy=logy, fit=growth, title=title)

    def slice(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        ax=None,
        **selection,
    ):
        """A two-dimensional color plot of one slice.

        Parameters
        ----------
        x, y:
            Displayed dimensions, e.g. ``x="e1", y="v1"``; inferred for two-dimensional data.
            The sweep dimension ``t`` may be displayed, which gives a space-time map.
        coords:
            ``"physical"`` draws on the mapped coordinates of ``plane`` instead of logical ones.
        **selection:
            One value per remaining dimension, e.g. ``t="last", component=2, e3=0``.

        Examples
        --------
        >>> out.ions.eta1_v1.f.struphy.plot.slice(x="e1", y="v1", t="last")
        >>> out.em_fields.b_field_phy.struphy.plot.slice(x="e1", y="e2", component=2, e3=0, coords="physical")
        """
        from struphy.diagnostics.plotting import plot_slice

        return plot_slice(
            self._array,
            view=self._view(x, y, "t", coords, plane, selection),
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            equal_aspect=equal_aspect,
            title=title,
        )

    def panels(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        nrows: int = 3,
        ncols: int = 4,
        shared_clim: bool = True,
        title: str | None = None,
        **selection,
    ):
        """Snapshots evenly spread along ``sweep`` (time by default), one panel each.

        Takes the same arguments as :meth:`slice`, except that ``sweep`` is not selected.
        """
        from struphy.diagnostics.plotting import plot_panels

        return plot_panels(
            self._array,
            view=self._view(x, y, sweep, coords, plane, selection),
            nrows=nrows,
            ncols=ncols,
            shared_clim=shared_clim,
            title=title,
        )

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
        **selection,
    ):
        """An interactive viewer with one slider per dimension that is neither displayed nor selected.

        Takes the same arguments as :meth:`slice`. Call ``.show()`` on the result, and keep it
        alive so that the sliders stay connected.
        """
        from struphy.diagnostics.plotting import InteractiveSliceViewer

        return InteractiveSliceViewer(
            self._array, view=self._view(x, y, sweep, coords, plane, selection), vmin=vmin, vmax=vmax
        )

    def animation(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        interval: int = 100,
        step: int = 1,
        vmin=None,
        vmax=None,
        **selection,
    ):
        """A Matplotlib animation along ``sweep``, taking the same arguments as :meth:`slice`."""
        from struphy.diagnostics.plotting import animate_slices

        return animate_slices(
            self._array,
            view=self._view(x, y, sweep, coords, plane, selection),
            interval=interval,
            step=step,
            vmin=vmin,
            vmax=vmax,
        )

    def frames(
        self,
        directory,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        step: int = 1,
        prefix: str = "frame",
        dpi: int = 110,
        **selection,
    ) -> list[str]:
        """Write the slices along ``sweep`` as numbered PNG files; returns their paths.

        Takes the same arguments as :meth:`slice`.
        """
        from struphy.diagnostics.plotting import save_frames

        return save_frames(
            self._array,
            directory,
            view=self._view(x, y, sweep, coords, plane, selection),
            step=step,
            prefix=prefix,
            dpi=dpi,
        )

    def trajectories(self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None):
        """Three-dimensional paths of saved markers; for an orbit product."""
        from struphy.diagnostics.plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._array, ax=ax, max_markers=max_markers, show_paths=show_paths)


class ArrayAnalysis(_ArrayAccessor):
    """Quantitative diagnostics of one array, as ``array.struphy.analysis.<quantity>(...)``."""


    def growth_rate(self, *, window: tuple[float | None, float | None] = (None, None), amplitude: bool = False):
        """Fit ``exp(rate * t + intercept)`` to this time series within ``window``.

        With ``amplitude=True`` the series is quadratic in an amplitude (e.g. an energy) and the
        amplitude's rate is returned. Returns a ``FitResult`` (``.rate``, ``.intercept``,
        ``.time``, ``.fitted``), or ``None`` with fewer than two valid samples.
        """
        from struphy.diagnostics.plotting import GrowthFit, growth_rate

        return growth_rate(self._array, GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def drift(self, *, ref=None) -> xr.DataArray:
        """Signed deviation of this time series from ``ref`` or from its first sample."""
        from struphy.diagnostics.plotting import drift

        return drift(self._array, ref=ref)

    def relative_error(self, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Absolute relative deviation from ``ref`` or from this series' first sample."""
        from struphy.diagnostics.plotting import relative_error

        return relative_error(self._array, ref=ref, skip_first=skip_first)

    def dispersion(self, *, component: int = 0, slice_at: tuple = (None, 0, 0), physical: bool = False, **kwargs):
        """Space-time power spectrum of this field and fitted dispersion branches.

        The time coordinate must be normalized, see :meth:`struphy.Output.with_time_units`. See
        :func:`struphy.diagnostics.diagn_tools.power_spectrum_2d` for ``slice_at``, the fit options
        and ``do_plot``. Returns ``(omega, kvec, spectrum, coeffs)``.
        """
        from struphy.diagnostics.diagn_tools import power_spectrum_2d

        if self._array.t.attrs.get("units") == "s":
            raise ValueError(
                "the spectrum needs normalized time; take the field from out.with_time_units('normalized')"
            )
        return power_spectrum_2d(self._array, component=component, slice_at=slice_at, physical=physical, **kwargs)
