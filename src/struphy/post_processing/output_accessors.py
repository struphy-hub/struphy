"""``out.plot`` and ``out.analysis``: plotting and analysis without extra imports.

Every method accepts a product name (``"en_phi"``, ``"em_fields/phi_log"``,
``"kinetic_ions/e1_v1_density/f_binned"``; see :meth:`Output.__getitem__`) or any labeled
array, including arrays derived from or belonging to another run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

if TYPE_CHECKING:
    from struphy.post_processing.output import Output

Coordinates = Literal["logical", "physical"]
Plane = Literal["XY", "XZ", "YZ", "RZ"]


class OutputPlots:
    """Standard plots of a run, as ``out.plot.<kind>(...)``.

    Plots return a rendered :class:`~struphy.diagnostics.plotting.PlotResult` with
    ``.show()`` and ``.save(path)``. Figures are titled with the run's numerical parameters;
    time series of different runs are labeled by run.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, data) -> xr.DataArray:
        return self._output[data] if isinstance(data, str) else data

    def _label(self, arrays) -> str:
        from struphy.diagnostics.plotting import shared_run_label

        runs = {array.attrs.get("run") for array in arrays} - {None, ""}
        return shared_run_label(arrays) if runs else self._output.label

    @staticmethod
    def _view(x, y, sweep, coords, plane, select, isel):
        from struphy.diagnostics.plotting import View

        return View(
            x=x, y=y, sweep=sweep, select=dict(select or {}), isel=dict(isel or {}), coordinates=coords, plane=plane
        )

    def scalars(self, names=None, *, relative_to: str | None = None, logy: bool = False):
        """Overview of the scalar time series in one axes.

        Parameters
        ----------
        names:
            Scalars to show; all by default.
        relative_to:
            Show every scalar divided by this one.
        logy:
            Logarithmic value axis.
        """
        from struphy.diagnostics.plotting import plot_scalars

        return plot_scalars(
            self._output.scalars, names=names, relative_to=relative_to, logy=logy, run_label=self._output.label
        )

    def timeseries(
        self,
        *data,
        logy: bool = True,
        fit: tuple[float | None, float | None] | bool | None = None,
        fit_amplitude: bool = False,
        title: str | None = None,
        ax=None,
    ):
        """One or more time series, optionally with an exponential growth-rate fit.

        Parameters
        ----------
        *data:
            Names or arrays with the single dimension ``t``, e.g. ``"en_phi"``.
        logy:
            Logarithmic value axis.
        fit:
            Time window ``(t0, t1)`` of an exponential fit per series (``None`` for an open
            end), or ``True`` for the whole series. Rates are in ``result.fit_results``.
        fit_amplitude:
            The series is quadratic in an amplitude (e.g. an energy); fit the amplitude's rate.
        title:
            Axes title; the first series' label by default.
        ax:
            Draw into these axes instead of a new figure.
        """
        from struphy.diagnostics.plotting import GrowthFit, plot_timeseries

        if not data:
            raise TypeError("timeseries() needs at least one name or array")
        series = [self._array(item) for item in data]
        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries(series, ax=ax, logy=logy, fit=growth, title=title, run_label=self._label(series))

    def slice(
        self,
        data,
        *,
        x: str | None = None,
        y: str | None = None,
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        select: dict | None = None,
        isel: dict | None = None,
        vmin=None,
        vmax=None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        ax=None,
    ):
        """A two-dimensional color plot of one slice.

        Parameters
        ----------
        data:
            Name or array; select all but two dimensions, here or with ``select``/``isel``.
        x, y:
            Displayed dimensions, e.g. ``x="e1", y="v1"``; inferred for two-dimensional data.
        coords:
            ``"physical"`` draws on the mapped coordinates of ``plane`` instead of logical ones.
        select, isel:
            Selections by nearest coordinate value or by index, e.g. ``isel={"t": -1}``.
        """
        from struphy.diagnostics.plotting import plot_slice

        array = self._array(data)
        return plot_slice(
            array,
            view=self._view(x, y, "t", coords, plane, select, isel),
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            equal_aspect=equal_aspect,
            title=title,
            run_label=self._label([array]),
        )

    def panels(
        self,
        data,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        select: dict | None = None,
        isel: dict | None = None,
        nrows: int = 3,
        ncols: int = 4,
        shared_clim: bool = True,
        title: str | None = None,
    ):
        """Snapshots evenly spread along ``sweep`` (time by default), one panel each."""
        from struphy.diagnostics.plotting import plot_panels

        array = self._array(data)
        return plot_panels(
            array,
            view=self._view(x, y, sweep, coords, plane, select, isel),
            nrows=nrows,
            ncols=ncols,
            shared_clim=shared_clim,
            title=title,
            run_label=self._label([array]),
        )

    def viewer(
        self,
        data,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        select: dict | None = None,
        isel: dict | None = None,
        vmin=None,
        vmax=None,
    ):
        """An interactive slice viewer with one slider per non-displayed dimension.

        Call ``.show()`` on the result; keep it alive so that the sliders stay connected.
        """
        from struphy.diagnostics.plotting import InteractiveSliceViewer

        array = self._array(data)
        return InteractiveSliceViewer(
            array,
            view=self._view(x, y, sweep, coords, plane, select, isel),
            vmin=vmin,
            vmax=vmax,
            run_label=self._label([array]),
        )

    def animation(
        self,
        data,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        select: dict | None = None,
        isel: dict | None = None,
        interval: int = 100,
        step: int = 1,
        vmin=None,
        vmax=None,
    ):
        """A Matplotlib animation along ``sweep``."""
        from struphy.diagnostics.plotting import animate_slices

        return animate_slices(
            self._array(data),
            view=self._view(x, y, sweep, coords, plane, select, isel),
            interval=interval,
            step=step,
            vmin=vmin,
            vmax=vmax,
        )

    def frames(
        self,
        data,
        directory,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        select: dict | None = None,
        isel: dict | None = None,
        step: int = 1,
        prefix: str = "frame",
        dpi: int = 110,
    ) -> list[str]:
        """Write the slices along ``sweep`` as numbered PNG files; returns their paths."""
        from struphy.diagnostics.plotting import save_frames

        return save_frames(
            self._array(data),
            directory,
            view=self._view(x, y, sweep, coords, plane, select, isel),
            step=step,
            prefix=prefix,
            dpi=dpi,
        )

    def orbits(self, species: str | None = None, *, max_markers: int = 200, show_paths: bool | None = None, ax=None):
        """Three-dimensional trajectories of the saved markers of ``species``."""
        from struphy.diagnostics.plotting import plot_marker_trajectories

        available = tuple(self._output.orbits)
        if species is None:
            if len(available) != 1:
                raise ValueError(f"choose a species from {available}")
            species = available[0]
        return plot_marker_trajectories(
            self._output.orbits[species], ax=ax, max_markers=max_markers, show_paths=show_paths
        )

    def equilibrium(self, ax=None):
        """Radial equilibrium profiles, from the geometry written at the start of the run."""
        from struphy.diagnostics.plotting import plot_equilibrium_profile

        return plot_equilibrium_profile(self._output.path_out, ax=ax)


class OutputAnalysis:
    """Quantitative diagnostics of a run, as ``out.analysis.<quantity>(...)``."""

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, data) -> xr.DataArray:
        return self._output[data] if isinstance(data, str) else data

    def growth_rate(self, data, *, window: tuple[float | None, float | None] = (None, None), amplitude: bool = False):
        """Fit ``exp(rate * t + intercept)`` to a time series within ``window``.

        With ``amplitude=True`` the series is quadratic in an amplitude (e.g. an energy) and the
        amplitude's rate is returned. Returns a ``FitResult`` (``.rate``, ``.intercept``,
        ``.time``, ``.fitted``), or ``None`` with fewer than two valid samples.
        """
        from struphy.diagnostics.plotting import GrowthFit, growth_rate

        return growth_rate(self._array(data), GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude))

    def drift(self, data, *, ref=None) -> xr.DataArray:
        """Signed deviation of a time series from ``ref`` or from its first sample."""
        from struphy.diagnostics.plotting import drift

        return drift(self._array(data), ref=ref)

    def relative_error(self, data, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Absolute relative deviation of a time series from ``ref`` or from its first sample."""
        from struphy.diagnostics.plotting import relative_error

        return relative_error(self._array(data), ref=ref, skip_first=skip_first)

    def dispersion(
        self, field, *, component: int = 0, slice_at: tuple = (None, 0, 0), physical: bool = False, **kwargs
    ):
        """Space-time power spectrum of a field and fitted dispersion branches.

        The spectrum is computed in normalized time. See
        :func:`struphy.diagnostics.diagn_tools.power_spectrum_2d` for ``slice_at``, the fit options
        and ``do_plot``. Returns ``(omega, kvec, spectrum, coeffs)``.
        """
        from struphy.diagnostics.diagn_tools import power_spectrum_2d

        if isinstance(field, str):
            run = (
                self._output if self._output.time_units == "normalized" else self._output.with_time_units("normalized")
            )
            field = run[field]
        elif field.t.attrs.get("units") == "s":
            raise ValueError("pass the field by name, or take it from out.with_time_units('normalized')")
        return power_spectrum_2d(field, component=component, slice_at=slice_at, physical=physical, **kwargs)
