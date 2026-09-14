"""Standardized plots for post-processed Struphy output.

Every plotter accepts a :class:`~struphy.post_processing.arrays.StruphyArray` and
derives its axis labels, coordinates and units from it, so a correct labeled figure
needs no further arguments.
"""

import logging
import os

import cunumpy as xp
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from struphy.post_processing.arrays import (
    SCALARS_EXCLUDE,
    StruphyArray,
    orbit_columns,
    save_scalars,
    scalar_names,
)

logger = logging.getLogger("struphy")

#: rcParams applied by every plotter, so figures from different scripts match.
STRUPHY_STYLE = {
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": "medium",
    "legend.frameon": False,
    "image.cmap": "viridis",
}


def growth_rate(y: StruphyArray, *, t0: float = None, t1: float = None, of_sqrt: bool = False):
    """Fit an exponential ``exp(gamma*t + b)`` over a time window.

    Parameters
    ----------
    y : StruphyArray
        Signal with a ``t`` dimension. Non-positive and non-finite samples are excluded.
    t0, t1 : float, optional
        Window bounds. Default to the full range.
    of_sqrt : bool
        Fit the growth rate of ``sqrt(y)`` rather than of ``y``. Use this for a
        quadratic quantity such as an energy whose amplitude growth rate is wanted.

    Returns
    -------
    gamma, b, window : float, float, slice
        ``None`` in place of all three if fewer than two usable samples remain.
    """
    t = xp.asarray(y.coord("t"))
    vals = xp.asarray(y)

    lo = float(t[0]) if t0 is None else float(t0)
    hi = float(t[-1]) if t1 is None else float(t1)
    lo, hi = sorted((lo, hi))

    mask = (t >= lo) & (t <= hi) & xp.isfinite(vals) & (vals > 0.0)
    if xp.count_nonzero(mask) < 2:
        mask = xp.isfinite(vals) & (vals > 0.0)
    if xp.count_nonzero(mask) < 2:
        return None, None, None

    idx = xp.nonzero(mask)[0]
    window = slice(int(idx[0]), int(idx[-1]) + 1)

    signal = xp.log(xp.sqrt(vals[window])) if of_sqrt else xp.log(vals[window])
    gamma, b = xp.polyfit(t[window], signal, 1)
    return float(gamma), float(b), window


def drift(y: StruphyArray, *, ref: float = None) -> StruphyArray:
    """Deviation ``y(t) - y_ref`` of a conserved quantity, with ``y_ref = y(0)`` by default."""
    values = xp.asarray(y)
    reference = float(values[0]) if ref is None else float(ref)
    return StruphyArray(
        values - reference,
        dims=y.dims,
        coords=y.coords,
        unit=y.unit,
        label=f"{y.label} drift" if y.label else "drift",
    ).with_coord_units(**y.coord_units)


def relative_error(y: StruphyArray, *, ref: float = None, skip_first: bool = True) -> StruphyArray:
    """Relative deviation ``|y(t) - y_ref| / |y_ref|`` of a conserved quantity.

    The standard energy-conservation diagnostic: a run that conserves energy exactly
    stays at zero, so on a log axis this shows the scheme's error over time.

    Parameters
    ----------
    y : StruphyArray
        Signal with a ``t`` dimension.
    ref : float, optional
        Reference value. Defaults to the first sample.
    skip_first : bool
        Drop ``t = 0``, where the error is identically zero and so cannot be
        drawn on a log axis.
    """
    values = xp.asarray(y)
    reference = float(values[0]) if ref is None else float(ref)
    if reference == 0.0:
        raise ValueError("cannot take a relative error against a reference of zero")

    out = StruphyArray(
        xp.abs(values - reference) / abs(reference),
        dims=y.dims,
        coords=y.coords,
        label=rf"$|\Delta$ {y.label}$| / |${y.label}$(0)|$" if y.label else "relative error",
    ).with_coord_units(**y.coord_units)
    return out.isel(t=slice(1, None)) if skip_first else out


def match_to_grid(values, xgrid):
    """Return ``values`` oriented to match ``xgrid``, transposing if that is what fits."""
    values = xp.asarray(values)
    if values.shape == xgrid.shape:
        return values
    if values.T.shape == xgrid.shape:
        return values.T
    raise ValueError(f"cannot match data shape {values.shape} to grid shape {xgrid.shape}")


def physical_grids(data: StruphyArray, domain, *, axes: str = "XY", fixed_eta=(0.5, 0.0, 0.0)):
    """Map the two logical dimensions of ``data`` through ``domain`` to physical coordinates.

    Parameters
    ----------
    data : StruphyArray
        Must have exactly two ``e<i>`` dimensions.
    domain : Domain
        Struphy domain, called as ``domain(eta1, eta2, eta3, squeeze_out=True)``.
    axes : str
        Which physical plane to return: ``"XY"``, ``"RZ"``, ``"XZ"`` or ``"YZ"``.
    fixed_eta : tuple
        Logical position along the dimension that is not binned.

    Returns
    -------
    xgrid, ygrid, xlabel, ylabel
    """
    logical = [d for d in data.dims if d.startswith("e") and d[1:].isdigit()]
    if len(logical) != 2:
        raise ValueError(f"expected two logical dims, got {logical} from dims {data.dims}")

    nums = [int(d[1]) for d in logical]
    etas = [data.coord(logical[nums.index(ax)]) if ax in nums else fixed_eta[ax - 1] for ax in (1, 2, 3)]

    if axes not in PLANES:
        raise ValueError(f"unknown axes {axes!r}, expected one of {sorted(PLANES)}")

    x, y, z = domain(*etas, squeeze_out=True)
    fx, fy, xlabel, ylabel = PLANES[axes]
    return fx(x, y, z), fy(x, y, z), xlabel, ylabel


def logical_grids(data: StruphyArray):
    """Meshgrid of the two plotted dimensions of ``data``, for plotting without a domain map.

    The plotted dimensions are whatever remains after ``t``, so this covers phase-space
    slices such as ``(e1, v1)`` as well as purely spatial ones.
    """
    plotted = [d for d in data.dims if d != "t"]
    if len(plotted) != 2:
        raise ValueError(f"expected two non-time dims, got {plotted} from dims {data.dims}")
    g0, g1 = (data.coord(d) for d in plotted)
    xgrid, ygrid = xp.meshgrid(g0, g1, indexing="ij")
    return xgrid, ygrid, data.axis_label(plotted[0]), data.axis_label(plotted[1])


#: Physical coordinate planes, as a function of the (X, Y, Z) meshgrids.
PLANES = {
    "XY": (lambda x, y, z: x, lambda x, y, z: y, "X", "Y"),
    "XZ": (lambda x, y, z: x, lambda x, y, z: z, "X", "Z"),
    "YZ": (lambda x, y, z: y, lambda x, y, z: z, "Y", "Z"),
    "RZ": (lambda x, y, z: xp.sqrt(x**2 + y**2), lambda x, y, z: z, "R", "Z"),
}


def field_slice_grids(grids_phy, *, fixed_dim: str = "e3", index: int = 0, plane: str = "XY"):
    """Physical grids for a 2D cut through the 3D evaluation grid.

    Companion to ``StruphyArray.isel(**{fixed_dim: index})``: pass the same ``fixed_dim``
    and ``index`` here to get grids matching the sliced field.

    Parameters
    ----------
    grids_phy : list
        The three 3D physical coordinate arrays from :attr:`PlottingData.grids_phy`.
    fixed_dim : str
        Logical dimension held constant, ``"e1"``, ``"e2"`` or ``"e3"``.
    index : int
        Index along ``fixed_dim``.
    plane : str
        Which physical plane to return, one of :data:`PLANES`.

    Returns
    -------
    xgrid, ygrid, xlabel, ylabel
    """
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}, expected one of {sorted(PLANES)}")

    axis = {"e1": 0, "e2": 1, "e3": 2}
    if fixed_dim not in axis:
        raise ValueError(f"fixed_dim must be one of {sorted(axis)}, got {fixed_dim!r}")

    cut = [slice(None)] * 3
    cut[axis[fixed_dim]] = index
    x, y, z = (xp.asarray(g)[tuple(cut)] for g in grids_phy)

    fx, fy, xlabel, ylabel = PLANES[plane]
    return fx(x, y, z), fy(x, y, z), xlabel, ylabel


class StruphyPlot:
    """Base for the plotters: owns style, figure creation, titling and output.

    Parameters
    ----------
    data : StruphyArray
        The quantity to draw.
    ax : matplotlib Axes, optional
        Draw into an existing axes instead of creating a figure.
    title : str, optional
        Defaults to the quantity's label.
    params : ParamsIn, optional
        When given, run settings are appended to the figure as a suptitle.
    """

    #: Slider-bearing subclasses position their axes manually.
    tight = True

    def __init__(self, data: StruphyArray, *, ax=None, title: str = None, params=None, **kwargs):
        self.data = data
        self.title = title if title is not None else (data.label or "")
        self.params = params
        self.options = kwargs
        self._ax = ax
        self.fig = None
        self.ax = None

    def _make_axes(self, **subplot_kw):
        if self._ax is not None:
            self.ax = self._ax
            self.fig = self._ax.get_figure()
        else:
            self.fig, self.ax = plt.subplots(**subplot_kw)
        return self.fig, self.ax

    def _run_label(self) -> str:
        """One-line summary of the run settings, from the output folder's parameters."""
        if self.params is None:
            return ""
        bits = []
        for obj, attr, name in (
            ("time_opts", "dt", "dt"),
            ("time_opts", "split_algo", "algo"),
            ("grid", "num_elements", "Nel"),
            ("derham_opts", "degree", "p"),
        ):
            holder = getattr(self.params, obj, None)
            value = getattr(holder, attr, None) if holder is not None else None
            if value is not None:
                bits.append(f"{name}={value}")
        return ", ".join(bits)

    def draw(self):
        raise NotImplementedError

    def _finish(self):
        run = self._run_label()
        if run and self.fig is not None and self._ax is None:
            self.fig.suptitle(run, fontsize="small")
        if self.tight and self.fig is not None and self._ax is None:
            self.fig.tight_layout()
        return self

    def plot(self):
        """Draw into the axes and return self."""
        with plt.rc_context(STRUPHY_STYLE):
            self.draw()
            self._finish()
        return self

    def show(self):
        self.plot()
        plt.show()
        return self

    def save(self, path, *, close: bool = False, **kwargs):
        """Draw and write the figure to ``path``.

        Parameters
        ----------
        close : bool
            Close the figure afterwards. Pass this when saving many figures in a
            loop, so that they do not all stay open.
        """
        self.plot()
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        if close:
            self.close()
        return self

    def close(self):
        """Close the figure, unless it was supplied by the caller."""
        if self.fig is not None and self._ax is None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        return self


class FrameSequence:
    """Frame-by-frame output for the plotters that sweep a 2D quantity over time.

    A subclass says what a frame contains (:meth:`_frame_values`, :meth:`_frame_grids`,
    :meth:`_frame_title`, :meth:`_clim`); this draws them into a single reused figure.
    """

    #: keep every ``step``-th time index
    step = 1

    @property
    def frames(self):
        """Time indices that will be drawn."""
        return range(0, self.data.shape[self.data.axis("t")], self.step)

    def _frame_grids(self):
        return self.grids if self.grids is not None else logical_grids(self._frame_values(0))

    def _frame_values(self, index):
        return self.data.isel(t=index)

    def _frame_title(self, index):
        return f"{self.title} at t = {float(self.data.coord('t')[index]):.4e}"

    def _clim(self):
        return self.vmin, self.vmax

    def _setup(self):
        xgrid, ygrid, xlabel, ylabel = self._frame_grids()
        vmin, vmax = self._clim()

        fig, ax = self._make_axes()
        pcm = ax.pcolormesh(
            xgrid,
            ygrid,
            match_to_grid(self._frame_values(0), xgrid),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(pcm, ax=ax, label=self.data.value_label)
        if self.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        return fig, ax, pcm, xgrid

    def _update(self, ax, pcm, xgrid, index):
        pcm.set_array(match_to_grid(self._frame_values(index), xgrid).ravel())
        ax.set_title(self._frame_title(index))

    def save_frames(self, directory, *, prefix="frame", dpi=110):
        """Write one PNG per frame into ``directory``, creating it if needed.

        Returns the list of paths written.
        """
        os.makedirs(directory, exist_ok=True)
        paths = []

        with plt.rc_context(STRUPHY_STYLE):
            fig, ax, pcm, xgrid = self._setup()
            for n, index in enumerate(self.frames):
                self._update(ax, pcm, xgrid, index)
                path = os.path.join(directory, f"{prefix}_{n:04d}.png")
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
            self.fig = None
            self.ax = None

        return paths


class TimeSeriesPlot(StruphyPlot):
    """Scalar quantities against time, optionally log-scaled with a growth-rate fit.

    Parameters
    ----------
    data : StruphyArray or sequence of StruphyArray
        One or more signals sharing a ``t`` dimension.
    logy : bool
        Log-scale the ordinate.
    fit : bool
        Overlay an exponential fit and report the rate in the legend.
    fit_window : tuple, optional
        ``(t0, t1)`` bounds for the fit.
    fit_of_sqrt : bool
        Fit the growth rate of the amplitude rather than of the plotted quantity.
    """

    def __init__(self, data, *, logy=True, fit=False, fit_window=None, fit_of_sqrt=False, **kwargs):
        series = [data] if isinstance(data, StruphyArray) else list(data)
        super().__init__(series[0], **kwargs)
        self.series = series
        self.logy = logy
        self.fit = fit
        self.fit_window = fit_window or (None, None)
        self.fit_of_sqrt = fit_of_sqrt
        #: one ``(gamma, b, window)`` per series once drawn, for reporting the rates
        self.fit_results = []

    def draw(self):
        fig, ax = self._make_axes()

        self.fit_results = []
        for s in self.series:
            (line,) = ax.plot(s.coord("t"), xp.asarray(s), label=s.label or None)

            if not self.fit:
                continue

            gamma, b, window = growth_rate(
                s,
                t0=self.fit_window[0],
                t1=self.fit_window[1],
                of_sqrt=self.fit_of_sqrt,
            )
            self.fit_results.append((gamma, b, window))
            if gamma is None:
                continue

            t_fit = xp.asarray(s.coord("t"))[window]
            scale = 2.0 if self.fit_of_sqrt else 1.0
            ax.plot(
                t_fit,
                xp.exp(scale * (gamma * t_fit + b)),
                "--",
                color=line.get_color(),
                label=rf"fit: $\gamma$ = {gamma:.4e}",
            )
            ax.axvspan(t_fit[0], t_fit[-1], alpha=0.12, color="grey")

        if self.logy:
            ax.set_yscale("log")

        ax.set_xlabel(self.data.axis_label("t"))
        ax.set_ylabel(self.data.value_label)
        ax.set_title(self.title)
        if any(s.label for s in self.series) or self.fit:
            ax.legend()


class ScalarsPlot(StruphyPlot):
    """Every scalar recorded during a run on one axes, over an energy-error panel.

    The overview figure of a run: all tracked scalars against time, plus the
    relative error of the conserved quantity underneath.

    Parameters
    ----------
    scalars : Scalars or dict
        Maps a name to a :class:`~struphy.post_processing.arrays.StruphyArray` over
        ``t``, as :attr:`~struphy.post_processing.post_processing_tools.PlottingData.scalars`
        provides.
    names : sequence of str, optional
        Plot these, in this order. Defaults to all of them.
    exclude : sequence of str
        Names to leave out when ``names`` is not given.
    logy : bool
        Log-scale the ordinate of the main axes.
    relative_to : str, optional
        Divide every series by this one, e.g. ``"en_tot"``. Use it when the scalars
        do not share a unit, so that the common ordinate means something.
    error_panel : str or None
        Scalar whose conservation error is drawn in a panel below, if it was
        recorded. ``None`` suppresses the panel.

    Attributes
    ----------
    error : StruphyArray or None
        The relative error that was drawn, so a script can report its final value.
    """

    tight = False

    def __init__(
        self,
        scalars,
        *,
        names=None,
        exclude=SCALARS_EXCLUDE,
        logy=False,
        relative_to=None,
        error_panel="en_tot",
        **kwargs,
    ):
        self.names = scalar_names(scalars, names=names, exclude=exclude)
        if not self.names:
            raise ValueError(f"no scalars to plot, available: {tuple(scalars.keys())}")

        kwargs.setdefault("title", "Scalars")
        super().__init__(scalars[self.names[0]], **kwargs)

        self.scalars = scalars
        self.logy = logy
        self.relative_to = relative_to
        # a panel cannot be added to an axes the caller supplied
        self.error_panel = error_panel if (error_panel in scalars and self._ax is None) else None
        self.error = None
        self.error_ax = None

    def _series(self, name) -> StruphyArray:
        values = xp.asarray(self.scalars[name])
        if self.relative_to is None:
            return values
        return values / xp.asarray(self.scalars[self.relative_to])

    def _ylabel(self) -> str:
        if self.relative_to is not None:
            return f"quantity / {self.relative_to}"
        units = {self.scalars[n].unit for n in self.names}
        return f"[{units.pop()}]" if len(units) == 1 else "[a.u.]"

    def draw(self):
        if self.error_panel is None:
            fig, ax = self._make_axes()
            ax_err = None
        else:
            fig, (ax, ax_err) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(8.0, 6.5),
                height_ratios=(2, 1),
                layout="constrained",
            )
            self.fig, self.ax = fig, ax
        self.error_ax = ax_err

        for name in self.names:
            ax.plot(self.scalars[name].coord("t"), self._series(name), label=name)

        if self.logy:
            ax.set_yscale("log")
        ax.set_ylabel(self._ylabel())
        ax.set_title(self.title)
        ax.legend(fontsize="small", ncols=max(1, len(self.names) // 6))

        if ax_err is None:
            ax.set_xlabel(self.data.axis_label("t"))
            return

        self.error = relative_error(self.scalars[self.error_panel])
        error_values = xp.asarray(self.error)
        ax_err.plot(self.error.coord("t"), error_values)
        # an exactly conserved quantity has nothing to show on a log axis
        if xp.any(error_values > 0.0):
            ax_err.set_yscale("log")
        ax_err.set_xlabel(self.data.axis_label("t"))
        ax_err.set_ylabel(rf"$|\Delta$ {self.error_panel}$|$ / {self.error_panel}$(0)$", fontsize="small")


class Slice2DPlot(StruphyPlot):
    """A 2D quantity as a pcolormesh, with the colorbar and orientation handled.

    Parameters
    ----------
    data : StruphyArray
        Two-dimensional, or higher with the extra dimensions already selected.
    grids : tuple, optional
        ``(xgrid, ygrid, xlabel, ylabel)`` from :func:`physical_grids` or
        :func:`logical_grids`. Defaults to the logical grids of ``data``.
    equal_aspect : bool
        Force an equal aspect ratio, appropriate for physical coordinates.
    """

    def __init__(self, data, *, grids=None, vmin=None, vmax=None, equal_aspect=False, **kwargs):
        super().__init__(data, **kwargs)
        self.grids = grids if grids is not None else logical_grids(data)
        self.vmin = vmin
        self.vmax = vmax
        self.equal_aspect = equal_aspect

    def draw(self):
        fig, ax = self._make_axes()
        xgrid, ygrid, xlabel, ylabel = self.grids

        values = match_to_grid(self.data, xgrid)
        pcm = ax.pcolormesh(xgrid, ygrid, values, shading="auto", vmin=self.vmin, vmax=self.vmax)
        fig.colorbar(pcm, ax=ax, label=self.data.value_label)

        if self.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self.title)
        ax.grid(False)
        self.mesh = pcm


class PanelGridPlot(StruphyPlot):
    """A grid of 2D snapshots at times spread evenly over the run.

    Replaces the hand-rolled ``nrows``/``ncols``/``time_indices`` loop.

    Parameters
    ----------
    data : StruphyArray
        Must have a ``t`` dimension and two further dimensions.
    nrows, ncols : int
        Panel layout. ``nrows * ncols`` snapshots are shown.
    shared_clim : bool
        Use one colour range across all panels, so panels are comparable.
    """

    tight = False

    def __init__(self, data, *, nrows=3, ncols=4, grids=None, shared_clim=False, equal_aspect=False, **kwargs):
        super().__init__(data, **kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.grids = grids
        self.shared_clim = shared_clim
        self.equal_aspect = equal_aspect

    def draw(self):
        n = self.nrows * self.ncols
        nt = self.data.shape[self.data.axis("t")]
        indices = [int(i / max(n - 1, 1) * (nt - 1)) for i in range(n)]

        t = self.data.coord("t")
        snapshots = [self.data.isel(t=i) for i in indices]
        grids = self.grids if self.grids is not None else logical_grids(snapshots[0])
        xgrid, ygrid, xlabel, ylabel = grids

        vmin = vmax = None
        if self.shared_clim:
            vmin = float(min(xp.nanmin(xp.asarray(s)) for s in snapshots))
            vmax = float(max(xp.nanmax(xp.asarray(s)) for s in snapshots))

        fig, axs = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            figsize=(3.5 * self.ncols, 2.8 * self.nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        self.fig, self.ax = fig, axs

        for panel, (idx, snap) in enumerate(zip(indices, snapshots)):
            ax = axs[panel // self.ncols][panel % self.ncols]
            pcm = ax.pcolormesh(
                xgrid,
                ygrid,
                match_to_grid(snap, xgrid),
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"t = {float(t[idx]):.2e}")
            ax.grid(False)
            if self.equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            if not self.shared_clim:
                fig.colorbar(pcm, ax=ax)

        for ax in axs[-1]:
            ax.set_xlabel(xlabel)
        for row in axs:
            row[0].set_ylabel(ylabel)

        if self.shared_clim:
            fig.colorbar(pcm, ax=list(axs.ravel()), label=self.data.value_label)

        fig.suptitle(" — ".join(filter(None, (self.title, self._run_label()))))


class SliderPlot(FrameSequence, StruphyPlot):
    """A 2D quantity with a time slider, and a second slider for the free axis in 3D.

    The returned object keeps a reference to its sliders; discarding it stops the
    widgets from responding.

    Parameters
    ----------
    data : StruphyArray
        Dimensions ``(t, a, b)`` or ``(t, a, b, c)``; the fourth is swept by the
        second slider.
    slice_dim : str, optional
        Which dimension the second slider steps through. Defaults to the last.
    slice_index : int, optional
        Where the second slider starts, and which cut :meth:`save_frames` writes.
        Defaults to the middle of ``slice_dim``.
    step : int
        Keep every ``step``-th time index when writing frames.
    grids : tuple or callable, optional
        Either fixed ``(xgrid, ygrid, xlabel, ylabel)``, or a function of the slice
        index returning them. Pass a callable when the physical grid depends on where
        the cut is taken, so that it follows the slider instead of going stale.
    """

    tight = False

    def __init__(
        self,
        data,
        *,
        grids=None,
        slice_dim=None,
        slice_index=None,
        step=1,
        vmin=None,
        vmax=None,
        equal_aspect=True,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.grids = grids
        self.step = step
        self.vmin = vmin
        self.vmax = vmax
        self.equal_aspect = equal_aspect
        spatial = [d for d in data.dims if d != "t"]
        self.slice_dim = slice_dim if slice_dim is not None else (spatial[-1] if len(spatial) > 2 else None)
        n_slice = data.shape[data.axis(self.slice_dim)] if self.slice_dim else 0
        self.slice_index = n_slice // 2 if slice_index is None else slice_index
        self.sliders = []

    def _frame_values(self, index):
        """The frame sequence holds the cut fixed and sweeps time, as the time slider does."""
        return self._frame(index, self.slice_index)

    def _frame_grids(self):
        return self._grids_for(self.slice_index)

    def _frame(self, t_index, slice_index):
        frame = self.data.isel(t=t_index)
        if self.slice_dim is not None:
            frame = frame.isel(**{self.slice_dim: slice_index})
        return frame

    def _grids_for(self, slice_index):
        if callable(self.grids):
            return self.grids(slice_index)
        if self.grids is not None:
            return self.grids
        return logical_grids(self._frame(0, slice_index))

    def draw(self):
        nt = self.data.shape[self.data.axis("t")]
        t = self.data.coord("t")

        n_slice = self.data.shape[self.data.axis(self.slice_dim)] if self.slice_dim else 0
        slice_index = self.slice_index if n_slice else 0

        first = self._frame(0, slice_index)
        xgrid, ygrid, xlabel, ylabel = self._grids_for(slice_index)

        fig, ax = self._make_axes()
        fig.subplots_adjust(bottom=0.24 if self.slice_dim else 0.18)

        pcm = ax.pcolormesh(
            xgrid,
            ygrid,
            match_to_grid(first, xgrid),
            shading="auto",
            vmin=self.vmin,
            vmax=self.vmax,
        )
        cbar = fig.colorbar(pcm, ax=ax, label=self.data.value_label)
        if self.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{self.title} at t = {float(t[0]):.4e}")
        ax.grid(False)

        s_time = Slider(fig.add_axes([0.20, 0.08, 0.60, 0.03]), "time", 0, nt - 1, valinit=0, valstep=1)
        self.sliders = [s_time]
        s_slice = None
        if self.slice_dim:
            s_slice = Slider(
                fig.add_axes([0.20, 0.03, 0.60, 0.03]),
                f"{self.slice_dim} index",
                0,
                n_slice - 1,
                valinit=slice_index,
                valstep=1,
            )
            self.sliders.append(s_slice)

        state = {"mesh": pcm, "xgrid": xgrid, "slice": slice_index}

        def update(_):
            ti = int(s_time.val)
            si = int(s_slice.val) if s_slice is not None else 0
            # so that a cut found with the slider is the one save_frames writes
            self.slice_index = si

            # a grid that depends on the cut has to be redrawn, not just refilled
            if callable(self.grids) and si != state["slice"]:
                xg, yg, _, _ = self._grids_for(si)
                state["mesh"].remove()
                state["mesh"] = ax.pcolormesh(
                    xg,
                    yg,
                    match_to_grid(self._frame(ti, si), xg),
                    shading="auto",
                    vmin=self.vmin,
                    vmax=self.vmax,
                )
                state["xgrid"] = xg
                state["slice"] = si
                cbar.update_normal(state["mesh"])

            mesh, grid = state["mesh"], state["xgrid"]
            frame = match_to_grid(self._frame(ti, si), grid)

            mesh.set_array(frame.ravel())
            if self.vmin is None and self.vmax is None:
                mesh.set_clim(float(xp.nanmin(frame)), float(xp.nanmax(frame)))
                cbar.update_normal(mesh)
            ax.set_title(f"{self.title} at t = {float(t[ti]):.4e}")
            fig.canvas.draw_idle()

        for s in self.sliders:
            s.on_changed(update)

        self.mesh = pcm


class AnimationPlot(FrameSequence, StruphyPlot):
    """Sweep a 2D quantity over time, as a matplotlib animation or a frame sequence.

    Parameters
    ----------
    data : StruphyArray
        Dimensions ``(t, a, b)``.
    step : int
        Keep every ``step``-th time index.
    shared_clim : bool
        Hold the colour range fixed across frames, so brightness changes are physical.
    """

    tight = False

    def __init__(
        self, data, *, grids=None, step=1, vmin=None, vmax=None, shared_clim=True, equal_aspect=False, **kwargs
    ):
        super().__init__(data, **kwargs)
        self.grids = grids
        self.step = step
        self.vmin = vmin
        self.vmax = vmax
        self.shared_clim = shared_clim
        self.equal_aspect = equal_aspect

    def _clim(self):
        if self.shared_clim and self.vmin is None and self.vmax is None:
            values = xp.asarray(self.data)
            return float(xp.nanmin(values)), float(xp.nanmax(values))
        return self.vmin, self.vmax

    def draw(self):
        fig, ax, pcm, xgrid = self._setup()
        self._update(ax, pcm, xgrid, 0)
        self.mesh = pcm

    def animate(self, *, interval=100):
        """Return a :class:`matplotlib.animation.FuncAnimation` over the frames."""
        from matplotlib.animation import FuncAnimation

        with plt.rc_context(STRUPHY_STYLE):
            fig, ax, pcm, xgrid = self._setup()
            anim = FuncAnimation(
                fig,
                lambda i: self._update(ax, pcm, xgrid, i),
                frames=list(self.frames),
                interval=interval,
                blit=False,
            )
        self.fig = fig
        return anim


class MarkerTrajectoryPlot(StruphyPlot):
    """Marker positions in 3D over time, coloured by weight, with a time slider.

    Parameters
    ----------
    orbits : StruphyArray
        Dimensions ``(t, marker, attribute)``; columns 0-2 are position, 6 is weight.
    max_markers : int
        Cap on the number of markers drawn.
    show_paths : bool, optional
        Trail each marker's history. Defaults to on for small marker counts.
    """

    tight = False

    def __init__(self, orbits, *, max_markers=200, show_paths=None, **kwargs):
        kwargs.setdefault("title", "Marker trajectories")
        super().__init__(orbits, **kwargs)
        self.max_markers = max_markers
        self.show_paths = show_paths if show_paths is not None else max_markers <= 200
        self.sliders = []

    def draw(self):
        orbs = xp.asarray(self.data)
        n = min(orbs.shape[1], self.max_markers)
        cols = getattr(self.data, "columns", None) or orbit_columns(orbs.shape[-1])

        x, y, z = (orbs[:, :n, i] for i in range(cols["position"].start, cols["position"].stop))
        w = orbs[:, :n, cols["weight"]] if "weight" in cols else None
        nt = x.shape[0]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        self.fig, self.ax = fig, ax
        fig.subplots_adjust(bottom=0.18)

        colouring = {"c": w[0], "cmap": "viridis"} if w is not None else {}
        scatter = ax.scatter(x[0], y[0], z[0], s=8, **colouring)
        lines = (
            [ax.plot(x[:1, j], y[:1, j], z[:1, j], lw=0.8, alpha=0.5)[0] for j in range(n)] if self.show_paths else []
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{self.title} | step 0/{nt - 1}")
        if w is not None:
            fig.colorbar(scatter, ax=ax, label="marker weight")

        slider = Slider(fig.add_axes([0.18, 0.06, 0.65, 0.03]), "time", 0, nt - 1, valinit=0, valstep=1)
        self.sliders = [slider]

        def update(_):
            it = int(slider.val)
            scatter._offsets3d = (x[it], y[it], z[it])
            if w is not None:
                scatter.set_array(w[it])
            for j, line in enumerate(lines):
                line.set_data(x[: it + 1, j], y[: it + 1, j])
                line.set_3d_properties(z[: it + 1, j])
            ax.set_title(f"{self.title} | step {it}/{nt - 1}")
            fig.canvas.draw_idle()

        slider.on_changed(update)


def save_all_scalars(
    scalars,
    directory,
    *,
    names=None,
    exclude=SCALARS_EXCLUDE,
    logy=False,
    params=None,
    table: str = "csv",
    file_format: str = "png",
    dpi: int = 110,
) -> list[str]:
    """Write the standard scalar output of a run: the table, an overview, one figure each.

    Everything a finished run should leave behind for its scalars, in one call.

    Parameters
    ----------
    scalars : Scalars or dict
        Maps a name to a :class:`~struphy.post_processing.arrays.StruphyArray` over ``t``.
    directory : str
        Created if it does not exist.
    names, exclude
        See :func:`~struphy.post_processing.arrays.scalar_names`.
    logy : bool
        Log-scale the ordinate of every figure.
    params : ParamsIn, optional
        Run settings, added to each figure as a suptitle.
    table : str or None
        Format of the per-time-step table, ``"csv"`` or ``"npz"``; ``None`` to skip it.
    file_format : str
        Image format of the figures.

    Returns
    -------
    list of str
        The paths written, the table first.
    """
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        logger.warning("No scalars to save.")
        return []

    os.makedirs(directory, exist_ok=True)
    paths = []

    if table is not None:
        paths.append(save_scalars(scalars, os.path.join(directory, f"scalars.{table}"), names=selected, fmt=table))

    overview = os.path.join(directory, f"scalars.{file_format}")
    ScalarsPlot(scalars, names=selected, logy=logy, params=params).save(overview, dpi=dpi, close=True)
    paths.append(overview)

    for name in selected:
        path = os.path.join(directory, f"{name}.{file_format}")
        TimeSeriesPlot(
            scalars[name],
            logy=logy,
            fit=False,
            title=name,
            params=params,
        ).save(path, dpi=dpi, close=True)
        paths.append(path)

    logger.info(f"Wrote {len(paths)} scalar output files to {directory}")
    return paths


def plot_equilibrium_profile(path_out, *, ax=None):
    """Radial profiles of the equilibrium written to ``geometry.vts``."""
    import pyvista as pv

    equil = pv.read(os.path.join(path_out, "geometry.vts"))
    dims = equil.dimensions
    grid = xp.reshape(equil.points, dims + (3,))
    r = xp.sqrt(grid[:, :, :, 0] ** 2 + grid[:, :, :, 1] ** 2)
    p0 = xp.reshape(equil.point_data["p0"], dims)

    with plt.rc_context(STRUPHY_STYLE):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(r[0, 0, :], p0[0, 0, :], label=r"$p_0$")

        if "n0" in equil.point_data:
            n0 = xp.reshape(equil.point_data["n0"], dims)
            ax.plot(r[0, 0, :], n0[0, 0, :], label=r"$n_0$")
            ax.plot(r[0, 0, :], p0[0, 0, :] / n0[0, 0, :], label=r"$T_0$")

        ax.set_xlabel(r"$R$")
        ax.set_title("Radial equilibrium profiles")
        ax.legend()
    return ax
