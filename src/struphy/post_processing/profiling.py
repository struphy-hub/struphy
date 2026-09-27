"""Timing regions of a run recorded with ``sim.run(profiling_activated=True)``."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import xarray as xr

SESSION_REGION = "scope_profiler.session"
METRICS = ("calls", "total_time", "mean_time", "min_time", "max_time", "fraction")


class Profile:
    """The timing regions of one run, read from its ``profiling_data.h5``.

    Obtain it from :attr:`Output.profile`. Every region is a named span of the run, such as a
    propagator (``prop: <name>``), a compiled kernel (``kernel: <name>``) or a linear solve.
    Regions nest, so a region's time includes the regions it calls: times of different regions
    must not be added up. All times are in seconds.

    With several MPI ranks, ``calls`` and ``total_time`` are averages over the ranks, and
    ``mean_time``, ``min_time`` and ``max_time`` refer to single calls on any rank.
    """

    def __init__(self, path, label: str = ""):
        self.path = Path(path)
        self.label = label or self.path.parent.name

    def __repr__(self):
        return f"{type(self).__name__}({str(self.path)!r})"

    @cached_property
    def results(self):
        """The full ``scope_profiler`` results of this run, for anything not covered here."""
        from scope_profiler import read_h5

        return read_h5(self.path)

    @property
    def num_ranks(self) -> int:
        return int(self.results.num_ranks)

    def summary(
        self, *, prefix: str | None = None, ranks=None, sort_by: str = "total_time", top: int | None = None
    ) -> xr.Dataset:
        """Statistics of every region, as a Dataset along the dimension ``region``.

        Parameters
        ----------
        prefix:
            Keep only regions whose name starts with this, e.g. ``"kernel:"`` or ``"prop:"``.
        ranks:
            An MPI rank or a list of ranks to use; all ranks by default.
        sort_by:
            The variable to sort the regions by, largest first: ``calls``, ``total_time``,
            ``mean_time``, ``min_time``, ``max_time`` or ``fraction``.
        top:
            Keep only this many regions after sorting.

        The variables are ``calls`` and ``total_time`` per rank, ``mean_time``, ``min_time`` and
        ``max_time`` per call, and ``fraction``, the ``total_time`` as a share of the whole run.
        """
        if sort_by not in METRICS:
            raise ValueError(f"cannot sort by {sort_by!r}; choose one of {METRICS}")
        from scope_profiler import collect_region_statistics

        statistics = collect_region_statistics(self.results, ranks=ranks)["files"][0]
        regions = statistics["region_statistics"]
        n_ranks = len(next(iter(regions.values()))["per_rank"]) if regions else 1
        n_ranks = max(n_ranks, 1)

        def number(value):
            return np.nan if value is None else float(value)

        names = [name for name in regions if prefix is None or name.startswith(prefix)]
        rows = {
            "calls": [number(regions[name]["count"]) / n_ranks for name in names],
            "total_time": [number(regions[name]["total_duration_seconds"]) / n_ranks for name in names],
            "mean_time": [number(regions[name]["average_duration_seconds"]) for name in names],
            "min_time": [number(regions[name]["min_duration_seconds"]) for name in names],
            "max_time": [number(regions[name]["max_duration_seconds"]) for name in names],
        }
        session = regions.get(SESSION_REGION)
        run_time = number(session["total_duration_seconds"]) / n_ranks if session else statistics["total_time_seconds"]
        rows["fraction"] = [value / run_time for value in rows["total_time"]]

        out = xr.Dataset(
            {key: ("region", np.asarray(values)) for key, values in rows.items()}, coords={"region": names}
        )
        out["calls"].attrs["label"] = "calls per rank"
        for key in ("total_time", "mean_time", "min_time", "max_time"):
            out[key].attrs.update(label=key.replace("_", " "), units="s")
        out["fraction"].attrs["label"] = "share of the run"
        out.attrs.update(run=self.label, num_ranks=n_ranks, run_time=run_time)
        # stable and NaN-last, so regions that tie keep the order in which the profiler lists them
        order = np.argsort(-np.nan_to_num(out[sort_by].values, nan=-np.inf), kind="stable")
        return out.isel(region=order[:top])

    def table(self, **kwargs) -> str:
        """The :meth:`summary` as a text table; use ``print(profile.table(top=10))``."""
        summary = self.summary(**kwargs)
        width = max((len(name) for name in summary.region.values), default=6)
        lines = [
            f"Profile: {self.label} ({summary.attrs['num_ranks']} rank(s), {summary.attrs['run_time']:.3f} s)",
            "",
            f"{'Region':<{width}}  {'Calls':>8}  {'Total [s]':>10}  {'Mean [ms]':>10}  {'Share':>7}",
            f"{'-' * width}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 7}",
        ]
        for name in summary.region.values:
            row = summary.sel(region=name)
            lines.append(
                f"{name:<{width}}  {row.calls.item():>8.0f}  {row.total_time.item():>10.3f}  "
                f"{row.mean_time.item() * 1e3:>10.3f}  {row.fraction.item() * 100:>6.1f}%"
            )
        return "\n".join(lines)

    def compare(self, *others, metric: str = "total_time", prefix: str | None = None, ranks=None) -> xr.DataArray:
        """One :meth:`summary` variable of this and other runs, side by side.

        ``others`` are :class:`Profile` objects, or anything with a ``profile`` attribute such as an
        :class:`Output`. The result has the dimensions ``region`` and ``run``; a region missing
        from a run is NaN. Regions are sorted by their largest value over the runs. For a ratio
        between two runs, divide ``result.isel(run=1) / result.isel(run=0)``.
        """
        if metric not in METRICS:
            raise ValueError(f"cannot compare {metric!r}; choose one of {METRICS}")
        profiles = [self] + [getattr(other, "profile", other) for other in others]
        labels = [profile.label for profile in profiles]
        if len(set(labels)) < len(labels):
            labels = [f"{profile.label} [{profile.path.parent.name}]" for profile in profiles]
        arrays = [profile.summary(prefix=prefix, ranks=ranks)[metric] for profile in profiles]
        out = xr.concat(arrays, dim=xr.DataArray(labels, dims="run", name="run"), join="outer")
        out.attrs = dict(arrays[0].attrs)
        out.name = metric
        order = np.argsort(-np.nan_to_num(out.max("run").values, nan=-np.inf), kind="stable")
        return out.isel(region=order)
