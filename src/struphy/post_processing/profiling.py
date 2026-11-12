"""Timing regions of a run recorded with ``sim.run(profiling_activated=True)``, and the results of profiling runs.

:class:`Profile` reads the timing regions of one run. :func:`solver_iterations` and :func:`save_profiling_results` are
for the post-processing scripts of profiling cases (``pproc_*.py --save-results``, see the ``submit_*.py`` scripts in
``examples/``): the solver iterations per time step logged by the run, and the scope-profiler report and a summary of
the cost per step written into ``<path_out>/results``, where the profiling driver packages them.
"""

from __future__ import annotations

import json
import os
import re
import statistics
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

    def step_times(self, region: str = "model.integrate") -> np.ndarray:
        """The duration of every call of ``region`` (by default one time step), the maximum over the ranks per call."""
        timings = self.results[region]
        per_rank = [timings[rank].durations for rank in timings.ranks]
        n_calls = min(len(durations) for durations in per_rank)
        return np.max([np.asarray(durations[:n_calls]) for durations in per_rank], axis=0)

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


# the line logged (on rank 0, at level INFO) by the propagators with ``SolverParameters(info=True)``, e.g.
# "Iterations for ShearAlfven: 14"; some propagators log another format or no iterations
_ITERATIONS = re.compile(r"Iterations for (\w+): (\d+)")


def solver_iterations(path_out) -> dict[str, list[int]]:
    """The solver iterations logged by a run, per propagator: ``{name: [iterations of every call]}``.

    The propagators log ``Iterations for <name>: N`` (on rank 0, at logging level INFO) when their solver parameters
    have ``info=True``. They are read from the standard output of the run, ``<path_out>/struphy.out`` (written by the
    profiling driver), else from the log file (``$STRUPHY_LOG_FILE`` or ``<path_out>/struphy.log``) and its rotated
    backups, oldest first; the log keeps only the last few 10 kB, so it may miss the first steps of a long run.

    Parameters
    ----------
    path_out : str | Path
        Output folder of the run.

    Returns
    -------
    dict[str, list[int]]
        The iterations per propagator name, in the order of the calls; empty if nothing was logged.
    """
    path_out = Path(path_out)
    stdout_file = path_out / "struphy.out"
    if stdout_file.is_file():
        files = [stdout_file]
    else:
        log_file = Path(os.environ.get("STRUPHY_LOG_FILE") or path_out / "struphy.log")
        files = [log_file.with_name(f"{log_file.name}.{i}") for i in range(9, 0, -1)] + [log_file]
    iterations: dict[str, list[int]] = {}
    for file in files:
        if file.is_file():
            for name, count in _ITERATIONS.findall(file.read_text(encoding="utf-8", errors="replace")):
                iterations.setdefault(name, []).append(int(count))
    return iterations


def save_profiling_results(path_out, info: dict | None = None, region: str = "model.integrate") -> dict:
    """Write the results of a profiling run into ``<path_out>/results``, where the profiling driver packages them.

    - ``report.html``: the scope-profiler report (call tree, charts, per-rank statistics);
    - ``summary.json``: ``info`` (the case-specific fields, e.g. the grid, the variant, the number of markers), the
      number of MPI ranks, the duration of every call of ``region`` (by default one time step; the maximum over the
      ranks), their median without the first call (which includes one-time work such as allocations and the first
      solver calls), and the solver iterations of :func:`solver_iterations` with their mean per propagator.

    It only reads and writes files: call it on one process (the profiling driver runs the post-processing serially).

    Parameters
    ----------
    path_out : str | Path
        Output folder of a run made with ``sim.run(profiling_activated=True)``.
    info : dict | None
        Further entries of the summary (JSON-serializable); they come first.
    region : str
        The timing region of one step.

    Returns
    -------
    dict
        The summary written into ``summary.json``.
    """
    from scope_profiler import create_html_report

    from struphy.post_processing.output import Output

    out = Output(path_out)
    profile = out.profile
    results_dir = Path(out.path_out) / "results"
    results_dir.mkdir(exist_ok=True)
    create_html_report(profile.results, results_dir / "report.html")

    step_times = profile.step_times(region)
    iterations = solver_iterations(out.path_out)
    summary = {
        **(info or {}),
        "mpi_ranks": profile.num_ranks,
        "region": region,
        "steps": len(step_times),
        "step_times": step_times.tolist(),
        "step_median": float(statistics.median(step_times[1:] if len(step_times) > 1 else step_times)),
        "solver_iterations": iterations,
        "solver_iterations_mean": {name: float(np.mean(counts)) for name, counts in iterations.items()},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
