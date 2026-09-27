"""Shared output helpers for the gallery scripts: figure files, profiling exports, metadata.

Every helper writes to the current directory, which is `docs/public/examples/` when a script is
run as described in README.md, and names its files after the script stem.

The scripts also support MPI (`mpirun -n 4 python <script>.py`): the simulation, the
post-processing and the analysis run on every rank, and only rank 0 writes files. The helpers
below take care of that, so a script needs no rank checks of its own.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from struphy import set_logging_level
from struphy.utils.mpi_launch import launched_under_mpi

from scope_profiler import plot_durations as _plot_durations

def plot_durations(*args, metrics=("total",), **kwargs):
    """Export several metrics with scope-profiler's one-metric API."""
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    data_filepath = kwargs.pop("data_filepath", None)
    if not data_filepath or len(metrics) == 1:
        return _plot_durations(*args, metric=metrics[0], data_filepath=data_filepath, **kwargs)

    path = Path(data_filepath)
    payloads = []
    results = []
    for metric in metrics:
        temporary_path = path.with_name(f"{path.stem}.{metric}{path.suffix}")
        results.extend(_plot_durations(*args, metric=metric, data_filepath=temporary_path, **kwargs))
        payloads.append(json.loads(temporary_path.read_text()))
        temporary_path.unlink()

    merged = payloads[0]
    merged["bars"] = [bar for payload in payloads for bar in payload["bars"]]
    merged["metrics"] = list(metrics)
    path.write_text(json.dumps(merged))
    return results

# Importing mpi4py initializes MPI. In particular, a serial command run inside
# a Slurm allocation must not initialize MPI merely because mpi4py is present.
# Struphy's launcher detector also supports STRUPHY_MPI=0 for that case.
if launched_under_mpi():
    try:
        from mpi4py import MPI

        _COMM = MPI.COMM_WORLD
    except ImportError:  # a serial install without MPI
        _COMM = None
else:
    _COMM = None

# A gantt bar is one call, not an aggregate, and a run with thousands of steps draws tens of
# thousands of near-identical bars -- heavy enough to hang the tab. Keeping the first
# GANTT_MAX_INTERVALS in time order leaves the setup phase plus several complete step-loop iterations.
GANTT_MAX_INTERVALS = 5000

# Every script imports this module before it runs its simulation. INFO makes Struphy print one block per
# time step (step number, times, wall clock, scalar quantities), which is what one wants to see in a CI log.
set_logging_level(logging.INFO)


def is_root() -> bool:
    """True on MPI rank 0 (and in a serial run): the only process that writes files."""
    return _COMM is None or _COMM.Get_rank() == 0


def barrier() -> None:
    """Wait until every rank got here, e.g. before rank 0 reads a file the ranks wrote together."""
    if _COMM is not None:
        _COMM.Barrier()


def _scientific_ticks(figure) -> None:
    """Label tick values as 1×10⁻⁶ instead of Plotly's default SI prefixes (µ, n, k, M, ...).

    Only fills in `exponentformat` where the example has not chosen one itself.
    """

    def default(obj) -> None:
        if obj.exponentformat is None:
            obj.exponentformat = "power"

    figure.for_each_xaxis(default)
    figure.for_each_yaxis(default)
    figure.for_each_scene(lambda scene: [default(axis) for axis in (scene.xaxis, scene.yaxis, scene.zaxis)])
    figure.for_each_coloraxis(lambda coloraxis: default(coloraxis.colorbar))
    for trace in figure.data:
        # Touching `colorbar` on a trace that has none would make Plotly draw an empty one.
        marker = getattr(trace, "marker", None)
        if marker is not None and marker.showscale:
            default(marker.colorbar)
        if "colorbar" in trace._valid_props and trace.showscale is not False and trace.type != "scatter":
            default(trace.colorbar)


def save_figure(
    figure,
    stem: str,
    *,
    width: int = 1100,
    height: int = 650,
    suffix: str = "",
    static_z=None,
    static_data=None,
    static_active=None,
) -> None:
    """Write static PNG, Plotly JSON and standalone HTML versions of a figure.

    The PNG is also copied to `../images/examples/` when that directory exists, as the gallery thumbnail.

    An animation that starts at t = 0 can still have an informative static image: `static_z`
    replaces the heatmap in the first trace for the PNG only, and `static_data` (a list of traces,
    e.g. one frame's `data`) gives the traces of the PNG (with the figure's layout, its slider set to `static_active`).
    """
    if not is_root():
        return
    _scientific_ticks(figure)
    png_path = Path(f"{stem}{suffix}.png")
    json_path = Path(f"{stem}{suffix}.plotly.json")
    html_path = Path(f"{stem}{suffix}.html")
    if static_data is not None:
        still = go.Figure(data=static_data, layout=figure.layout)
        if static_active is not None and still.layout.sliders:
            still.layout.sliders[0].active = static_active  # the slider shows the still's frame
        still.write_image(png_path, width=width, height=height, scale=2)
    elif static_z is not None:
        initial_z = figure.data[0].z
        initial_active = figure.layout.sliders[0].active if figure.layout.sliders else None
        figure.data[0].z = static_z
        if static_active is not None and figure.layout.sliders:
            figure.layout.sliders[0].active = static_active
        figure.write_image(png_path, width=width, height=height, scale=2)
        figure.data[0].z = initial_z
        if figure.layout.sliders:
            figure.layout.sliders[0].active = initial_active
    else:
        figure.write_image(png_path, width=width, height=height, scale=2)
    figure.write_json(json_path, pretty=False)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        config={"responsive": True},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {json_path.resolve()}")
    print(f"Saved {html_path.resolve()}")
    # The gallery, the model pages and the static fallbacks of the example page (shown on phones and
    # without JavaScript, in place of the interactive plot) read the PNG from the images directory,
    # under the same name as here. Both directories are generated and untracked.
    images = Path("../images/examples")
    if images.is_dir():
        shutil.copyfile(png_path, images / png_path.name)


def save_extra_figure(
    figure,
    stem: str,
    key: str,
    *,
    alt: str,
    caption: str,
    static_z=None,
    static_active=None,
) -> dict:
    """Save an additional figure of an example and return its entry for the `figures` metadata list.

    Writes `<stem>-<key>.png/.plotly.json/.html` here (the PNG is copied to
    `../images/examples/` by `save_figure`).
    The example page shows every entry of `figures` below its main
    figure: pass the list to `merge_metadata(stem, figures=[...])`.
    """
    save_figure(
        figure,
        stem,
        suffix=f"-{key}",
        static_z=static_z,
        static_active=static_active,
    )
    return {
        "key": key,
        "interactive": f"/examples/{stem}-{key}.plotly.json",
        "thumbnail": f"/images/examples/{stem}-{key}.png",
        "alt": alt,
        "caption": caption,
    }


def _finish_layout(figure, title, xaxis_title, yaxis_title, **layout):
    layout.setdefault("margin", {"l": 70, "r": 30, "t": 80, "b": 60})
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        autosize=True,
        **layout,
    )
    return figure


def heatmap_figure(
    data,
    *,
    x: str,
    y: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colorbar_title: str = "",
    colorscale: str = "Viridis",
    zmin=None,
    zmax=None,
    x_values=None,
    y_values=None,
):
    """A Plotly heatmap of a two-dimensional xarray array, with `x` and `y` naming its dimensions.

    `x_values` and `y_values` replace the coordinates of those dimensions, e.g. to plot a logical
    coordinate in physical length.
    """
    x_values = data[x].values if x_values is None else x_values
    y_values = data[y].values if y_values is None else y_values
    figure = go.Figure(
        go.Heatmap(
            z=data.transpose(y, x).values,
            x=x_values,
            y=y_values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar_title},
        )
    )
    return _finish_layout(figure, title, xaxis_title, yaxis_title)


def space_time_figure(
    data,
    *,
    space: str,
    title: str,
    colorbar_title: str,
    xaxis_title: str = "x [a.u.]",
    x_values=None,
    colorscale: str = "RdBu",
):
    """A space-time map of a field with dimensions `(t, space)`: space along x, time up, colors symmetric about zero."""
    limit = float(abs(data).max())
    return heatmap_figure(
        data,
        x=space,
        y="t",
        x_values=x_values,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="t [a.u.]",
        colorbar_title=colorbar_title,
        colorscale=colorscale,
        zmin=-limit,
        zmax=limit,
    )


def heatmap_movie(
    data,
    *,
    x: str,
    y: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colorbar_title: str = "",
    sweep: str = "t",
    colorscale: str = "Viridis",
    zmin=0.0,
    zmax=None,
    x_values=None,
    y_values=None,
    max_frames: int = 150,
):
    """An animated Plotly heatmap of a three-dimensional xarray array, one frame per `sweep` value.

    A frame per saved step would embed tens of megabytes in the page, so at most `max_frames`
    evenly spaced frames are kept. Returns `(figure, static_z)`, where `static_z` is a
    well-developed frame from the middle of the sweep, for `save_figure(..., static_z=...)`.
    """
    x_values = data[x].values if x_values is None else x_values
    y_values = data[y].values if y_values is None else y_values
    frames_data = data.transpose(sweep, y, x).values
    labels = data[sweep].values
    picks = np.linspace(0, len(labels) - 1, min(max_frames, len(labels)), dtype=int)

    def heatmap(z, **extra):
        return go.Heatmap(
            z=z,
            x=x_values,
            y=y_values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            **extra,
        )

    frames = [go.Frame(name=f"{labels[i]:.1f}", data=[heatmap(frames_data[i])]) for i in picks]
    figure = go.Figure(
        data=[heatmap(frames_data[0], colorbar={"title": colorbar_title})],
        frames=frames,
    )
    _finish_layout(
        figure,
        title,
        xaxis_title,
        yaxis_title,
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.28,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": f"{sweep} = "},
            },
        ],
    )
    return figure, frames_data[len(frames_data) // 2]


def merge_metadata(stem: str, **fields) -> Path:
    """Add result fields to `<stem>.metadata.json`, keeping the structural fields already there."""
    path = Path(f"{stem}.metadata.json")
    if not is_root():
        return path
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(fields)
    # NaN and Infinity are not valid JSON: Python writes them anyway, and the site build then fails to
    # parse the file. A non-finite result is a broken run, so stop here and name it.
    broken = [key for key, value in metadata.items() if isinstance(value, float) and not np.isfinite(value)]
    if broken:
        raise RuntimeError(f"Non-finite values in the metadata of {stem}: {', '.join(broken)}; refusing to publish the run")
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved {path.resolve()}")
    return path


def export_profiling(sim, stem: str) -> dict:
    """Export scope-profiler data of a run made with `profiling_activated=True`.

    Writes the raw HDF5 plus the plot-data JSON (durations, gantt, region statistics) that the
    example page renders as native Plotly figures. Returns the metadata fields that point to them,
    ready for `merge_metadata`.
    """

    from scope_profiler import plot_gantt, read_h5, write_region_statistics_json

    barrier()  # the ranks write the profiling file together
    profile_h5_path = Path(f"{stem}-profile.h5")
    durations_path = Path(f"{stem}-durations.json")
    gantt_path = Path(f"{stem}-gantt.json")
    region_stats_path = Path(f"{stem}-region-stats.json")
    fields = {
        "profilingData": f"/examples/{profile_h5_path.name}",
        "profilingDurations": f"/examples/{durations_path.name}",
        "profilingGantt": f"/examples/{gantt_path.name}",
        "profilingRegionStats": f"/examples/{region_stats_path.name}",
    }
    if not is_root():
        return fields

    profile_reader = read_h5(sim.profiling_filepath)
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    # `stack_children` splits each bar into the region's own time plus one segment per region it
    # calls, which is what the page's durations chart stacks; it only decomposes total/avg.
    # `sort_by` fixes the region order the chart draws, biggest total first.
    plot_durations(
        [profile_reader],
        ranks=[0],
        metrics=("total", "avg"),
        sort_by="total",
        stack_children=True,
        data_filepath=durations_path,
        data_format="json",
        verbose=False,
    )
    # The stacked export is a dense region x segment grid, but a call graph is sparse (~90% zeros
    # for regions that never call each other); the chart reads a missing pair as null, so dropping
    # them cuts the file ~10x with no change on the page.
    durations_payload = json.loads(durations_path.read_text())
    durations_payload["bars"] = [bar for bar in durations_payload["bars"] if bar["value_seconds"]]
    durations_path.write_text(json.dumps(durations_payload))

    plot_gantt(
        [profile_reader],
        ranks=[0],
        data_filepath=gantt_path,
        data_format="json",
        verbose=False,
    )
    write_region_statistics_json([profile_reader], region_stats_path, ranks=[0])
    # Keep the run's recorded host alongside its time and rank count for the gallery summary.
    # Read it from the profile, since exports can be regenerated on a different machine.
    region_stats_payload = json.loads(region_stats_path.read_text())
    region_stats_payload["files"][0]["hostname"] = profile_reader.metadata.get("hostname")
    region_stats_path.write_text(json.dumps(region_stats_payload))

    gantt_payload = json.loads(gantt_path.read_text())
    if len(gantt_payload["intervals"]) > GANTT_MAX_INTERVALS:
        gantt_payload["intervals"] = sorted(gantt_payload["intervals"], key=lambda c: c["start_seconds"])[
            :GANTT_MAX_INTERVALS
        ]
        gantt_path.write_text(json.dumps(gantt_payload))

    print(f"Saved {profile_h5_path.resolve()}")
    print(f"Saved {durations_path.resolve()}, {gantt_path.resolve()}, {region_stats_path.resolve()}")
    return fields

