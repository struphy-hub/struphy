import os
import pickle
import re

import cunumpy as xp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scope_profiler.h5reader import ProfilingH5Reader

# pio.kaleido.scope.mathjax = None
import struphy.post_processing.likwid.maxplotlylib as mply


def glob_to_regex(pat: str) -> str:
    # Escape all regex metachars, then convert \* → .* and \? → .
    esc = re.escape(pat)
    return "^" + esc.replace(r"\*", ".*").replace(r"\?", ".") + "$"


def plot_region(region_name, groups_include=["*"], groups_skip=[]):
    from fnmatch import fnmatch

    for pattern in groups_skip:
        if fnmatch(region_name, pattern):
            return False
    for pattern in groups_include:
        if fnmatch(region_name, pattern):
            return True
    return False


def plot_time_vs_duration(
    reader,
    output_path,
    groups_include=["*"],
    groups_skip=[],
    show_spans=False,
):
    """
    Plot start times versus durations for all profiling regions from all MPI ranks,
    using the new _region_dict[region][rank] = RegionData structure.

    Parameters
    ----------
    reader : ProfilingH5Reader
        The profiling reader object with _region_dict.
    """

    import os

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 6))

    # Collect all unique region names for color mapping
    unique_regions = list(reader._region_dict.keys())
    color_map = {region_name: plt.cm.tab10(i % 10) for i, region_name in enumerate(unique_regions)}

    # Iterate over regions and ranks
    for region_name, region in reader._region_dict.items():
        if not plot_region(region_name=region_name, groups_include=groups_include, groups_skip=groups_skip):
            continue

        color = color_map[region_name]

        for rank_idx, rank_data in region.items():
            start_times = rank_data.start_times
            durations = rank_data.end_times - rank_data.start_times

            # Only label the first rank for legend
            label = region_name if rank_idx == 0 else None

            plt.plot(start_times, durations, "x-", color=color, label=label)

            # Optionally show sampling spans
            if show_spans and hasattr(rank_data, "config"):
                x = 0
                xmax = np.max(start_times) if len(start_times) > 0 else 0
                while x < xmax:
                    sample_duration = rank_data.config["sample_duration"]
                    sample_interval = rank_data.config["sample_interval"]
                    plt.axvspan(
                        x + sample_duration,
                        x + sample_interval,
                        alpha=0.5,
                        color="red",
                        zorder=1,
                    )
                    x += sample_interval

    plt.title("Time vs. Duration for Profiling Regions")
    plt.xlabel("Start Time (s)")
    plt.ylabel("Duration (s)")
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    figure_path = os.path.join(output_path, "time_vs_duration.pdf")
    plt.savefig(figure_path)
    print(f"Saved time trace to: {figure_path}")


def plot_avg_duration_bar_chart(
    reader,
    output_path,
    groups_include=["*"],
    groups_skip=[],
):
    """
    Plot average duration per profiling region across all ranks.
    Uses the new data structure:
        reader._region_dict[region][rank] = RegionData(start_times, end_times)
    """

    import os

    import matplotlib.pyplot as plt
    import numpy as np

    region_durations = {}

    # Gather all durations per region across all ranks
    for region_name, region in reader._region_dict.items():
        if any(skip in region_name for skip in groups_skip):
            continue
        if groups_include != ["*"] and not any(inc in region_name for inc in groups_include):
            continue

        for rank_data in region.values():
            durations = rank_data.end_times - rank_data.start_times
            region_durations.setdefault(region_name, []).extend(durations)

    if len(region_durations) == 0:
        print("No regions matched the filter.")
        return

    # Compute statistics per region
    regions = sorted(region_durations.keys())
    avg_durations = [np.mean(region_durations[r]) for r in regions]
    min_durations = [np.min(region_durations[r]) for r in regions]
    max_durations = [np.max(region_durations[r]) for r in regions]

    # Error bars (min-max)
    yerr = [
        [avg - min_ for avg, min_ in zip(avg_durations, min_durations)],
        [max_ - avg for avg, max_ in zip(avg_durations, max_durations)],
    ]

    # ---- Plot ----
    plt.figure(figsize=(12, 6))
    x = np.arange(len(regions))
    plt.bar(x, avg_durations, yerr=yerr, capsize=5, color="skyblue", edgecolor="k")
    plt.yscale("log")
    plt.xticks(x, regions, rotation=45, ha="right")
    plt.ylabel("Duration (s)")
    plt.title("Average Duration per Profiling Region (with Min-Max Span)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # ---- Save ----
    os.makedirs(output_path, exist_ok=True)
    figure_path = os.path.join(output_path, "avg_duration_per_region.pdf")
    plt.savefig(figure_path)
    print(f"Saved average duration bar chart to: {figure_path}")


def plot_gantt_chart_plotly(
    reader: ProfilingH5Reader,
    output_path: str,
    groups_include: list = ["*"],
    groups_skip: list = [],
    show: bool = False,
):
    """
    Plot an interactive Plotly Gantt chart using the new:
        _region_dict[region][rank] = RegionData(start_times, end_times)
    structure.

    Each rank gets its own horizontal lane, only one y-axis label per region,
    and regions are sorted by the earliest start time.
    """

    import os

    import numpy as np
    import plotly.graph_objects as go

    # ---- Compute earliest start time per region ----
    region_start_times = {}
    for region_name, region in reader._region_dict.items():
        earliest = np.inf
        for rank_data in region.values():
            if len(rank_data.start_times) > 0:
                earliest = min(earliest, min(rank_data.start_times))
        region_start_times[region_name] = earliest

    # ---- Sort regions by earliest start time ----
    region_names = sorted(region_start_times, key=region_start_times.get)

    if len(region_names) == 0:
        print("No regions found.")
        return

    # ---- Determine number of ranks ----
    first_region = reader._region_dict[region_names[0]]
    n_ranks = len(first_region.keys())
    rank_names = list(range(n_ranks))

    # ---- Collect bars for Plotly ----
    bars = []
    y_positions = []

    for i, region_name in enumerate(region_names):
        if not plot_region(region_name, groups_include, groups_skip):
            continue

        region = reader._region_dict[region_name]

        for r in rank_names:
            y = i * n_ranks + r
            y_positions.append(y)

            if r not in region:
                continue

            region_data = region[r]
            starts = region_data.start_times
            ends = region_data.end_times
            durations = ends - starts

            for s, e, d in zip(starts, ends, durations):
                bars.append(
                    dict(
                        y=y,
                        region=region_name,
                        rank=r,
                        start=float(s),
                        duration=float(d),
                    )
                )

    if len(bars) == 0:
        print("No regions matched the filter.")
        return

    # ---- Create Plotly figure ----
    fig = go.Figure()

    for bar in bars:
        if "kernel" in bar["region"]:
            color = "blue"
        elif "prop" in bar["region"]:
            color = "red"
        else:
            color = "black"

        fig.add_trace(
            go.Bar(
                x=[bar["duration"]],
                y=[bar["y"]],
                base=[bar["start"]],
                orientation="h",
                marker_color=color,
                hovertemplate=(
                    f"Region: {bar['region']}<br>"
                    f"Rank: {bar['rank']}<br>"
                    f"Start: {bar['start']:.6f}s<br>"
                    f"Duration: {bar['duration']:.6f}s"
                ),
                showlegend=False,
            )
        )

    # ---- Label only first rank of each region ----
    yticks = []
    yticklabels = []
    for i, region_name in enumerate(region_names):
        y_first_rank = i * n_ranks
        yticks.append(y_first_rank)
        yticklabels.append(region_name)

    # ---- Layout ----
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Time (s)",
        yaxis=dict(
            tickmode="array",
            tickvals=yticks,
            ticktext=yticklabels,
        ),
        height=300 + len(y_positions) * 12,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )

    # ---- Formatting helpers ----
    mply.format_axes(fig)
    mply.format_font(fig)
    mply.format_grid(fig)
    mply.format_size(fig, width=1600, height=800)

    # ---- Save ----
    os.makedirs(output_path, exist_ok=True)
    out_html = os.path.join(output_path, "gantt_chart_plotly.html")
    fig.write_html(out_html)

    if show:
        fig.show()

    print(f"Saved interactive gantt chart to: {out_html}")


if __name__ == "__main__":
    import argparse

    import struphy.utils.utils as utils

    # Read struphy state file
    state = utils.read_state()
    o_path = state["o_path"]

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot profiling time trace.")
    # parser.add_argument(
    #     "--path",
    #     type=str,
    #     default=os.path.join(o_path, "sim_1", "profiling_time_trace.pkl"),
    #     help="Path to the profiling data file (default: o_path from struphy state)",
    # )

    parser.add_argument(
        "--simulations",
        nargs="+",
        default=["sim_1"],
        help="One or more simulations to compare (e.g. --simulations sim_1 sim_2)",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        default=["*"],
        help="One or more groups to include (e.g. --groups model.integrate PushEta PushVxB)",
    )

    parser.add_argument(
        "--groups-skip",
        nargs="+",
        default=[],
        help="One or more groups to skip (e.g. --groups-skip model.integrate PushEta PushVxB)",
    )

    args = parser.parse_args()
    # path = os.path.abspath(args.path)  # Convert to absolute path
    # simulations = parser.simulations

    for simulation in args.simulations:
        reader = ProfilingH5Reader(os.path.join(simulation, "profiling_data.h5"))

        # Plot the time traces
        plot_gantt_chart_plotly(
            reader=reader,
            output_path=o_path,
            groups_include=args.groups,
            groups_skip=args.groups_skip,
        )
        plot_avg_duration_bar_chart(
            reader=reader,
            output_path=o_path,
            groups_include=args.groups,
            groups_skip=args.groups_skip,
        )
        plot_time_vs_duration(
            reader,
            output_path=o_path,
            groups_include=args.groups,
            groups_skip=args.groups_skip,
        )
