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
    paths,
    output_path,
    groups_include=["*"],
    groups_skip=[],
    show_spans=False,
):
    """
    Plot start times versus durations for all profiling regions from all MPI ranks,
    with each region using the same color across ranks.

    Parameters
    ----------
    path: str
        Path to the file where profiling data is saved.
    """

    if isinstance(paths, str):
        paths = [paths]

    plt.figure(figsize=(10, 6))
    for path in paths:
        print(f"{path =}")
        with open(path, "rb") as file:
            profiling_data = pickle.load(file)

        # Create a color map for each unique region
        unique_regions = set(
            region_name for rank_data in profiling_data["rank_data"].values() for region_name in rank_data
        )
        color_map = {region_name: plt.cm.tab10(i % 10) for i, region_name in enumerate(unique_regions)}

        # Iterate through each rank's data
        for rank_name, rank_data in profiling_data["rank_data"].items():
            for region_name, info in rank_data.items():
                if plot_region(region_name=region_name, groups_include=groups_include, groups_skip=groups_skip):
                    start_times = info["start_times"]
                    durations = info["durations"]

                    # Use the color from the color_map for each region
                    color = color_map[region_name]
                    label = f"{region_name}" if rank_name == "rank_0" else None
                    if len(paths) > 1:
                        label += f" ({path.split('/')[-2]})"
                    plt.plot(start_times, durations, "x-", color=color, label=label)
        xmax = max(start_times)
        x = 0
        if show_spans:
            while x < xmax:
                xa = x + info["config"]["sample_duration"]
                xb = xa + (info["config"]["sample_interval"] - info["config"]["sample_duration"])
                plt.axvspan(xa, xb, alpha=0.5, color="red", zorder=1)
                x += info["config"]["sample_interval"]

    plt.title("Time vs. Duration for Profiling Regions")
    plt.xlabel("Start Time (s)")
    plt.ylabel("Duration (s)")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # plt.show()
    figure_path = os.path.join(output_path, "time_vs_duration.pdf")
    plt.savefig(figure_path)
    print(f"Saved time trace to:{figure_path}")


def plot_avg_duration_bar_chart(
    path,
    output_path,
    groups_include=["*"],
    groups_skip=[],
):
    with open(path, "rb") as file:
        profiling_data = pickle.load(file)

    region_durations = {}

    # Gather all durations per region across all ranks
    for rank_data in profiling_data["rank_data"].values():
        for region_name, info in rank_data.items():
            if any(skip in region_name for skip in groups_skip):
                continue
            if groups_include != ["*"] and not any(inc in region_name for inc in groups_include):
                continue
            durations = info["durations"]
            region_durations.setdefault(region_name, []).extend(durations)

    # Compute statistics per region
    regions = sorted(region_durations.keys())
    avg_durations = [xp.mean(region_durations[r]) for r in regions]
    min_durations = [xp.min(region_durations[r]) for r in regions]
    max_durations = [xp.max(region_durations[r]) for r in regions]
    yerr = [
        [avg - min_ for avg, min_ in zip(avg_durations, min_durations)],
        [max_ - avg for avg, max_ in zip(avg_durations, max_durations)],
    ]

    # Plot bar chart with error bars (min-max spans)
    plt.figure(figsize=(12, 6))
    x = xp.arange(len(regions))
    plt.bar(x, avg_durations, yerr=yerr, capsize=5, color="skyblue", edgecolor="k")
    plt.yscale("log")
    plt.xticks(x, regions, rotation=45, ha="right")
    plt.ylabel("Duration (s)")
    plt.title("Average Duration per Profiling Region (with Min-Max Span)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save the figure
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


def plot_gantt_chart(
    paths,
    output_path,
    groups_include=["*"],
    groups_skip=[],
    backend="matplotlib",
):
    """
    Plot Gantt chart of profiling regions from all MPI ranks using a grouped bar plot,
    where bars are grouped by region and stacked for different ranks, with each rank having a specific color.

    Parameters
    ----------
    path: str
        Path to the file where profiling data is saved.
    """
    assert backend in ["matplotlib", "plotly"], "backend must be either matplotlib or plotly"

    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        # Load the profiling data from the specified path
        with open(path, "rb") as file:
            profiling_data = pickle.load(file)

        plt.figure(figsize=(12, 8))
        # color_map = matplotlib.cm.get_cmap('tab10')  # Use 'tab10' colormap
        color_map = plt.get_cmap("tab10")

        # Collect unique region names and their earliest start times
        region_start_times = {}
        for rank_data in profiling_data["rank_data"].values():
            for region_name, info in rank_data.items():
                first_start_time = xp.min(info["start_times"])
                if region_name not in region_start_times or first_start_time < region_start_times[region_name]:
                    region_start_times[region_name] = first_start_time

        # Sort region names by their earliest start time
        region_names = sorted(region_start_times, key=region_start_times.get)

        # Generate a color map for each rank
        rank_names = list(profiling_data["rank_data"].keys())
        num_ranks = len(rank_names)

        # Parameters for spacing
        bar_height = 0.1  # Height of each bar for a rank
        rank_spacing = 0.1  # Vertical spacing between bars for different ranks
        region_spacing = 0.5  # Vertical spacing between groups for different regions

        y_positions = []  # To store y positions for labels
        region_labels = []  # To store labels for the y-axis
        current_y = 0  # Initial y position

        # Iterate through each region in the sorted order
        for region_idx, region_name in enumerate(region_names):
            if not plot_region(region_name=region_name, groups_include=groups_include, groups_skip=groups_skip):
                continue
            start_y = current_y  # Starting y position for the first rank in this region

            # Plot bars for each rank for the current region
            for rank_idx, (rank_name, rank_data) in enumerate(profiling_data["rank_data"].items()):
                if region_name in rank_data:
                    info = rank_data[region_name]
                    start_times = info["start_times"]
                    end_times = info["end_times"]
                    durations = end_times - start_times

                    # Calculate the y position for this bar
                    y_position = current_y + rank_idx * rank_spacing
                    # Plot each call as a bar with a specific color for the rank
                    plt.barh(
                        y=y_position,
                        width=durations,
                        left=start_times,
                        color=color_map(rank_idx / num_ranks),
                        alpha=0.6,
                        height=bar_height,
                        label=f"{rank_name}" if region_idx == 0 else None,
                    )

            # Calculate the middle y position for this region's label
            middle_y = start_y + (rank_idx * rank_spacing) / 2
            y_positions.append(middle_y)
            region_labels.append(region_name)

            # Move to the next y position for the next region, adding region spacing
            current_y += (rank_idx + 1) * rank_spacing + region_spacing

        # Customize the plot
        plt.xlim(left=-1)
        plt.yticks(y_positions, region_labels)  # Label the y-axis with region names
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Profiling Regions")
        # plt.title("Gantt chart of profiling regions")
        if num_ranks < 10:
            plt.legend(title="MPI Ranks", loc="upper left")  # Add legend for MPI ranks
        plt.grid(visible=True, linestyle="--", alpha=0.5, axis="x")  # Grid only on x-axis
        plt.tight_layout()
        # plt.show()

        # Save the plot as a PDF file
        figure_path = os.path.join(output_path, "gantt_chart.pdf")
        plt.savefig(figure_path)
        print(f"Saved gantt chart to:{figure_path}")


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
        plot_gantt_chart_plotly(reader, output_path=o_path, groups_include=args.groups, groups_skip=args.groups_skip)

        # Call the plotting functions with the appropriate arguments
        # reader.plot_gantt(show = True)#regions=, filepath=gantt_path, show=args.show)
        # reader.plot_durations(show = True)#regions=regions, filepath=durations_path, show=args.show)

    # paths = [os.path.join(o_path, simulation, "profiling_time_trace.pkl") for simulation in args.simulations]
    # paths = [os.path.join(simulation, "profiling_time_trace.pkl") for simulation in args.simulations]

    # Plot the time trace
    # plot_gantt_chart_plotly(path=paths[0], output_path=o_path, groups_include=args.groups, groups_skip=args.groups_skip)
    # plot_time_vs_duration(paths=paths, output_path=o_path, groups_include=args.groups, groups_skip=args.groups_skip)
    # plot_gantt_chart(paths=paths, output_path=o_path, groups_include=args.groups, groups_skip=args.groups_skip)
