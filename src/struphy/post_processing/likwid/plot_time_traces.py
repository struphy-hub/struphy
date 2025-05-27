import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_time_vs_duration(
    path,
    output_path,
    groups_include=["*"],
    groups_skip=[],
):
    """
    Plot start times versus durations for all profiling regions from all MPI ranks,
    with each region using the same color across ranks.

    Parameters
    ----------
    path: str
        Path to the file where profiling data is saved.
    """
    with open(path, "rb") as file:
        profiling_data = pickle.load(file)

    # Create a color map for each unique region
    unique_regions = set(region_name for rank_data in profiling_data["rank_data"].values() for region_name in rank_data)
    color_map = {region_name: plt.cm.tab10(i % 10) for i, region_name in enumerate(unique_regions)}
    plt.figure(figsize=(10, 6))

    # Iterate through each rank's data
    for rank_name, rank_data in profiling_data["rank_data"].items():
        for region_name, info in rank_data.items():
            skip_region = False
            for reg in ["main", "setup", "initialize"]:
                if reg in region_name or reg == region_name:
                    skip_region = True
            if skip_region:
                continue
            start_times = info["start_times"]
            durations = info["durations"]

            # Use the color from the color_map for each region
            color = color_map[region_name]
            plt.plot(
                start_times, durations, "x-", color=color, label=f"{region_name}" if rank_name == "rank_0" else None
            )
    xmax = max(start_times)
    x = 0
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
    avg_durations = [np.mean(region_durations[r]) for r in regions]
    min_durations = [np.min(region_durations[r]) for r in regions]
    max_durations = [np.max(region_durations[r]) for r in regions]
    yerr = [[avg - min_ for avg, min_ in zip(avg_durations, min_durations)],
            [max_ - avg for avg, max_ in zip(avg_durations, max_durations)]]

    # Plot bar chart with error bars (min-max spans)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(regions))
    plt.bar(x, avg_durations, yerr=yerr, capsize=5, color="skyblue", edgecolor="k")
    plt.xticks(x, regions, rotation=45, ha="right")
    plt.ylabel("Duration (s)")
    plt.title("Average Duration per Profiling Region (with Min-Max Span)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save the figure
    figure_path = os.path.join(output_path, "avg_duration_per_region.pdf")
    plt.savefig(figure_path)
    print(f"Saved average duration bar chart to: {figure_path}")

def plot_gantt_chart(
    path,
    output_path,
    groups_include=["*"],
    groups_skip=[],
):
    """
    Plot Gantt chart of profiling regions from all MPI ranks using a grouped bar plot,
    where bars are grouped by region and stacked for different ranks, with each rank having a specific color.

    Parameters
    ----------
    path: str
        Path to the file where profiling data is saved.
    """
    # Load the profiling data from the specified path
    with open(path, "rb") as file:
        profiling_data = pickle.load(file)

    plt.figure(figsize=(12, 8))

    # Collect unique region names and their earliest start times
    region_start_times = {}
    for rank_data in profiling_data["rank_data"].values():
        for region_name, info in rank_data.items():
            first_start_time = np.min(info["start_times"])
            if region_name not in region_start_times or first_start_time < region_start_times[region_name]:
                region_start_times[region_name] = first_start_time

    # Sort region names by their earliest start time
    region_names = sorted(region_start_times, key=region_start_times.get)

    # Generate a color map for each rank
    rank_names = list(profiling_data["rank_data"].keys())
    num_ranks = len(rank_names)
    # color_map = matplotlib.cm.get_cmap('tab10')  # Use 'tab10' colormap
    color_map = plt.get_cmap("tab10")
    # Parameters for spacing
    bar_height = 0.1  # Height of each bar for a rank
    rank_spacing = 0.1  # Vertical spacing between bars for different ranks
    region_spacing = 0.5  # Vertical spacing between groups for different regions

    y_positions = []  # To store y positions for labels
    region_labels = []  # To store labels for the y-axis
    current_y = 0  # Initial y position

    # Iterate through each region in the sorted order
    for region_idx, region_name in enumerate(region_names):
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
    plt.title("Gantt chart of profiling regions (colored by MPI rank)")
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
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(o_path, "sim_1", "profiling_time_trace.pkl"),
        help="Path to the profiling data file (default: o_path from struphy state)",
    )

    args = parser.parse_args()
    path = os.path.abspath(args.path)  # Convert to absolute path

    # Plot the time trace
    plot_time_vs_duration(path, output_path=o_path)
    plot_gantt_chart(path, output_path=o_path)
    plot_avg_duration_bar_chart(path, output_path=o_path,groups_skip=['main'])
