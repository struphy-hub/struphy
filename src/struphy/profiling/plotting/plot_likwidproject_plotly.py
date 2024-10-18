# Standard modules
import argparse
import glob
import math
import os
import random
import re

import hardware_dicts as hwd
import likwid_parser as lp
import matplotlib.pyplot as plt
import maxplotlylib as mply
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# My modules
import roofline_plotter as rp


def clean_string(string_in):
    return re.sub(r"[^\w\s-]", "", string_in.replace(" ", "_"))


def pad_numbers(input_string):
    # Split the string into parts
    parts = re.split(r"(:)", input_string)  # This will keep the colons in the list

    # Function to pad with zeros to at least four digits
    def pad_with_zeros(s):
        if s.isdigit():
            return s.zfill(4)
        return s

    # Apply padding function to each part
    padded_parts = [pad_with_zeros(part) for part in parts]

    # Join the parts back into a single string
    output_string = "".join(padded_parts)

    return output_string


def get_job_name(project, simulation_name_type):
    if simulation_name_type == "simulation_name":
        job_name = project.name
    elif simulation_name_type == "clone_configuration":
        job_name = project.get_clone_configuration()
    elif simulation_name_type == "mpi_configuration":
        job_name = project.get_mpi_configuration()
    elif simulation_name_type == "node_configuration":
        job_name = project.get_node_configuration()
    else:
        print("Incorrect simulation_name_type", simulation_name_type)
        exit()
    return job_name


def skip_group(group):
    if group in [
        "model.integrate",
        "self.derham.clone_comm.allreduce",
        "self._pusher",
        "self.derham.clone_comm.Barrier",
    ]:  # or '_' in group:
        return True
    if "pusher_" in group:
        return True
    if "_pusher" in group:
        return True
    if "self._pusher" in group:
        return True
    if group[0] == "_":
        return True
    if "_barrier" in group:
        return True
    if "initialize_" in group:
        return True
    if "derham" in group:
        return True
    if "model_class" in group:
        return True
    return False


def plot_roofline_data(
    projects,
    output_path,
    group=None,
    column_name="Sum",
    xmin=0.001,
    xmax=100,
    ymin=100,
    ymax=10_000_000,
    title="",
    theoretical_max_bandwidth_GBps=200,
    theoretical_max_gflops=5530,
    sorting_key="group",  # options: 'simulation_name', 'group'
):

    fig = go.Figure()

    x0 = 1e-10
    x1 = 1e10
    y0 = 1e-5
    y1 = 1e6
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[theoretical_max_gflops, theoretical_max_gflops],
            mode="lines",
            line=dict(dash="dash"),
            name=f"Theoretical max ({theoretical_max_gflops:.0f} GFLOP/s)",
        )
    )

    for frac in [0.01]:
        gflops = theoretical_max_gflops * frac
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[gflops, gflops],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{frac*100}% of theoretical ({gflops:.0f} GFLOP/s)",
            )
        )

    roofs_memory_bound = []

    bandwidth_GBps = theoretical_max_bandwidth_GBps

    operational_intensity_FLOPpMB = [x0, x1]
    max_performance_GFLOP = [
        oi * bandwidth_GBps for oi in operational_intensity_FLOPpMB
    ]
    data = {
        "operational_intensity_FLOPpMB": operational_intensity_FLOPpMB,
        "max_performance_GFLOP": max_performance_GFLOP,
        "label": f"{bandwidth_GBps:.1f} GB/s",
    }
    roofs_memory_bound.append(data)

    for roof in roofs_memory_bound:
        fig.add_trace(
            go.Scatter(
                x=roof["operational_intensity_FLOPpMB"],
                y=roof["max_performance_GFLOP"],
                mode="lines",
                name=roof["label"],
            )
        )
        pass

    # Read the data paths and add the data to the point in the roofline plot
    data = []
    for project in projects:
        if group is None:
            groups = project.get_likwid_groups()
        for group in groups:

            imax = project.get_maximum_id(
                "DP [MFLOP/s] STAT",
                group=group,
                table="Metric STAT",
                column=column_name,
            )
            if imax == None:
                print("Result missing")
                continue

            runtime = project.get_value(
                "Runtime (RDTSC) [s] STAT",
                likwid_output_id=imax,
                group=group,
                table="Metric STAT",
                column="Avg",
            )

            # if split_by_simulation_label:
            #     group_data["job_name"] = group.split("_")[0]
            # else:
            #     group_data["job_name"] = job_name
            # print(project.num_mpi)
            point_dict = {
                "simulation_name": project.name + "_" + group,  #
                # "simulation_collection": lp.pad_numbers(
                #     data_path.split("/")[-1]
                # ),  # "MPI scan"
                "description": f"<b>{project.name}</b><br>{group}<br>Runtime: {runtime} s",  # + project.parameters.replace('\n','<br>')
                "dp_GFLOPps": project.get_value(
                    "DP [MFLOP/s] STAT",
                    likwid_output_id=imax,
                    group=group,
                    table="Metric STAT",
                    column=column_name,
                )
                * 1e-3,
                "bandwidth_MBps": project.get_value(
                    "Memory bandwidth [MBytes/s] STAT",
                    likwid_output_id=imax,
                    group=group,
                    table="Metric STAT",
                    column=column_name,
                ),
                "num_mpi": project.num_mpi,
                "group": group,
            }
            point_dict["operational_intensity_FLOPpB"] = (
                1e3 * point_dict["dp_GFLOPps"]
            ) / point_dict["bandwidth_MBps"]
            data.append(point_dict)

    data = sorted(data, key=lambda x: x[sorting_key])
    df = pd.DataFrame(data)

    for simulation_name, group in df.groupby(sorting_key):
        # print('group')
        # print(group)
        # Sort data to ensure the lines connect correctly
        sorted_group = group.sort_values("simulation_name")
        fig.add_trace(
            go.Scatter(
                x=sorted_group["operational_intensity_FLOPpB"],
                y=sorted_group["dp_GFLOPps"],
                mode="markers+lines",
                text=sorted_group["description"],  # Custom text for each point
                name=f"{simulation_name}",  # Name displayed in the legend
            )
        )

    xtick_values = [10**t for t in range(int(math.log10(x0)), 1 + int(math.log10(x1)))]
    ytick_values = [10**t for t in range(int(math.log10(y0)), 1 + int(math.log10(y1)))]

    fig.update_xaxes(
        type="log",  # Ensure the x-axis is logarithmic
        range=[np.log10(xmin), np.log10(xmax)],
        title="Operational intensity (FLOP/Byte)",
        tickvals=xtick_values,  # Set where ticks appear
        ticktext=[str(t) for t in xtick_values],
        # ticktext=[f'$10^{{{int(np.log10(t))}}}$' for t in xtick_values]  # Set tick labels
    )

    fig.update_yaxes(
        type="log",  # Ensure the x-axis is logarithmic
        range=[np.log10(ymin), np.log10(ymax)],
        title="Performance (GFLOP/s)",
        tickvals=ytick_values,  # Set where ticks appear
        ticktext=[str(t) for t in ytick_values],
    )
    # Set all markers to the same size, e.g., 10
    mply.format_axes(fig)
    mply.format_font(fig)
    mply.format_grid(fig)
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(title=f"{title}")
    #

    file_path_html = f"{output_path}/{project.name}_roofline.html"
    file_path_pdf = f"{output_path}/{project.name}_roofline.pdf"

    # Save the figure as an HTML file
    fig.write_html(file_path_html, include_mathjax="cdn")
    mply.format_size(fig)  # ,width=2000,height=800)
    fig.write_image(file_path_pdf)
    print(f"open {file_path_html}")


def plot_bars(
    metric,
    projects,
    output_path,
    data_path=None,
    data_paths=None,
    groups=None,
    column_name="Sum",
    procs_per_clone="any",
    xvalname="job_name",
    skip_groups=True,
    simulation_name_type="simulation_name",  # simulation_name | clone_configuration | mpi_tasks
    title="",
    split_by_simulation_label=True,
):

    if groups == None:
        groupname = "all_groups"
    else:
        groupname = "_".join(groups)

    data = []
    for project in projects:
        if groups is None:
            groups = project.get_likwid_groups()

        job_name = get_job_name(project, simulation_name_type)
        for group in groups:
            group_data = {
                "project_name": project.name,
                # "job_name":group.split('_')[0],
                # "job_name": job_name, #project.get_clone_configuration(),  # project.name,
                "group": group,
                "val": project.get_maximum(metric, group=group, column=column_name),
            }
            if split_by_simulation_label:
                group_data["job_name"] = group.split("_")[0]
            else:
                group_data["job_name"] = job_name
            data.append(group_data)

    data = sorted(data, key=lambda x: x["project_name"])
    df = pd.DataFrame(data)
    # Add the relative value column
    df["relative_val"] = df.groupby("group")["val"].transform(lambda x: x / x.iloc[0])
    unique_groups = df["group"].unique()

    # tab20 colormap
    # colormap = plt.colormaps.get_cmap("tab20")
    # group_color_map = {group: colormap(i) for i, group in enumerate(unique_groups)}
    # Generate a color for each group
    n_colors = len(unique_groups)
    colors = plt.cm.turbo(np.linspace(0, 1, n_colors))
    random.shuffle(colors)
    # Create a color map for each group
    group_color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

    # Convert RGBA to Plotly color format
    df["color"] = df["group"].map(
        lambda group: f"rgba({int(group_color_map[group][0] * 255)}, "
        f"{int(group_color_map[group][1] * 255)}, "
        f"{int(group_color_map[group][2] * 255)}, "
        f"{group_color_map[group][3]})"
    )
    # print(df)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[xvalname],
            y=df["val"],
            text=df["group"],
            textposition="auto",
            marker=dict(color=df["color"]),
        )
    )

    fig.update_layout(
        # xaxis_title='Job name',
        yaxis_title=f"{metric.replace(' STAT','')} ({column_name})",
        showlegend=False,
    )
    fig.update_layout(title=f"{title}")
    mply.format_axes(fig)
    mply.format_font(fig)
    mply.format_size(fig)

    # print(groups)

    sanitized_metric = clean_string(metric)
    sanitized_metric = sanitized_metric.replace("/", "p").replace(" ", "_")
    column_name = clean_string(column_name)
    # fig_dir = f"{output_path}/barplots/{groupname}/{column_name}"

    file_path_html = f"{output_path}/{project.name}_barplot_{sanitized_metric}_{column_name}_{groupname}.html"
    file_path_pdf = f"{output_path}/{project.name}_barplot_{sanitized_metric}_{column_name}_{groupname}.pdf"

    fig.write_html(file_path_html, include_mathjax="cdn")
    fig.write_image(file_path_pdf)
    print(f"open {file_path_html}")


def plot_correlation(
    metric1,
    metric2,
    projects,
    output_path,
    invert_y=False,
    group="model.integrate",
    column_name="Avg",
    procs_per_clone="any",
    title="",
    simulation_name_type="project_name",
):

    data = []

    for project in projects:
        groups = project.get_likwid_groups()
        for group in groups:
            # print('group',group)
            # if not 'model.inte' in group:
            #     #print('skip')
            #     continue

            if metric1 == "mpi":
                val1 = project.num_mpi
            else:
                val1 = project.get_maximum(metric1, group=group, column=column_name)
            # print('column_name',column_name, metric2)

            val2 = project.get_maximum(metric2, group=group, column=column_name)
            runtime = project.get_maximum(
                "Runtime (RDTSC) [s] STAT", group=group, column=column_name
            )

            data.append(
                {
                    "simulation_name": project.name,  #
                    "description": f"<b>{group}</b><br><b>{project.name}</b><br>Runtime: {runtime} s",
                    "job_name": project.get_clone_configuration(),  # project.name,
                    "metric1": val1,
                    "metric2": val2,
                    "group": group,
                }
            )

    fig = go.Figure()
    data = sorted(data, key=lambda x: x["simulation_name"])
    df = pd.DataFrame(data)

    xmin = 1e9
    ymin = 1
    xmax = 1
    ymax = 1
    # Group data by color and add line plots
    for simulation_collection, group in df.groupby("group"):
        # Sort data to ensure the lines connect correctly
        sorted_group = group.sort_values("simulation_name")

        sorted_group["metric2_relative"] = sorted_group.groupby("group")[
            "metric2"
        ].transform(lambda x: x / x.iloc[0])

        if invert_y:
            # Plot the speedup
            sorted_group["metric2_relative"] = 1.0 / sorted_group["metric2_relative"]

        xmin = min(xmin, min(sorted_group["metric1"]))
        xmax = max(xmax, max(sorted_group["metric1"]))
        # ymax = max(xmax, max(sorted_group['metric2_speedup']))
        fig.add_trace(
            go.Scatter(
                x=sorted_group["metric1"],
                y=sorted_group["metric2_relative"],
                mode="markers+lines",
                text=sorted_group["description"],  # Custom text for each point
                name=f"{simulation_collection}",  # Name displayed in the legend
            )
        )

    x_data = [xmin, xmax]
    ymin = 1
    ymax = ymin + (xmax - xmin) / xmin
    y_data = [ymin, ymax]
    if metric1 == "mpi":
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                name="Ideal speedup",
                line=dict(width=4, color="red"),
            )
        )

    mply.format_axes(fig)
    mply.format_font(fig)
    # mply.format_size(fig)
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(title=f"{title}")

    fig.update_layout(
        # xaxis_title='Job name',
        xaxis_title=f"MPI tasks (#)",
        yaxis_title=f"Speedup (relative)",
        showlegend=True,
        xaxis_tickformat=".1f",
        yaxis_tickformat=".1f",
    )

    fig_name = f"{project.name}_{clean_string(metric1)}_{clean_string(metric2)}_{clean_string(column_name)}".replace(
        "/", "p"
    ).replace(
        " ", "_"
    )

    file_path_html = f"{output_path}/{fig_name}.html"
    file_path_pdf = f"{output_path}/{fig_name}.pdf"

    fig.write_html(file_path_html, include_mathjax="cdn")
    fig.write_image(file_path_pdf)
    print(f"open {file_path_html}")


def plot_loadbalance(project, metric, output_path):

    data = []

    groups = project.get_likwid_groups()

    for group in groups:
        for node_name in project.get_columns(table="Metric", group=group):
            data.append(
                {
                    "node_name": pad_numbers(node_name),
                    "group": group,
                    "val": project.get_maximum(
                        metric, group=group, column=node_name, table="Metric"
                    ),
                }
            )

    data = sorted(data, key=lambda x: x["node_name"])
    df = pd.DataFrame(data)

    colormap = plt.colormaps.get_cmap("tab20")
    # print(df['group'].unique())
    # Generate a color for each group
    unique_groups = df["group"].unique()
    group_color_map = {group: colormap(i) for i, group in enumerate(unique_groups)}

    # Convert RGBA to Plotly color format
    df["color"] = df["group"].map(
        lambda group: f"rgba({int(group_color_map[group][0] * 255)}, "
        f"{int(group_color_map[group][1] * 255)}, "
        f"{int(group_color_map[group][2] * 255)}, "
        f"{group_color_map[group][3]})"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["node_name"],
            y=df["val"],
            text=df["group"],
            textposition="auto",
            marker=dict(color=df["color"]),
        )
    )

    # Sanitize the file path to remove non-standard characters
    sanitized_metric = clean_string(metric)

    title = f"{project.name}, {project.get_clone_configuration()}"
    fig.update_layout(
        yaxis_title=f"{metric.replace(' STAT','')}",
        showlegend=False,
        title=title,
    )

    mply.format_axes(fig)
    mply.format_font(fig)

    file_path_html = f"{output_path}/{project.name}_loadbalance_{sanitized_metric}.html"
    file_path_pdf = f"{output_path}/{project.name}_loadbalance_{sanitized_metric}.pdf"

    fig.write_html(file_path_html, include_mathjax="cdn")
    fig.write_image(file_path_pdf)

    print(f"open {file_path_html}")


def plot_socket_cores(
    projects,
    output_path,
    node_name="raven_login",
    split_char=None,
    # project_name="_temp_project",
    # collection_name="_temp_collection",
    title=None,
    # save_path="figures/pinned_cores.html",
    procs_per_clone="any",
):

    for project in projects:

        num_cores_per_socket = len(hwd.node_dict[node_name]["socket1"].split(" "))

        # Constants
        NODE_WIDTH = 150
        NODE_HEIGHT = 5.0

        SOCKET_WIDTH = NODE_WIDTH * 0.9
        SOCKET_HEIGHT = NODE_HEIGHT * 0.5
        THREADGROUP_WIDTH = NODE_WIDTH / (num_cores_per_socket / 2)

        shapes = []
        annotations = []

        annotations.append(
            dict(
                x=NODE_WIDTH * 0.5,
                y=NODE_HEIGHT * 0.8,
                text=f"{project.name}",
                showarrow=False,
                font=dict(color="Black", size=36),
                textangle=0,
            )
        )
        for inode, node in enumerate(project.nodes):
            # print(f"{node = }")
            socket0 = {
                int(s): 0 for s in hwd.node_dict[node_name]["socket0"].split(" ")
            }
            socket1 = {
                int(s): 0 for s in hwd.node_dict[node_name]["socket1"].split(" ")
            }
            sockets = {0: socket0, 1: socket1}

            for thread in project.threads:
                if thread.split(":")[0].strip() == node:
                    core = int(thread.split(":")[2].strip())
                    if core in socket0:
                        socket0[core] += 1
                    elif core in socket1:
                        socket1[core] += 1

            col_dict = {0: "gray", 1: "green", 2: "orange", 3: "red"}
            dist = 2.0

            xorigin = 0.0  # + inode * num_cores_per_socket * 1.25 * dist
            yorigin = 0.0 - inode * NODE_HEIGHT

            # Add a rectangle around the whole node
            x0 = xorigin - 3.0 * dist
            x1 = xorigin + NODE_WIDTH + 3.0 * dist

            xmid = x0 + 0.5 * (x1 - x0)
            y0 = yorigin - 0.4
            y1 = yorigin + 3.6
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="Red", width=3, dash="dash"),
                    fillcolor="salmon",
                )
            )

            annotations.append(
                dict(
                    x=xmid,
                    y=yorigin + 3.7,
                    text=f"Node: {node}",
                    showarrow=False,
                    font=dict(color="red", size=20),
                    textangle=0,
                )
            )

            x0 = xorigin - 2.0 * dist
            x1 = xorigin + NODE_WIDTH + 2.0 * dist

            xmid = x0 + 0.5 * (x1 - x0)
            y0 = yorigin - 0.3
            y1 = yorigin + 1.3
            for isocket in range(2):
                shapes.append(
                    dict(
                        type="rect",
                        x0=x0,
                        y0=y0 + isocket * 2.0,
                        x1=x1,
                        y1=y1 + isocket * 2.0,
                        line=dict(color="Blue", width=3),
                        fillcolor="whitesmoke",
                    )
                )
                # Add socket name
                annotations.append(
                    dict(
                        x=xmid,
                        y=yorigin + 1.4 if isocket == 0 else yorigin + 3.4,
                        text=f"Socket {isocket}",
                        showarrow=False,
                        font=dict(color="Blue", size=20),
                        textangle=0,
                    )
                )

                for threadgroup in range(int(num_cores_per_socket / 2)):
                    xa = xorigin + (threadgroup + 0.1) * THREADGROUP_WIDTH
                    xb = xorigin + (threadgroup + 0.9) * THREADGROUP_WIDTH
                    shapes.append(
                        dict(
                            type="rect",
                            x0=xa,
                            y0=yorigin + isocket * 2.0,
                            x1=xb,
                            y1=yorigin + 1 + isocket * 2.0,
                            line=dict(color="Black", width=2),
                            fillcolor="Black",
                        )
                    )

                for i, (place, used) in enumerate(sockets[isocket].items()):
                    xa = xorigin + (i + 0.15) * THREADGROUP_WIDTH * 0.5
                    xb = xorigin + (i + 0.85) * THREADGROUP_WIDTH * 0.5
                    xm = xa + 0.5 * (xb - xa)
                    shapes.append(
                        dict(
                            type="rect",
                            x0=xa,
                            y0=yorigin + isocket * 2.0,
                            x1=xb,
                            y1=yorigin + 1 + isocket * 2.0,
                            line=dict(color="Black", width=2),
                            fillcolor=col_dict[used],
                        )
                    )

                    annotations.append(
                        dict(
                            x=xm,
                            y=yorigin + 1.01 + isocket * 2.0,
                            text=str(place),
                            showarrow=False,
                            font=dict(color="red"),
                            textangle=90,
                            yanchor="bottom",
                        )
                    )

        fig = go.Figure()
        # fig.update_xaxes(range=[- 3.5 * dist, len(nodes) * num_cores_per_socket * 1.25 * dist])
        fig.update_xaxes(range=[-NODE_WIDTH * 0.1, NODE_WIDTH * 1.1], autorange=False)
        fig.update_yaxes(
            range=[-NODE_HEIGHT * ((len(project.nodes) - 1) + 0.1), 0.9 * NODE_HEIGHT],
            fixedrange=False,
        )
        # fig.update_yaxes(range=[-NODE_HEIGHT * 5, 0.9 * NODE_HEIGHT],fixedrange=False)
        # fig.update_yaxes(range=[0, 4],fixedrange=False)
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            # showlegend=False,
            plot_bgcolor="snow",
            width=1800,
            height=1000 * len(project.nodes),
        )

        fig.update_layout(
            title={
                "text": title,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(
                size=12,
            ),
        )

        # Add all shapes and annotations at once
        fig.update_layout(shapes=shapes, annotations=annotations)
        fig.update_layout(dragmode="pan")

        file_path_html = f"{output_path}/{project.name}_pinning.html"
        file_path_pdf = f"{output_path}/{project.name}_pinning.pdf"

        fig.write_html(file_path_html, include_mathjax="cdn")
        fig.write_image(file_path_pdf)
        print(f"open {file_path_html}")


def plot_files(
    projects,
    output_path,
    title="Collection",
    procs_per_clone="any",
    simulation_name_type="simulation_name",
    plots=[
        "plot_socket_cores",
        "plot_speedup",
        "plot_bars",
        "plot_loadbalance",
        "plot_roofline_data",
    ],
):
    metrics = [  #
        "Runtime (RDTSC) [s] STAT",
        # "Runtime unhalted [s] STAT",
        # "Clock [MHz] STAT",
        # "CPI STAT",
        # "Energy [J] STAT",
        # "Power [W] STAT",
        # "Energy DRAM [J] STAT",
        # "Power DRAM [W] STAT",
        # "DP [MFLOP/s] STAT",
        # "AVX DP [MFLOP/s] STAT",
        # "Packed [MUOPS/s] STAT",
        # "Scalar [MUOPS/s] STAT",
        # "Memory read bandwidth [MBytes/s] STAT",
        # "Memory read data volume [GBytes] STAT",
        # "Memory write bandwidth [MBytes/s] STAT",
        # "Memory write data volume [GBytes] STAT",
        # "Memory bandwidth [MBytes/s] STAT",
        # "Memory data volume [GBytes] STAT",
        # "Operational intensity [FLOP/Byte] STAT",
        # "Operational intensity STAT",
        # "Vectorization ratio [%] STAT",
    ]
    if "plot_socket_cores" in plots:
        plot_socket_cores(
            projects=projects,
            output_path=output_path,
            node_name="raven_login",
            title=None,
        )
    # Plot loadbalance
    for project in projects:

        for metric in [
            "Runtime (RDTSC) [s]",
            "DP [MFLOP/s]",
            # 'Memory bandwidth [MBytes/s]',
        ]:
            if "plot_loadbalance" in plots:
                plot_loadbalance(
                    project=project, metric=metric, output_path=output_path
                )
                pass

    if "plot_bars" in plots:
        for metric in metrics:
            for column_name in [
                # 'Sum',
                # 'Max',
                # 'Min',
                "Avg"
            ]:
                plot_bars(
                    metric=metric,
                    projects=projects,
                    output_path=output_path,
                    groups=None,
                    # project_name=project_name,
                    # collection_name=collection_name,
                    column_name=column_name,
                    procs_per_clone=procs_per_clone,
                    skip_groups=True,
                    title=title,
                    simulation_name_type=simulation_name_type,
                )

    if "plot_roofline_data" in plots:
        plot_roofline_data(
            projects,
            output_path=output_path,
            group=None,
            xmin=0.01,
            xmax=100,
            ymin=0.1,
            ymax=1e4,
            title=title,
        )
    if "plot_speedup" in plots:
        metric1 = "mpi"
        metric2 = "Runtime (RDTSC) [s] STAT"
        plot_correlation(
            metric1=metric1,
            metric2=metric2,
            projects=projects,
            output_path=output_path,
            invert_y=True,
            group="model.integrate",
            column_name="Avg",
            procs_per_clone="any",
            title="",
        )


def load_projects(data_paths, procs_per_clone="any"):
    projects = []
    for data_path in data_paths:
        for path in glob.glob(data_path):
            print(f"Reading {path}")
            if path[-1] == "/":
                sim = path.split("/")[-2]
            else:
                sim = path.split("/")[-1]
            project = lp.Project(
                name=lp.pad_numbers(sim),
                path=path,
                likwid_out_naming="struphy*.out",
                read_project=True,
            )
            if (procs_per_clone != "any") and (
                procs_per_clone != project.procs_per_clone
            ):
                print(
                    f"Incorrect number of procs_per_clone: {project.procs_per_clone = } {procs_per_clone = }"
                )
                continue
            project.read_project()
            if not project.simulation_finished:
                print("Project not finished")
                continue
            projects.append(project)
    return projects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the plot files script with a given directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the data directories (space-separated, supports wildcards)",
    )
    parser.add_argument(
        "--project_name", type=str, default="Testing", help="Name of the project"
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Expand wildcard directories
    expanded_dirs = []
    for d in args.dir:
        expanded_dirs.extend(glob.glob(d))

    print(f"Expanded Directory: {expanded_dirs}, Project Name: {args.project_name}")

    # Pass the expanded directories to load_projects
    projects = load_projects(expanded_dirs)

    procs_per_clone = "any"

    print(f"# Plotting simulation: {args.project_name}")
    title = "Standard likwid profiling"
    plot_files(
        projects=projects,
        output_path=args.output,
        title=title,
        plots=[
            # 'plot_socket_cores',
            "plot_speedup",
            # 'plot_bars',
            'plot_loadbalance',
            "plot_roofline_data",
        ],
    )
    print("done")
