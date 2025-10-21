import glob
import pickle

import pandas as pd
import yaml

from struphy.utils.arrays import xp


def sort_by_num_threads(bm):
    sorted_arrays = {}
    for filename, data in bm.items():
        # print(data)
        num_threads = data["num_threads"]
        # del data['num_threads']
        for key, value in data.items():
            if key not in sorted_arrays:
                sorted_arrays[key] = {"num_threads": [], "values": []}
            sorted_arrays[key]["num_threads"].append(num_threads)
            sorted_arrays[key]["values"].append(value)

    for key, value in sorted_arrays.items():
        sorted_indices = sorted(
            range(len(value["num_threads"])),
            key=lambda k: value["num_threads"][k],
        )
        sorted_arrays[key]["num_threads"] = [value["num_threads"][i] for i in sorted_indices]
        sorted_arrays[key]["values"] = [value["values"][i] for i in sorted_indices]

    return sorted_arrays


def read_pickle(filename):
    with open(filename, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def convert_kb_auto(kb_str):
    """
    Converts kilobytes to megabytes or gigabytes automatically.

    Args:
    kb_str (str): The number of kilobytes as a string.

    Returns:
    str: The converted value with the appropriate unit.
    """
    try:
        kb = float(kb_str[:-2])
    except ValueError:
        raise ValueError("Invalid input. Please provide a valid number.")

    if kb < 1024:
        return f"{kb:.0f} kB"
    elif kb < 1024 * 1024:
        return f"{kb / 1024:.0f} MB"
    else:
        return f"{kb / (1024 * 1024):.0f} GB"


def asciitable2df(table):
    # Split the table into lines
    lines = table.strip().split("\n")

    # Extract header and data rows
    header = [col.strip() for col in lines[1].split("|")[1:-1]]
    data = [[col.strip() for col in line.split("|")[1:-1]] for line in lines[3:-1]]

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert data types to numeric (excluding 'Metric' column)
    df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors="coerce")

    return df


def read_likwid_output(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        table_started = False
        table_horizontal_lines = 0

        tables = []
        for iline, line in enumerate(lines):
            if "+-" in line:
                table_horizontal_lines += 1
                table_started = True
                if table_horizontal_lines == 1:
                    table_startline = iline
                if table_horizontal_lines == 3:
                    table_endline = iline
                    table_started = False
                    table_horizontal_lines = 0
                    table = "".join(lines[table_startline : table_endline + 1])
                    tables.append(asciitable2df(table))
    return tables


def get_metric_value(df, metric="DP [MFLOP/s] STAT", column_name="Sum"):
    # Filter the DataFrame based on the specified metric
    filtered_df = df[df["Metric"] == metric]
    # print(filtered_df)
    # Check if the metric exists in the DataFrame
    if not filtered_df.empty:
        # Retrieve the 'Sum' value for the specified metric
        sum_value = filtered_df.iloc[0][column_name]
        # print(f"Sum value for {metric}: {sum_value}")
    else:
        print(f"Metric '{metric}' not found in the DataFrame.")
        sum_value = None
    return sum_value


def add_plot_flop(
    mfig,
    gflops,
    label=None,
    linestyle="-",
    color=None,
    theoretical_max_gflops=3072.0,
    xmax=1e3,
):
    if label == None:
        legend_label = f"{round(gflops)} GFLOP/s, {round(gflops / theoretical_max_gflops * 100, 2)} % of theoretical"
    else:
        legend_label = (
            f"{label}({round(gflops)} GFLOP/s, {round(gflops / theoretical_max_gflops * 100, 2)} % of theoretical)"
        )

    if color == None:
        # line, = mfig.axs.loglog([xmin,xmax],[gflops,gflops],linestyle=linestyle)#,label = legend_label)
        line = mfig.axs.axhline(y=gflops, linestyle=linestyle)
    else:
        # line, = mfig.axs.loglog([xmin,xmax],[gflops,gflops],linestyle=linestyle,color=color)#,label = legend_label)
        line = mfig.axs.axhline(y=gflops, linestyle=linestyle, color=color)
    mfig.axs.text(xmax, gflops, legend_label, color=line.get_color(), fontsize=8)


def add_plot_diagonal(
    mfig,
    bandwidth_GBps,
    label="",
    ymax=1e4,
    operational_intensity_FLOPpMB=xp.arange(0, 1000, 1),
):
    max_performance_GFLOP = operational_intensity_FLOPpMB * bandwidth_GBps
    (line,) = mfig.axs.plot(operational_intensity_FLOPpMB, max_performance_GFLOP)
    # Specify the y-value where you want to place the text
    specific_y = ymax
    # Interpolate to find the corresponding x-value
    specific_x = xp.interp(
        specific_y,
        max_performance_GFLOP,
        operational_intensity_FLOPpMB,
    )
    mfig.axs.text(
        specific_x,
        specific_y,
        f"{round(bandwidth_GBps)} GB/s {label}",
        ha="left",
        va="bottom",
        rotation=45,
        color=line.get_color(),
        fontsize=8,
    )


def get_likwid_value(
    filename,
    df_index,
    metric="Operational intensity [FLOP/Byte]",
    column_name="HWThread 0",
):
    table_dict = {}
    xvec = []
    yvec = []
    dfs = read_likwid_output(filename)
    if len(dfs) >= 3:
        return get_metric_value(dfs[df_index], metric=metric, column_name=column_name)
    return 0


def get_average_val(
    path,
    metric1="Operational intensity [FLOP/Byte]",
    metric2="DP [MFLOP/s]",
    column_name="HWThread 0",
):
    table_dict = {}
    xvec = []
    yvec = []
    for filename in glob.glob(path):
        # print(filename)
        tables = read_likwid_output(filename)
        table_dict[filename.split("/")[-1]] = tables

    for name, dfs in sorted(table_dict.items()):
        # print(name, len(dfs))
        if len(dfs) >= 3:
            # print(dfs[-1])
            x = get_metric_value(dfs[-1], metric=metric1, column_name=column_name)
            y = get_metric_value(dfs[-1], metric=metric2, column_name=column_name) * 1e-3
            # print(x, y)
            label = name.replace("output_", "").replace(".txt", "")
            # print(x, y)
            if x * y == 0:
                break
            xvec.append(x)
            yvec.append(y)
    # print('xvec', xvec, 'yvec', yvec)
    xvec = xp.array(xvec)
    yvec = xp.array(yvec)
    # print('xvec', xvec, 'yvec', yvec)
    return xp.average(xvec), xp.average(yvec), xp.std(xvec), xp.std(yvec)


def get_maximum(path, df_index=-1, metric="DP [MFLOP/s] STAT", column_name="Sum"):
    val = 0
    for filepath in glob.glob(path):
        # print(filepath)
        val = max(
            val,
            get_likwid_value(filepath, df_index=-1, metric=metric, column_name="Sum"),
        )
    return val


def get_roofline_point(path):
    # print(path)
    dp_MFLOPps = get_maximum(
        path,
        df_index=-1,
        metric="DP [MFLOP/s] STAT",
        column_name="Sum",
    )
    dp_GFLOPps = dp_MFLOPps * 1e-3

    bandwidth_MBps = get_maximum(
        path,
        df_index=-1,
        metric="Memory bandwidth [MBytes/s] STAT",
        column_name="Sum",
    )
    operational_intensity_FLOPpB = dp_MFLOPps / bandwidth_MBps

    # linpack_40thread_gflops = 1706.0003
    # label=f'{simulation_name} ({dp_GFLOPps:.1f} GFLOP/s, ${dp_GFLOPps / linpack_40thread_gflops * 100:.2f}$~\% of Linpack 40 threads)'

    filepath = glob.glob(path)[0]
    filepath_yaml = filepath.replace("struphy.out", "parameters.yml")
    filepath_yaml = "/".join(filepath.split("/")[:-1]) + "/parameters.yml"
    # print(filepath)
    # print(filepath_yaml)
    simulation_name = filepath.split("/")[-2]
    with open(filepath_yaml, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
    out_dict = {
        "simulation_name": simulation_name,
        "operational_intensity_FLOPpB": operational_intensity_FLOPpB,
        "dp_GFLOPps": dp_GFLOPps,
    }

    for key, item in parameters["grid"].items():
        out_dict[key] = str(item)
    # description = str(out_dict)
    # print(description)
    desc = f"<b>{simulation_name}</b><br>"
    with open(filepath_yaml, "r") as f:
        desc += "<br>".join(f.readlines())
    out_dict["description"] = desc
    return out_dict
