import glob
import os
import pickle
import re


def pad_numbers(s, pad_length=5):
    # This function replaces all numbers in the input string with zero-padded numbers
    return re.sub(r"\d+", lambda match: str(int(match.group(0))).zfill(pad_length), s)


def csvtable2dict(table_str):

    table_arrays = table_str.split("\n")
    header = table_arrays[0].split(",")
    data = [
        [col.strip() for col in line.split(",")[0:-1]] for line in table_arrays[1:-1]
    ]

    # Create dict
    ddict = {"header": header}
    for irow, rowdata in enumerate(data):
        metric = data[irow][0]
        ddict[metric] = {}

        for icol in range(1, len(data[irow])):
            try:
                # Attempt to convert the value to float and assign if successful
                ddict[metric][header[icol]] = float(data[irow][icol])
            except ValueError:
                # Handle the case where the conversion fails
                ddict[metric][header[icol]] = data[irow][icol]
    return ddict


def asciitable2dict(table):
    # Split the table into lines
    lines = table.strip().split("\n")

    # Extract header and data rows
    header = [col.strip() for col in lines[1].split("|")[1:-1]]
    data = [[col.strip() for col in line.split("|")[1:-1]] for line in lines[3:-1]]

    # Create dict
    ddict = {"header": header}
    for irow, rowdata in enumerate(data):
        metric = data[irow][0]
        ddict[metric] = {}
        for icol in range(1, len(data[irow])):
            try:
                # Attempt to convert the value to float and assign if successful
                ddict[metric][header[icol]] = float(data[irow][icol])
            except ValueError:
                # Handle the case where the conversion fails
                ddict[metric][header[icol]] = data[irow][icol]
    return ddict


def read_likwid_output(
    filename, likwid_markers=True, finish_line="struphy run finished",
):
    """
    Read and process LIKWID output from a specified file.

    Parameters:
    filename (str): Path to the LIKWID output file.
    likwid_markers (bool, optional): Flag indicating if LIKWID markers are present. Default is True.
    finish_line (str, optional): The line indicating the end of the Struphy run. Default is "struphy run finished".

    Returns:
    dict: A dictionary containing processed LIKWID output data.
    """

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    csv_format = "TABLE,Region" in "".join(lines)
    table_dict = {
        "filename": filename,
        "raw_file_lines": lines,
        "struphy_ended": False,
    }
    if csv_format:
        tables = []
        struphy_ended = False
        region_dict = {"likwid_output": True}

        for iline, line in enumerate(lines):

            # TODO get this from likwid directly
            if "MPI processes:" in line:
                table_dict["mpi_procs"] = int(line.split(":")[-1])

            if finish_line in line:
                struphy_ended = True
                table_dict["struphy_ended"] = True

            if "TABLE" in line:
                table_info = line
                region = table_info.split(",")[1].replace("Region ", "")
                table_type = table_info.split(",")[2].replace("Group 1 ", "")
                group = table_info.split(",")[3]
                tab_length = int(table_info.split(",")[4])
                tab_str = "".join(lines[iline + 1 : iline + 1 + tab_length])
                _tb = csvtable2dict(tab_str)

                _tb["table_type"] = table_type
                tables.append(_tb)
                region_dict["group"] = group

                if len(tables) == 4:
                    table_dict[region] = {
                        "dicts": {
                            "likwid_output": True,
                            "Raw": tables[0],
                            "Raw STAT": tables[1],
                            "Metric": tables[2],
                            "Metric STAT": tables[3],
                        },
                    }

                    tasks = [task for task in tables[0]["header"][2:] if task]
                    nodes = set(task.split(":")[0] for task in tasks)
                    table_dict[region]["nodes"] = {
                        node: {"processor_ids": [], "processor_order": []}
                        for node in nodes
                    }

                    for task in tasks:
                        node, task_id, processor_id = task.split(":")
                        table_dict[region]["nodes"][node]["processor_ids"].append(
                            processor_id,
                        )
                        table_dict[region]["nodes"][node]["processor_order"].append(
                            [task_id, processor_id],
                        )

                    tables = []
    else:
        table_horizontal_lines = 0
        struphy_ended = False
        tables = []
        region = ""

        for iline, line in enumerate(lines):

            # TODO get this from likwid directly
            if "MPI processes:" in line:
                table_dict["mpi_procs"] = int(line.split(":")[-1])

            if finish_line in line:
                struphy_ended = True

            if struphy_ended:
                table_dict["struphy_ended"] = True

                if likwid_markers and "Region:" in line:
                    region = line[8:].strip()
                    # print('region',region)
                # elif "Group:" in line:
                #     region = line[7:].strip()
                #     print('region group',region)

                if len(tables) == 4:
                    # print('TABLES = 4', region)
                    table_dict[region] = {
                        "dicts": {
                            "likwid_output": True,
                            "Raw": tables[0],
                            "Raw STAT": tables[1],
                            "Metric": tables[2],
                            "Metric STAT": tables[3],
                        },
                    }

                    tasks = tables[0]["header"][2:]
                    nodes = set(task.split(":")[0] for task in tasks)
                    table_dict[region]["nodes"] = {
                        node: {"processor_ids": [], "processor_order": []}
                        for node in nodes
                    }

                    for task in tasks:
                        node, task_id, processor_id = task.split(":")
                        table_dict[region]["nodes"][node]["processor_ids"].append(
                            processor_id,
                        )
                        table_dict[region]["nodes"][node]["processor_order"].append(
                            [task_id, processor_id],
                        )

                    tables = []

                if "+-" in line:
                    table_horizontal_lines += 1
                    if table_horizontal_lines == 1:
                        table_startline = iline
                    if table_horizontal_lines == 3:
                        table_endline = iline
                        table_horizontal_lines = 0
                        table = "".join(lines[table_startline : table_endline + 1])
                        tables.append(asciitable2dict(table))
    # print(table_dict['raw_file_lines'])
    # print(table_dict.keys())

    # exit()
    return table_dict


def expand_node_range(range_str):
    nodes = []
    parts = range_str.split(",")
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            nodes.extend(range(start, end + 1))
        else:
            nodes.append(int(part))
    return nodes


def parse_nodelist(nodelist_str):
    match = re.search(r"SLURM_NODELIST=(\w+)(\[(.+)\])?", nodelist_str)
    if not match:
        return []

    prefix = match.group(1)
    range_str = match.group(3)

    if range_str:  # Case with range
        node_numbers = expand_node_range(range_str)
        node_names = [f"{prefix}{num}" for num in node_numbers]
    else:  # Case without range, single node
        node_names = [prefix]

    return node_names


class Project:
    def __init__(
        self,
        name,
        path,
        read_project=True,
        likwid_out_naming="struphy_likwid_*.out",
    ):
        """
        Initialize the Project instance.

        Args:
            name (str): Name of the project.
            path (str): Path to the project directory.
            read_project (bool): Whether to read the project data on initialization.
            likwid_out_naming (str): Pattern for LIKWID output files.
        """

        # TODO: Update this to use setters instead
        self.name = name
        self.path = path
        self.nodelist = None
        self.nodes = None
        self.threads = None
        self.threadlist = None
        self.likwid_outputs = []
        self.pickle_filepath = f"{self.path}/{self.name}.pickle"
        self.likwid_out_naming = likwid_out_naming
        self.parameters = None
        self.simulation_finished = False
        self.num_mpi = 1
        self.Nclones = 1
        self._bandwidth_measured = True

        if "mpi" in self.name:
            self.num_mpi = int(self.name.split("_")[-1].replace("mpi", ""))

        for part in self.name.split("_"):
            if "Nclones" in part:
                self.Nclones = int(part.replace("Nclones", ""))

        self.procs_per_clone = int(self.num_mpi / self.Nclones) if self.num_mpi else 1

        if read_project:
            self.read_project()

    def read_project(self):
        """Read project data from files or pickle."""
        if os.path.exists(f"{self.path}/parameters.yml"):
            with open(f"{self.path}/parameters.yml", "r", encoding="utf-8") as file:
                self.parameters = file.read()
        self.misc_files = {}
        for filepath in glob.glob(f"{self.path}/misc*/*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                filename = filepath.split("/")[-1]
                self.misc_files[filename] = f.readlines()
        if "SLURM_VARIABLES.txt" in self.misc_files:
            for line in self.misc_files["SLURM_VARIABLES.txt"]:
                if "SLURM_NODELIST" in line:
                    self.nodelist = parse_nodelist(line)
                    break
        self.read_project_folder()

    def get_mpi_configuration(self):
        if self.num_mpi == 1:
            return f"{self.num_mpi} proc"
        else:
            return f"{self.num_mpi} proc(s)"

    def get_node_configuration(self):
        num_nodes = len(self.nodelist)
        # rint( len(self.nodelist),self.nodelist )
        if num_nodes == 1:
            return f"{num_nodes} node"
        else:
            return f"{num_nodes} nodes"

    def get_clone_configuration(self):
        """Get the configuration of clones and processors per clone.

        Returns:
            str: Configuration string.
        """
        if self.procs_per_clone == 1:
            return f"{self.Nclones} clone(s), {self.procs_per_clone} proc/clone"
        else:
            return f"{self.Nclones} clone(s), {self.procs_per_clone} proc(s)/clone"

    def read_project_folder(self):
        """Read project folder to gather LIKWID output data."""
        for likwid_output_path in glob.glob(f"{self.path}/{self.likwid_out_naming}"):
            lw_output = read_likwid_output(likwid_output_path)

            self.simulation_finished = lw_output["struphy_ended"]
            thread_list = None
            if self.nodelist:
                for line in lw_output["raw_file_lines"]:
                    if not thread_list:
                        for node in self.nodelist:
                            if node + ":" in line and "Metric" in line:
                                start_index = line.find(node)
                                thread_list = line[start_index:-2]
            lw_output["thread_list"] = thread_list
            self.threadlist = thread_list
            self.likwid_outputs.append(lw_output)
            
            # Check if bandwidth was measured
            bandwidth = self.get_value(
                    metric="Memory bandwidth [MBytes/s] STAT",
                    likwid_output_id=0,
                    group=self.get_likwid_groups()[0],
                    table="Metric STAT",
                    column="Avg",
                )
            if bandwidth:
                self._bandwidth_measured = True
            if self.simulation_finished:
                self.nodes = lw_output[self.get_likwid_groups()[0]]["nodes"].keys()
                self.threads = lw_output[self.get_likwid_groups()[0]]["dicts"]["Raw"][
                    "header"
                ][2:]

                # TODO get this from likwid directly
                self.num_mpi = lw_output["mpi_procs"]

    def get_likwid_groups(self, groups_include=["*"], groups_skip=[]):
        """Get list of LIKWID groups from the last output.

        Args:
            groups_include (list): List of patterns to include (default: ['*']).
            groups_skip (list): List of patterns to skip (default: []).

        Returns:
            list: List of LIKWID groups.
        """
        likwid_groups = [
            group
            for group in self.likwid_outputs[-1]
            if group
            not in [
                "filename",
                "raw_file_lines",
                "struphy_ended",
                "thread_list",
                "mpi_procs",
            ]
        ]

        # Convert '*' to a regex that matches any string, as users won't write '.*'
        groups_include = [pattern.replace("*", ".*") for pattern in groups_include]
        groups_skip = [pattern.replace("*", ".*") for pattern in groups_skip]

        # Compile include and skip patterns
        try:
            include_patterns = [
                re.compile(pattern) for pattern in groups_include if pattern
            ]
        except re.error as e:
            raise ValueError(f"Invalid pattern in groups_include: {e}")

        try:
            skip_patterns = [re.compile(pattern) for pattern in groups_skip if pattern]
        except re.error as e:
            raise ValueError(f"Invalid pattern in groups_skip: {e}")

        groups = []
        for group in likwid_groups:
            # Check if the group matches any pattern in groups_include
            if not any(pattern.match(group) for pattern in include_patterns):
                continue
            # Skip groups if they match any of the patterns in groups_skip
            if any(pattern.match(group) for pattern in skip_patterns):
                continue
            groups.append(group)

        return groups

    def get_value(
        self,
        metric,
        likwid_output_id = 0,
        group="model.integrate",
        table="Metric STAT",
        column="Sum",
    ):
        """Get a specific value from LIKWID output.

        Args:
            metric (str): The metric to retrieve.
            likwid_output_id (int): The ID of the LIKWID output.
            group (str): The group name.
            table (str): The table name.
            column (str): The column name.

        Returns:
            float: The retrieved value.
        """
        if likwid_output_id:
            try:
                return self.likwid_outputs[likwid_output_id][group]["dicts"][table][metric][
                    column
                ]
            except (KeyError, IndexError):
                    return None
        return None

    def get_columns(self, group="model.integrate", table="Metric STAT"):
        """Get columns from a specific group and table in the last LIKWID output.

        Args:
            group (str): The group name.
            table (str): The table name.

        Returns:
            list: List of column names.
        """
        table_data = self.likwid_outputs[-1][group]["dicts"][table]
        first_metric = list(table_data.keys())[1]
        return list(table_data[first_metric].keys())

    def get_maximum_id(
        self, metric, group="model.integrate", table="Metric STAT", column="Sum",
    ):
        """Get the ID of the LIKWID output with the maximum value for a specific metric.

        Args:
            metric (str): The metric to retrieve.
            group (str): The group name.
            table (str): The table name.
            column (str): The column name.

        Returns:
            int: The ID of the LIKWID output with the maximum value.
        """
        value, i_max = 0, None
        for ilw, likwid_output in enumerate(self.likwid_outputs):
            if group in likwid_output:
                try:
                    val = likwid_output[group]["dicts"][table][metric][column]
                except (KeyError, IndexError):
                    return None
                if val in ["nil", ""]:
                    val = 0
                if val >= value:
                    value = val
                    i_max = ilw
            # if group == 'model.integrate':
            #     for key in likwid_output.keys():
            #         if group in key:
            #             print('It actually exists!', key)

        return i_max

    def get_maximum(
        self, metric, group="model.integrate", table="Metric STAT", column="Sum",
    ):
        """Get the maximum value for a specific metric.

        Args:
            metric (str): The metric to retrieve.
            group (str): The group name.
            table (str): The table name.
            column (str): The column name.

        Returns:
            float: The maximum value.
        """
        if metric == "mpi":
            return self.num_mpi
        i_max = self.get_maximum_id(metric, group, table, column)
        return self.get_value(metric, i_max, group, table, column)

    def get_average(
        self, metric, group="model.integrate", table="Metric STAT", column="Sum",
    ):
        """Get the average value for a specific metric.

        Args:
            metric (str): The metric to retrieve.
            group (str): The group name.
            table (str): The table name.
            column (str): The column name.

        Returns:
            float: The average value.
        """
        total, count = 0, 0
        for likwid_output in self.likwid_outputs:
            if group in likwid_output:
                total += max(
                    total, likwid_output[group]["dicts"][table][metric][column],
                )
                count += 1
        return total / count if count > 0 else 0

    def get_description(self, group):
        imax = self.get_maximum_id(
            "DP [MFLOP/s] STAT",
            group=group,
            table="Metric STAT",
            column="Sum",
        )

        metrics = [
            "Runtime (RDTSC) [s] STAT",
            "DP [MFLOP/s] STAT",
            "Memory bandwidth [MBytes/s] STAT",
            "Memory data volume [GBytes] STAT",
        ]
        description = f"<b>{self.name}</b><br>"
        description += f"<b>Group</b>: {group}<br>"
        description += f"<b>MPI procs</b>: {self.num_mpi}<br>"
        for metric in metrics:
            try:
                data = self.get_value(
                    metric,
                    likwid_output_id=imax,
                    group=group,
                    table="Metric STAT",
                    column="Avg",
                )
                description += f"<b>{metric}</b>: {data}<br>"
            except:
                pass

        return description

    def __str__(self):
        """String representation of the Project instance."""
        return (
            f"Project(name={self.name},\npath={self.path},\nnum_mpi={self.num_mpi}, "
            f"simulation_finished={self.simulation_finished}, "
            f"nodelist={self.nodelist}, "
            f"likwid_outputs_count={len(self.likwid_outputs)})"
        )


def main():
    path = "./"
    project = Project("TestProject", path)
    print(project)


if __name__ == "__main__":
    main()
