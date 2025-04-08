import ast
import re
import os
import sys

from struphy.utils.utils import read_state

START_OMP_REGION = "#$ omp parallel"
END_OMP_REGION = "#$ omp end parallel"


def get_omp_regions(lines):
    regions = []
    for iline, line in enumerate(lines):
        if line[:3] == "def":
            pattern_fname = r"^def\s+(\w+)\s*\("
            match = re.search(pattern_fname, line)
            function_name = match.group(1)
        if START_OMP_REGION in line:
            start = iline
        if END_OMP_REGION in line:
            end = iline
            regions.append(
                {
                    "function": function_name,
                    "omp_region": lines[start : end + 1],
                }
            )
    return regions


def get_omp_variables(lines):
    # Pattern to capture variable names inside each clause
    # TODO: Explain this regex!
    pattern = r"(private|firstprivate|shared)\s*\(([^)]+)\)"
    variables = {"private": [], "firstprivate": [], "shared": []}

    for line in lines:
        if "#$ omp" in line:
            matches = re.findall(pattern, line)
            for kind, varlist in matches:
                variables[kind].extend([v.strip() for v in varlist.split(",")])
    return variables


def remove_tabbing(lines):
    line0 = lines[0]
    num_spaces = len(line0) - len(line0.lstrip(" "))
    # print("Number of leading spaces:", num_spaces)
    lines_out = []
    for line in lines:
        lines_out.append(line[num_spaces:])
    return lines_out


def get_python_variables(lines):
    lines = remove_tabbing(lines)

    # Parse the source into an AST
    source = "".join(lines)
    tree = ast.parse(source)

    # Define a visitor to collect all variable names
    class VariableCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            # Add the name (node.id) to the set
            self.names.add(node.id)
            self.generic_visit(node)

    # Create the visitor and walk the tree
    collector = VariableCollector()
    collector.visit(tree)

    # print("All variable names:", collector.names)
    return collector.names


def find_undeclared_variables(omp_region, verbose=False):
    omp_variables = get_omp_variables(omp_region)
    python_variables = get_python_variables(omp_region)
    undeclared_variables = []
    for variable in python_variables:
        declared = False
        for key, omp_var_list in omp_variables.items():
            if variable in omp_var_list:
                declared = True
                if verbose:
                    print(f"{variable} is {key}!")
        if not declared:
            if verbose:
                print(f"{variable} is undeclared!")
            undeclared_variables.append(variable)
    return undeclared_variables


def check_file(filename, verbose=False):
    print("-" * 80)
    print(f"# File: {filename}")
    with open(filename, "r") as file:
        lines = file.readlines()
    omp_region_dicts = get_omp_regions(lines)

    num_undeclared_variables = 0

    for omp_region_dict in omp_region_dicts:
        function = omp_region_dict["function"]
        omp_region = omp_region_dict["omp_region"]
        undeclared_variables = find_undeclared_variables(omp_region, verbose)
        if len(undeclared_variables) > 0:
            print(f"## Function: {function}")
            #print(f"Undeclared variables: {undeclared_variables}\n")
            print(f"shared({','.join(undeclared_variables)})\n")
            num_undeclared_variables += len(undeclared_variables)
    return num_undeclared_variables


if __name__ == "__main__":
    # If a path is provided as argument, check only that one
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f"Error: '{path}' is not a valid file path.")
            sys.exit(2)
        num = check_file(path, verbose=False)
        print(f"Undeclared variables in {path}: {num}")
        sys.exit(0 if num == 0 else 1)
    else:
        # No argument provided: check all kernels
        num_undeclared_variables = 0
        for kernel in state["kernels"]:
            num = check_file(kernel, verbose=False)
            num_undeclared_variables += num
        print(f"{num_undeclared_variables = }")
        sys.exit(0 if num_undeclared_variables == 0 else 1)