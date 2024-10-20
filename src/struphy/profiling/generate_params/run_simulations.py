#!/usr/bin/env python3

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

import struphy

libpath = struphy.__path__[0]
with open(os.path.join(libpath, "state.yml")) as f:
    state = yaml.load(f, Loader=yaml.FullLoader)
i_path = state["i_path"]
o_path = state["o_path"]

# Color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"  # No Color


def main():

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run or check simulations.")
    parser.add_argument(
        "action", choices=["run", "check"], help="Action to perform: run or check"
    )
    args = parser.parse_args()
    action = args.action

    i = 0  # Set project index to 0
    pattern = r"mpi(\d+)\.yml"  # Regex pattern to find the number before .yml

    # Loop over all parameter files
    param_files = glob.glob(f"{i_path}/params_*.yml")
    for file in param_files:
        param_filename = os.path.basename(file)  # Get the parameter filename
        match = re.search(pattern, param_filename)
        if match:
            nmpi = int(match.group(1))  # Extract the number part from the filename
            projectname = param_filename.replace("params_", "").replace(
                ".yml", ""
            )  # Create project name
            padded_index = f"{i:04d}"  # Pad the index with zeros

            # Determine the type of simulation based on the parameter file name
            if param_filename.startswith("params_Vlasov_"):
                model = "Vlasov"
            elif param_filename.startswith("params_Maxwell_"):
                model = "Maxwell"
            elif param_filename.startswith("params_LinearMHD_"):
                model = "LinearMHD"
            elif param_filename.startswith("params_ShearAlfven_"):
                model = "ShearAlfven"
            elif param_filename.startswith("params_LinearMHDDriftkineticCC_"):
                model = "LinearMHDDriftkineticCC"
            elif param_filename.startswith("params_LinearMHDVlasovCC_"):
                model = "LinearMHDVlasovCC"
            elif param_filename.startswith("params_VlasovAmpereOneSpecies_"):
                model = "VlasovAmpereOneSpecies"
            else:
                print(f"Unrecognized parameter file type: {param_filename}")
                continue

            # Check if directory exists
            output_dir = f"{o_path}/{projectname}"
            if os.path.isdir(output_dir):
                print(f"{YELLOW}Skipping:\t{NC}{projectname}")
            else:
                nodes = (nmpi + 71) // 72  # Calculate the number of nodes required
                submit_file = f"batch_raven_{nodes}node.sh"
                command = [
                        "struphy",
                        "run",
                        model,
                        "--mpi",
                        str(nmpi),
                        "--batch",
                        submit_file,
                        "--inp",
                        param_filename,
                        "--output",
                        projectname,
                        "--save-step",
                        "1",
                        "--runtime",
                        "90",
                        "--likwid",
                        "-li",
                        f"likwid_config_mpi{nmpi}.yml",
                        "-lr",
                        "1",
                    ]
                print(" ".join(command))
                if action == "run":
                    subprocess.run(command)
                    print(f"{GREEN}Running:\t{projectname} ({param_filename}){NC}")
                elif action == "check":
                    projectname_width = 50
                    print(
                        f"{GREEN}Running:\t{projectname.ljust(projectname_width)}\t({param_filename}) using {submit_file}{NC}"
                    )
                else:
                    print(f"{RED}Invalid action: {action}{NC}")
                    print("Usage: script.py [run|check]")
                    sys.exit(1)
                i += 1
        else:
            print(f"No number found in filename {file}")
    print(f"Total new simulations: {i}")


if __name__ == "__main__":
    main()
