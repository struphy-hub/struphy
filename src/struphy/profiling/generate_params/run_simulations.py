#!/usr/bin/env python3

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

import struphy
import struphy.utils.utils as utils
from struphy.console.utils import generate_batch_script, save_batch_script

state = utils.read_state()
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

    parser.add_argument('--venv',
                        type=str,
                        default='/ptmp/maxlin/struphy/virtual_envs/env_struphy_CI',
                        help='Path to virtual environment', )

    parser.add_argument('--jobname-prefix',
                        default='',
                        help='Slurm jobname prefix ("struphy_$CI_PIPELINE_ID", for example)', )
    
    args = parser.parse_args()
    action = args.action
    if args.jobname_prefix.isdigit():
        args.jobname_prefix = str(int(args.jobname_prefix))
    jobname_prefix = args.jobname_prefix
    venv_path = args.venv

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
                job_name = f'{jobname_prefix}_{projectname}'
                submit_file = f'submit_{projectname}.sh'

                script_params = {
                        'job_name': job_name,
                        'ntasks_per_node': 72,
                        'module_setup': "module load gcc/12 openmpi/4.1 anaconda/3/2023.03 git/2.43 pandoc/3.1 likwid/5.2",
                        'likwid': True,
                        'venv_path':venv_path,
                        'memory':'25GB',
                    }
                save_batch_script(generate_batch_script(**script_params), submit_file)
                
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
