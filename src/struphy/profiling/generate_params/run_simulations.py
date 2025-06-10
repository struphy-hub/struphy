#!/usr/bin/env python3
# pip install git+ssh://git@github.com/max-models/slurm-script-generator.git

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml
from slurm_script_generator.main import generate_script

import struphy
import struphy.utils.utils as utils
from struphy.console.parsers import add_likwid_parser

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
        "action",
        choices=["run", "check"],
        help="Action to perform: run or check",
    )

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        default="params_CI*.yml",
        help="Path matching the params to run",
    )

    parser.add_argument(
        "--mpi",
        type=int,
        default=1,
        help="Number of MPI processes",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes",
    )

    parser.add_argument(
        "--nclones",
        type=int,
        default=1,
        help="Number of nodes",
    )

    parser.add_argument(
        "--tasks-per-node",
        type=int,
        default=72,
        help="Number of MPI processes",
    )

    parser.add_argument(
        "--venv",
        type=str,
        default="/ptmp/maxlin/struphy/virtual_envs/env_struphy_CI",
        help="Path to virtual environment",
    )

    parser.add_argument(
        "--jobname-prefix",
        default="",
        help='Slurm jobname prefix ("struphy_$CI_PIPELINE_ID", for example)',
    )

    add_likwid_parser(parser)

    args = parser.parse_args()
    action = args.action
    if args.jobname_prefix.isdigit():
        args.jobname_prefix = str(int(args.jobname_prefix))

    jobname_prefix = args.jobname_prefix
    venv_path = args.venv
    inp = args.inp

    num_new_simulations = 0  # Counter for number of new simulations

    # Loop over all parameter files
    param_files = glob.glob(f"{i_path}/{inp}")
    for file in param_files:
        param_filename = os.path.basename(file)  # Get the parameter filename

        # Parse parameter file
        with open(file, "r") as f:
            params = yaml.safe_load(f)

        if not "model" in params:
            print(f"model missing in {param_filename}")
            continue
        model = params["model"]
        # Get setup params
        projectname = (
            f"{model}_nodes-{args.nodes}_mpi-{args.mpi}_nclones-{args.nclones}_{param_filename.replace('.yml', '')}"
        )

        # Check if directory exists
        output_dir = f"{o_path}/{projectname}"
        if os.path.isdir(output_dir):
            print(f"{YELLOW}Skipping:\t{NC}{output_dir}")
        else:
            # nodes = (nmpi + 71) // 72  # Calculate the number of nodes required
            job_name = f"{jobname_prefix}_{model}"
            submit_file = f"submit_{projectname}.sh"

            script_params = {
                "job_name": job_name,
                "ntasks_per_node": args.tasks_per_node,
                "nodes": args.nodes,
                "modules": ["gcc/12", "openmpi/4.1", "anaconda/3/2023.03", "git/2.43", "pandoc/3.1", "likwid/5.2"],
                "likwid": True,
                "venv": venv_path,
                "mem": "25GB",
                "time": "00:45:00",
            }
            save_batch_script(generate_script(script_params), submit_file)

            command = [
                "struphy",
                "run",
                model,
                "--mpi",
                str(args.mpi),
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
                "-g",
                "MEM_DP",
                "--stats",
                "--marker",
                "--hpcmd_suspend",
                "-lr",
                "1",
            ]
            # print(" ".join(command))
            if action == "run":
                subprocess.run(command)
                print(f"{GREEN}Running:\t{projectname} ({param_filename}){NC}")
            elif action == "check":
                projectname_width = 50
                print(
                    f"{GREEN}Would run:\t{projectname.ljust(projectname_width)}\t({param_filename}) using {submit_file}{NC}",
                )
                print(f"Command: {' '.join(command)}")
            else:
                print(f"{RED}Invalid action: {action}{NC}")
                print("Usage: script.py [run|check]")
                sys.exit(1)
            num_new_simulations += 1
    print(f"Total new simulations: {num_new_simulations}")
    cmd = ["squeue", "-u", "$USER"]
    subprocess.run(cmd)


def save_batch_script(batch_script, filename, path=None):
    if path is None:
        state = utils.read_state()
        path = state["b_path"]
    batch_path = os.path.join(path, filename)
    with open(batch_path, "w") as f:
        f.write(batch_script)


if __name__ == "__main__":
    main()
