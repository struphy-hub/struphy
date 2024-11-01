import os

import struphy.utils.utils as utils


def add_line(script, line, comment='', chars_until_comment=80):
    if len(line) > chars_until_comment:
        script += f"{line} # {comment}\n"
    else:
        script += f"{line} {' ' * (chars_until_comment - len(line))}# {comment}\n"
    return script


def generate_batch_script(**kwargs):
    # Default parameters for the batch script
    params = {
        'working_directory': './',
        'job-name': 'job_struphy',
        'output-file': "./job_struphy_%j.out",
        'error-file': "./job_struphy_%j.err",
        'nodes': 1,
        'ntasks-per-node': 72,
        'mail-user': "",
        'time': "00:10:00",
        'venv_path': "~/git_repos/env_struphy_devel",
        'partition': None,
        'ntasks_per_core': None,
        'cpus_per_task': None,
        'memory': '2GB',
        'module-setup': "module load anaconda/3/2023.03 gcc/12 openmpi/4.1 likwid/5.2",
        'likwid': False,
    }

    # Update params with any provided keyword arguments
    params.update(kwargs)

    # Start generating the SLURM batch script
    script = "#!/bin/bash\n"
    print(kwargs)
    
    script += generate_slurm_header(**params)
    print(script)
    exit()
    script = add_line(script, f"#SBATCH -o {params['output_file']}", "Standard output file")
    script = add_line(script, f"#SBATCH -e {params['error_file']}", "Standard error file")
    script = add_line(script, f"#SBATCH -D {params['working_directory']}", "Working directory")
    script = add_line(script, f"#SBATCH -J {params['job_name']}", "Job name")

    if params['partition']:
        script = add_line(script, f"#SBATCH --partition={params['partition']}", "Partition")
    script = add_line(script, f"#SBATCH --nodes={params['nodes']}", "Number of compute nodes")

    if params['ntasks_per_core']:
        script = add_line(script, f"#SBATCH --ntasks-per-core={params['ntasks_per_core']}", "Number of tasks per core")
    script = add_line(
        script, f"#SBATCH --ntasks-per-node={params['ntasks_per_node']}", "Number of MPI processes per node")

    if params['cpus_per_task']:
        script = add_line(script, f"#SBATCH --cpus-per-task={params['cpus_per_task']}", "Number of CPUs per task")
    if params['memory']:
        script = add_line(script, f"#SBATCH --mem={params['memory']}", "Memory allocation")
    script = add_line(script, "#SBATCH --mail-type=all", "Send email notifications for all events")
    script = add_line(script, f"#SBATCH --mail-user={params['mail_user']}", "Email address for notifications")
    script = add_line(script, f"#SBATCH --time={params['time']}", "Maximum runtime")
    script += "\n"

    # Activate environment
    script += "# Activate environment\n"
    script = add_line(script, f"source {params['venv_path']}/bin/activate", "Activate the virtual environment")
    script = add_line(script, "module purge", "Purge modules")
    script = add_line(script, params['module_setup'], "Load necessary modules")
    # script = add_line(script, f"export PATH={params['venv_path']}/bin/:$PATH", "Export path")

    script += "\n"

    # Set up environment variables
    script += "# Pinning\n"
    # script = add_line(script, "# Set the number of OMP threads *per process* to avoid overloading of the node!", "")
    # script = add_line(script, "#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK", "")
    # script = add_line(script, "#export OMP_PLACES=cores", "")
    script = add_line(script, "KMP_AFFINITY=scatter", "For pinning threads correctly")
    script += "\n"

    # Save hardware information
    script += "# Save hardware information\n"
    script = add_line(script, "misc=\"misc_$SLURM_JOB_ID\"", "")
    script = add_line(script, "mkdir -p $misc", "")
    script = add_line(script, "module list > \"$misc/module_list.txt\"", "Save loaded modules")
    script = add_line(script, "echo $OMP_NUM_THREADS > \"$misc/OMP_NUM_THREADS.txt\"", "Save OMP_NUM_THREADS value")
    script = add_line(script, "printenv > \"$misc/printenv.txt\"", "Save environment variables")
    script = add_line(script, "cp $0 $misc/", "Save a copy of the batch script")
    script += "\n"

    # Save SLURM-specific environment variables
    script += "# Save SLURM-specific environment variables\n"
    script = add_line(script, "for var in $(env | grep ^SLURM_ | cut -d= -f1); do", "Loop through SLURM variables")
    script = add_line(script, "    echo \"$var=${!var}\" >> \"$misc/SLURM_VARIABLES.txt\"", "Save SLURM variable")
    script = add_line(script, "done", "End of SLURM variable loop")
    script += "\n"

    # Add LIKWID-related commands if requested
    if params['likwid']:
        likwid_section = "# Add LIKWID-related commands\n"
        likwid_section = add_line(
            likwid_section, "LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)", "Set LIKWID prefix")
        likwid_section = add_line(likwid_section, "export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib",
                                  "Update LD_LIBRARY_PATH for LIKWID")
        likwid_section = add_line(likwid_section, "likwid-topology > \"$misc/likwid-topology.txt\"",
                                  "Save LIKWID topology information")
        likwid_section = add_line(likwid_section, "likwid-topology -g > \"$misc/likwid-topology-g.txt\"",
                                  "Save extended LIKWID topology information")
        script += likwid_section
        script += "\n"

    script += "\n"
    return script


def generate_slurm_header(**kwargs):
    """
    Generate a Slurm batch script with all possible SBATCH options,
    only adding those provided in kwargs.

    Parameters:
    - kwargs: Dictionary of parameters for the Slurm script, including SBATCH options.

    Returns:
    - str: The complete batch script as a string.
    """
    # List of all possible SBATCH options with their descriptions
    sbatch_options = {
        "job-name": "Job name",
        "output": "Standard output file",
        "error": "Standard error file",
        "workdir": "Working directory",
        "partition": "Partition to submit to",
        "nodes": "Number of compute nodes",
        "ntasks": "Total number of tasks",
        "ntasks-per-node": "Number of tasks per node",
        "cpus-per-task": "Number of CPUs per task",
        "time": "Maximum runtime (HH:MM:SS)",
        "mem": "Memory allocation",
        "mail-user": "Email address for notifications",
        "mail-type": "Type of email notifications (e.g., BEGIN, END, FAIL)",
        "constraint": "Constraints for selecting nodes",
        "gres": "Generic resources (e.g., GPUs)",
        "qos": "Quality of Service",
        "account": "Account name for resource allocation",
        "exclude": "Nodes to exclude",
        "mincpus": "Minimum number of CPUs per node",
        "requeue": "Requeue the job if it fails",
        "signal": "Send a signal to the job before it is terminated",
        "nice": "Set the scheduling priority",
        "export": "Export environment variables"
        # Add more options as needed
    }

    # Start generating the SLURM batch script
    script = "#!/bin/bash\n"

    # Add SBATCH directives based on kwargs
    for option, description in sbatch_options.items():
        if option in kwargs and kwargs[option] is not None:
            script = add_line(script, f"#SBATCH --{option}={kwargs[option]}", description)
    return script



def save_batch_script(batch_script, filename, path=None):
    if path is None:
        state = utils.read_state()
        path = state['b_path']
    batch_path = os.path.join(path, filename)
    with open(batch_path, 'w') as f:
        f.write(batch_script)
    # print(batch_script)
    # print(batch_path)
