import os

import struphy.utils.utils as utils


def add_line(script, line, comment='', chars_until_comment=60):
    if len(line) > chars_until_comment:
        script += f"{line} # {comment}\n"
    else:
        script += f"{line} {' ' * (chars_until_comment - len(line))}# {comment}\n"
    return script


def generate_batch_script(chars_until_comment=80, **kwargs):
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
    header = generate_slurm_header(chars_until_comment=chars_until_comment, **params)

    setup = generate_setup(chars_until_comment=chars_until_comment, **params)

    run_script = "\n"  # generate_run_script(**params)

    script = "#!/bin/bash\n"
    script += header + "\n\n"
    script += setup + "\n\n"
    # script += run_script

    return script


def generate_setup(chars_until_comment=80, **params):

    script = ""

    # Activate environment
    script += "# Activate environment\n"
    script = add_line(script, f"source {params['venv_path']}/bin/activate", "Activate the virtual environment")

    script += "\n\n"
    script += "# Load modules\n"
    script = add_line(script, "module purge", "Purge modules")
    # script = add_line(script, params['module-setup'], "Load necessary modules")
    modules = params.get('modules', None)
    if modules:
        for module in modules:
            script = add_line(script, f"module load {module}", f"Load {module}")
    # script = add_line(script, f"export PATH={params['venv_path']}/bin/:$PATH", "Export path")

    script += "\n"

    # Set up environment variables
    script += "# Pinning\n"
    # script = add_line(script, "# Set the number of OMP threads *per process* to avoid overloading of the node!", "")
    # script = add_line(script, "#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK", "")
    # script = add_line(script, "#export OMP_PLACES=cores", "")
    script = add_line(script, "KMP_AFFINITY=scatter", "For pinning threads correctly")
    script += "\n"

    # Display hardware information directly
    script += "# Display hardware information directly\n"
    script = add_line(script, "echo \"Loaded modules:\"", "Show loaded modules")
    script = add_line(script, "module list", "")

    script = add_line(script, "echo \"OMP_NUM_THREADS value:\"", "Show OMP_NUM_THREADS value")
    script = add_line(script, "echo $OMP_NUM_THREADS", "")

    script = add_line(script, "echo \"Environment variables:\"", "Show environment variables")
    script = add_line(script, "printenv", "")

    script = add_line(script, "echo \"Content of this batch script:\"", "Show content of the batch script")
    script = add_line(script, "cat $0", "")
    script += "\n"

    # Display SLURM-specific environment variables
    script += "# Display SLURM-specific environment variables\n"
    script = add_line(script, "echo \"SLURM-specific environment variables:\"", "")
    script = add_line(script, "for var in $(env | grep ^SLURM_ | cut -d= -f1); do", "Loop through SLURM variables")
    script = add_line(script, "    echo \"$var=${!var}\"", "Show SLURM variable")
    script = add_line(script, "done", "End of SLURM variable loop")
    script += "\n"

    # Add LIKWID-related commands if requested
    if params['likwid']:
        likwid_section = "# Add LIKWID-related commands\n"
        likwid_section = add_line(
            likwid_section, "LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)", "Set LIKWID prefix",
        )
        likwid_section = add_line(
            likwid_section, "export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib",
            "Update LD_LIBRARY_PATH for LIKWID",
        )
        likwid_section = add_line(
            likwid_section, "likwid-topology",
            "Show LIKWID topology information",
        )
        likwid_section = add_line(
            likwid_section, "likwid-topology -g",
            "Show graphical LIKWID topology information",
        )
        script += likwid_section
        script += "\n"

    script += "\n"
    return script


def generate_slurm_header(chars_until_comment=80, **kwargs):
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
        "export": "Export environment variables",
        # Add more options as needed
    }

    # Start generating the SLURM batch script
    script = "#"*(chars_until_comment+2)
    script += "\n"
    # Add SBATCH directives based on kwargs
    for option, description in sbatch_options.items():
        if option in kwargs and kwargs[option] is not None:
            script = add_line(script, f"#SBATCH --{option}={kwargs[option]}",
                              description, chars_until_comment=chars_until_comment)
    script += "#"*(chars_until_comment+2)
    script += "\n"
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
