def struphy_run(model,
                inp=None,
                input_abs=None,
                output='sim_1',
                output_abs=None,
                batch=None,
                batch_abs=None,
                batch_auto='mpcdf',
                runtime=300,
                save_step=1,
                restart=False,
                mpi=1,
                debug=False,
                cprofile=False,
                likwid=False,
                likwid_inp=None,
                likwid_input_abs=None,
                likwid_repetitions=1,):
    """
    Run a Struphy model: prepare arguments, output folder and execute main().

    Parameters
    ----------
    model : str
        The name of the Struphy model.

    inp : str
        The .yml input parameter file relative to <struphy_path>/io/inp.

    input_abs : str
        The absolute path to the .yml input parameter file.

    output : str
        Name of the output folder in <struphy_path>/io/out.

    output_abs : str
        Absolute path to the output folder.

    batch : str
        Name of the batch script for runs on a cluster.

    batch_abs : str
        Absolute path to the batch scripts for runs on a cluster.

    runtime : int
        Maximum runtime of the simulation in minutes. Will complete the time step and exit after this time is reached.

    save_step : int
        How often to save data in hdf5 file, i.e. every "save_step" time step.

    restart : bool
        Whether to restart an existing simulation.

    mpi : int
        Number of MPI processes for runs with "mpirun".

    debug : bool
        Whether to run in Cobra debug mode, see https://docs.mpcdf.mpg.de/doc/computing/cobra-user-guide.html#interactive-debug-runs'.

    cprofile : bool
        Whether to run with Cprofile (slower).
    
    likwid : bool
        Whether to run with Likwid (Needs to be installed first). Default is False.
    
    likwid_inp : str, optional
        The .yml input parameter file for Likwid relative to <struphy_path>/io/inp. Default is None.

    likwid_input_abs : str, optional
        The absolute path to the .yml input parameter file for Likwid. Default is None.

    likwid_repetitions : int, optional
        Number of repetitions for Likwid profiling. Default is 1.
    """

    import subprocess
    import shutil
    import os
    import struphy
    import yaml

    libpath = struphy.__path__[0]

    # Struphy paths
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    i_path = state['i_path']
    o_path = state['o_path']
    b_path = state['b_path']

    # create absolute i/o paths
    if input_abs is None:
        if inp is None:
            default_yml = os.path.join(i_path, f'params_{model}.yml')
            if os.path.isfile(default_yml):
                print('\nRunning with default parameter file ...')
                input_abs = default_yml
            else:
                # load model class
                from struphy.models import fluid, kinetic, hybrid, toy
                objs = [fluid, kinetic, hybrid, toy]
                for obj in objs:
                    try:
                        model_class = getattr(obj, model)
                    except AttributeError:
                        pass

                params = model_class.generate_default_parameter_file()
                exit()
        else:
            input_abs = os.path.join(i_path, inp)

    if output_abs is None:
        output_abs = os.path.join(o_path, output)

    if batch_abs is None:
        if batch is not None:
            batch_abs = os.path.join(b_path, batch)

    # take existing parameter file for restart
    if restart:
        input_abs = os.path.join(output_abs, 'parameters.yml')


    # Read likwid params
    if likwid:
        if likwid_inp is None and likwid_input_abs is None:
            # use default likwid parameters
            likwid_command = ['likwid-mpirun', '-n', str(mpi), '-g', 'MEM_DP', '-stats', '-marker']
        else:
            if likwid_inp is not None:
                likwid_input_abs = os.path.join(i_path, likwid_inp)
            
            
            with open(likwid_input_abs, 'r') as file:
                config = yaml.safe_load(file)
            
            # Get the command from the configuration
            command_base = config.get('command', None)
            if not command_base:
                print("Missing required configuration: 'command'")
                exit(1)
            
            likwid_config = config.get(command_base, {})
            
            # Get the options list
            options = likwid_config.get('options', [])
            
            # Flatten the options list
            flattened_options = ['-np',str(mpi)]
            for item in options:
                if isinstance(item, dict):
                    for key, value in item.items():
                        flattened_options.append(key)
                        flattened_options.append(str(value))  # Ensure the value is a string
                else:
                    flattened_options.append(item)

            
            # Construct the command as a list
            likwid_command = [command_base]
            likwid_command.extend(flattened_options)
            

    # command parts
    cmd_python = ['python3']
    cmd_main = ['main.py',
                model,
                '-i',
                input_abs,
                '-o',
                output_abs,
                '--runtime',
                str(runtime),
                '-s',
                str(save_step)]
    cmd_cprofile = ['-m',
                    'cProfile',
                    '-o',
                    os.path.join(output_abs, 'profile_tmp'),
                    '-s',
                    'time']

    # run in normal or debug mode
    if batch_abs is None:

        if debug:
            print('\nLaunching main() in Cobra debug mode ...')
            command = ['srun',
                       '-n',
                       str(mpi),
                       '-p',  # interactive commands
                       'interactive',
                       '--time',
                       '119',
                       '--mem',
                       '2000'] + cmd_python + cprofile*cmd_cprofile + cmd_main
        elif likwid:
            cmd_main[0] = f"{libpath}/{cmd_main[0]}"
            command = likwid_command + cmd_python + cprofile*cmd_cprofile + cmd_main + ['--likwid']
        else:
            print('\nLaunching main() in normal mode ...')
            command = ['mpirun',
                       '-n',
                       str(mpi)] + cmd_python + cprofile*cmd_cprofile + cmd_main

        # add restart flag
        if restart:
            command += ['-r']
            
        if cprofile:
            print('\nCprofile turned on.')
        else:
            print('\nCprofile turned off.')

        # run command as subprocess
        print(command)
        print(f"\nRunning the following command:\n{' '.join(command)}")
        if likwid:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, cwd=libpath)
        
    # run in batch mode
    else:

        # create output folder if it does not exit
        if not os.path.exists(output_abs):
            os.mkdir(output_abs)
            os.mkdir(os.path.join(output_abs, 'data/'))

        # remove sim.out file
        file = os.path.join(output_abs, 'sim.out')
        if os.path.exists(file):
            os.remove(file)
            print('Removed file ' + file)

        # remove sim.err file
        file = os.path.join(output_abs, 'sim.err')
        if os.path.exists(file):
            os.remove(file)
            print('Removed file ' + file)

        # remove old batch script
        file = os.path.join(output_abs, 'batch_script.sh')
        if os.path.exists(file):
            os.remove(file)
            print('Removed file ' + file)

        # remove struphy.out file
        file = os.path.join(output_abs, 'sim.out')
        if os.path.exists(file):
            os.remove(file)
            print('Removed file ' + file)

        # copy batch script to output folder
        batch_abs_new = os.path.join(output_abs, 'batch_script.sh')
        if batch_auto:
            batch_auto = 'raven'
            sbatch_params = {
                'raven':{
                    'ntasks_per_node': 72,
                    'likwid': True,
                }
            }

            batch_script = generate_batch_script(**sbatch_params[batch_auto])
            print(batch_script)
            exit()
            # with open(batch_abs_new,'w') as f:
            #     f.write()
        else:   
            shutil.copy2(batch_abs, batch_abs_new)

        # delete srun command from batch script
        with open(batch_abs_new, 'r') as f:
            lines = f.readlines()
            if 'srun' in lines[-1]:
                lines = lines[:-2]

        with open(batch_abs_new, 'w') as f:

            for line in lines:
                f.write(line)
            f.write('# Run command added by Struphy\n')
            
            
            command = cmd_python + cprofile*cmd_cprofile + [f"{libpath}/{' '.join(cmd_main)}"]
            if restart:
                command += ['-r']
            
            if likwid:
                command = likwid_command + command + ['--likwid']
            
            if likwid:
                print('Running with likwid')
                f.write(f'# Launching likwid {likwid_repetitions} times with likwid-mpirun\n')
                for i in range(likwid_repetitions):
                    f.write(f'\n\n# Run number {i:03}\n')
                    f.write(' '.join(command) + ' > ' + os.path.join(output_abs, f'struphy_likwid_{i:03}.out'))
            else:
                print('Running with srun')
                f.write('srun ' + ' '.join(command) + ' > ' + os.path.join(output_abs, 'struphy.out'))
        
        # submit batch script in output folder
        print('\nLaunching main() in batch mode ...')
        subprocess.run(['sbatch',
                        'batch_script.sh',
                        ],
                       check=True, cwd=output_abs)


def generate_batch_script(**kwargs):
    """
    Generate a batch script for submitting jobs with SLURM.

    Parameters:
    **kwargs : dict
        Keyword arguments that override the default SLURM job parameters. Examples include:
        - output_file (str): Path for the job output file.
        - error_file (str): Path for the job error file.
        - nodes (int): Number of compute nodes.
        - ntasks_per_node (int): Number of MPI processes per node.
        - mail_user (str): Email address for notifications.
        - time (str): Maximum runtime for the job in HH:MM:SS format.
        - activate_env (str): Path to the environment activation script.
        - partition (str): Partition name.
        - ntasks_per_core (int): Number of tasks per core.
        - cpus_per_task (int): Number of CPUs per task.
        - memory (str): Memory allocation for the job.
        - module_setup (str): Modules to be loaded before running the job.
        - likwid (bool): If True, include LIKWID-related commands in the script.

    Returns:
    str
        The generated batch script as a string.
    """

    # Default parameters for the batch script
    params = {
        'output_file': "./job_struphy_%j.out",
        'error_file': "./job_struphy_%j.err",
        'nodes': 1,
        'ntasks_per_node': 72,
        'mail_user': "",
        'time': "00:45:00",
        'activate_env': "/u/maxlin/NMPP/01_envs/env_struphy/bin/activate",
        'partition': None,
        'ntasks_per_core': None,
        'cpus_per_task': None,
        'memory': None,
        'module_setup': "module load anaconda/3/2023.03 gcc/10 openmpi/4.1 likwid/5.2",
        'likwid': False
    }

    # Update params with any provided keyword arguments
    params.update(kwargs)

    # Start generating the SLURM batch script
    script = f"""#!/bin/bash
#SBATCH -o {params['output_file']}                      # Standard output file
#SBATCH -e {params['error_file']}                      # Standard error file
#SBATCH -D ./                                          # Working directory
#SBATCH -J job_struphy                                 # Job name
"""

    if params['partition']:
        script += f"#SBATCH --partition={params['partition']}           # Partition\n"
    script += f"#SBATCH --nodes={params['nodes']}                      # Number of compute nodes\n"
    if params['ntasks_per_core']:
        script += f"#SBATCH --ntasks-per-core={params['ntasks_per_core']}  # Number of tasks per core\n"
    script += f"#SBATCH --ntasks-per-node={params['ntasks_per_node']}  # Number of MPI processes per node\n"
    if params['cpus_per_task']:
        script += f"#SBATCH --cpus-per-task={params['cpus_per_task']}       # Number of CPUs per task\n"
    if params['memory']:
        script += f"#SBATCH --mem={params['memory']}                   # Memory allocation\n"
    script += f"#SBATCH --mail-type=all                                # Send email notifications for all events\n"
    script += f"#SBATCH --mail-user={params['mail_user']}              # Email address for notifications\n"
    script += f"#SBATCH --time={params['time']}                        # Maximum runtime\n"

    # Activate environment
    script += f"\n# Activate environment\nsource {params['activate_env']}                        # Activate the virtual environment\nmodule purge\n{params['module_setup']}                               # Load necessary modules\n"

    # Set up environment variables
    script += """

export PATH=/raven/u/maxlin/NMPP/01_envs/env_struphy/bin:$PATH

# Set the number of OMP threads *per process* to avoid overloading of the node!
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly
#export OMP_PLACES=cores
KMP_AFFINITY=scatter

# Save hardware information
misc="misc_$SLURM_JOB_ID"
mkdir -p $misc
module list > "$misc/module_list.txt"                # Save loaded modules
echo $OMP_NUM_THREADS > "$misc/OMP_NUM_THREADS.txt"  # Save OMP_NUM_THREADS value
printenv > "$misc/printenv.txt"                      # Save environment variables
cp $0 $misc/                                         # Save a copy of the batch script

# Save SLURM-specific environment variables
for var in $(env | grep ^SLURM_ | cut -d= -f1); do
    echo "$var=${{!var}}">>"$misc/SLURM_VARIABLES.txt"
done
"""

    # Add LIKWID-related commands if requested
    if params['likwid']:
        likwid_section = """
LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib

likwid-topology > "$misc/likwid-topology.txt"         # Save LIKWID topology information
likwid-topology -g > "$misc/likwid-topology-g.txt"   # Save extended LIKWID topology information
"""
        script += likwid_section

    return script