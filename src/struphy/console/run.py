import subprocess
import shutil
import os
import struphy
import yaml


def update_kwargs_inplace(**kwargs):
    """Update keyword arguments in-place with defaults."""
    defaults = {
        'inp': None,
        'input_abs': None,
        'output': 'sim_1',
        'output_abs': None,
        'batch': None,
        'batch_abs': None,
        'runtime': 300,
        'save_step': 1,
        'restart': False,
        'mpi': 1,
        'debug': False,
        'cprofile': False,
        'likwid': False,
        'likwid_inp': None,
        'likwid_input_abs': None,
        'likwid_repetitions': 1
    }
    for key, default in defaults.items():
        kwargs[key] = kwargs.get(key, default)
    return kwargs


def load_state(libpath):
    """Load the Struphy state from a YAML file."""
    state_path = os.path.join(libpath, 'state.yml')
    with open(state_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def generate_io_path(model, libpath, **kwargs):
    
    inp = kwargs.get('inp', None)
    input_abs = kwargs.get('input_abs', None)
    output = kwargs.get('output', 'sim_1')
    output_abs = kwargs.get('output_abs', None)
    batch = kwargs.get('batch', None)
    batch_abs = kwargs.get('batch_abs', None)
    restart = kwargs.get('restart', False)

    # Struphy paths
    state = load_state(libpath)
    i_path = state['i_path']
    o_path = state['o_path']
    b_path = state['b_path']

    if input_abs is None:
        input_abs = _get_default_input_path(i_path, model, kwargs['inp'])
        
    if output_abs is None:
        output_abs = os.path.join(o_path, output)

    if batch_abs is None:
        if batch is not None:
            batch_abs = os.path.join(b_path, batch)

    # take existing parameter file for restart
    if restart:
        input_abs = os.path.join(output_abs, 'parameters.yml')
    
    return input_abs, output_abs, batch_abs

def _get_default_input_path(i_path, model, inp):
    """Determine the default input path for a given model."""

    if inp:
        return os.path.join(i_path, inp)

    default_yml = os.path.join(i_path, f'params_{model}.yml')
    if os.path.isfile(default_yml):
        print('\nRunning with default parameter file...')
        return default_yml

    from struphy.models import fluid, kinetic, hybrid, toy
    model_classes = [fluid, kinetic, hybrid, toy]
    for obj in model_classes:
        try:
            model_class = getattr(obj, model)
            model_class.generate_default_parameter_file()
        except AttributeError:
            pass

    exit()  # Exit if no suitable model class is found

def generate_likwid_command(mpi, i_path, likwid_inp, likwid_input_abs):
    
    if not likwid_inp and not likwid_input_abs:
        return ['likwid-mpirun', '-n', str(mpi), '-g', 'MEM_DP', '-stats', '-marker']

    
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
    return likwid_command

def cleanup_out_dir(output_abs):
    """Clean up output directory by removing specific files."""
    _ensure_dir_exists(output_abs)
    _ensure_dir_exists(os.path.join(output_abs, 'data/'))
    _remove_file_if_exists(os.path.join(output_abs, 'sim.out'))
    _remove_file_if_exists(os.path.join(output_abs, 'sim.err'))
    _remove_file_if_exists(os.path.join(output_abs, 'batch_script.sh'))


def _ensure_dir_exists(dir_path):
    """Ensure a directory exists, create it if not."""
    if not os.path.exists(dir_path):
        os.makedirs(os.path.join(dir_path, 'data/'))


def _remove_file_if_exists(file_path):
    """Remove file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Removed file {file_path}')


def generate_run_command(model, **kwargs):
    libpath = struphy.__path__[0]
    input_abs = kwargs.get('input_abs')
    output_abs = kwargs.get('output_abs')
    runtime = kwargs.get('runtime')
    save_step = kwargs.get('save_step')
    restart = kwargs.get('restart')
    mpi = kwargs.get('mpi')
    debug = kwargs.get('debug')
    cprofile = kwargs.get('cprofile')
    likwid = kwargs.get('likwid')
    likwid_inp = kwargs.get('likwid_inp')
    likwid_input_abs = kwargs.get('likwid_input_abs')
    batch = kwargs.get('batch')
    batch_abs = kwargs.get('batch_abs')
    state = load_state(libpath)
    if cprofile:
        print('\nCprofile turned on.')
    else:
        print('\nCprofile turned off.')
    
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
    if debug:
        print('\nLaunching main() in Cobra debug mode ...')
        return _generate_debug_command(cmd_python, cmd_main, cmd_cprofile, mpi, restart)
    
    if batch is not None or batch_abs is not None or likwid:
        cmd_main[0] = os.path.join(libpath, cmd_main[0])

    if likwid:
        likwid_command = generate_likwid_command(mpi, state['i_path'], likwid_inp, likwid_input_abs)
        return likwid_command + cmd_python + cprofile*cmd_cprofile + cmd_main + ['--likwid'] +  ['-r' * restart]
    
    print('\nLaunching main() in normal mode ...')

    return ['mpirun','-n', str(mpi)] + cmd_python + cprofile*cmd_cprofile + cmd_main + ['-r' * restart]

def _generate_debug_command(cmd_python, cmd_main, cmd_cprofile, mpi, restart):
    """Generate the debug run command."""
    print('\nLaunching main() in Cobra debug mode...')
    return [
        'srun', '-n', str(mpi), '-p', 'interactive', '--time', '119', '--mem', '2000'
    ] + cmd_python + cmd_cprofile + cmd_main + ['-r' * restart]

def generate_batch_script(model, **kwargs):
    output_abs = kwargs.get('output_abs', None)
    batch_abs = kwargs.get('batch_abs', None)
    mpi = kwargs.get('mpi', 1)
    likwid = kwargs.get('likwid', False)
    likwid_repetitions = kwargs.get('likwid_repetitions', 1)

    # copy batch script to output folder
    if batch_abs and output_abs:
        batch_abs_new = os.path.join(output_abs, 'batch_script.sh')
        shutil.copy2(batch_abs, batch_abs_new)

    # delete srun command from batch script
    with open(batch_abs_new, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1 and 'srun' in lines[-1]:
            lines = lines[:-2]

    with open(batch_abs_new, 'w') as f:
        for line in lines:
            f.write(line)
        f.write('# Run command added by Struphy\n')
        
        
        command = generate_run_command(model, **kwargs)
        print(f'{command = }')
        if likwid:
            print('Running with likwid')
            f.write(f'# Launching likwid {likwid_repetitions} times with likwid-mpirun\n')
            for i in range(likwid_repetitions):
                f.write(f'\n\n# Run number {i:03}\n')
                f.write(' '.join(command) + ' > ' + os.path.join(output_abs, f'struphy_likwid_{i:03}.out'))
        else:
            print('Running with srun')
            command = ['srun'] + command[3:] #TODO
            f.write(' '.join(command) + ' > ' + os.path.join(output_abs, 'struphy.out'))
    return command
def struphy_run(model, **kwargs):
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
    # Set default values for the arguments
    kwargs = update_kwargs_inplace(**kwargs)
    libpath = struphy.__path__[0]

    kwargs['input_abs'], kwargs['output_abs'], kwargs['batch_abs'] = generate_io_path(model, libpath, **kwargs)

    # run in normal or debug mode
    if kwargs['batch_abs'] is None:
        command = generate_run_command(model, **kwargs)
        # run command as subprocess
        print(f"\nRunning the following command:\n{' '.join(command)}")
        subprocess.run(command, check=True, cwd=libpath)
    # run in batch mode
    else:
        cleanup_out_dir(kwargs['output_abs'])
        command = generate_batch_script(model, **kwargs)
        
        # submit batch script in output folder
        print('\nLaunching main() in batch mode ...')
        subprocess.run(['sbatch',
                        'batch_script.sh',
                        ],
                       check=True, cwd=kwargs['output_abs'])
    return command
