import os
import shutil
import subprocess
import sys

import yaml

import struphy
import struphy.utils.utils as utils
from struphy.utils.utils import subp_run

libpath = struphy.__path__[0]


def struphy_run(
    model=None,
    inp=None,
    input_abs=None,
    output="sim_1",
    output_abs=None,
    batch=None,
    batch_abs=None,
    runtime=300,
    save_step=1,
    sort_step=0,
    restart=False,
    mpi=1,
    nclones=1,
    cprofile=False,
    verbose=False,
    likwid=False,
    nperdomain=None,
    stats=None,
    marker=None,
    hpcmd_suspend=None,
    likwid_repetitions=1,
    group="MEM_DP",
    time_trace=False,
    sample_duration=1.0,
    sample_interval=1.0,
):
    """Run a Struphy model: prepare arguments, output folder and execute main().

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

    nclones : int
        Number of domain clones.

    verbose : bool
        Show full screen output.

    cprofile : bool
        Whether to run with Cprofile (slower).

    likwid : bool
        Whether to run with Likwid (Needs to be installed first). Default is False.

    nperdomain
    stats=None,
    marker=None,
    hpcmd_suspend=None,


    likwid_repetitions : int, optional
        Number of repetitions for Likwid profiling. Default is 1.
    """

    # Read struphy state file
    state = utils.read_state()

    # Struphy paths
    i_path, o_path, b_path = utils.get_paths(state=state)

    assert os.path.exists(i_path), f"The path '{i_path}' does not exist. Set path with `struphy --set-i PATH`"
    if batch is not None or batch_abs is not None:
        assert os.path.exists(b_path), f"The path '{b_path}' does not exist. Set path with `struphy --set-b PATH`"

    # create absolute i/o paths
    input_abs, output_abs, batch_abs = generate_absolute_io_paths(
        model,
        inp,
        input_abs,
        output,
        output_abs,
        batch,
        batch_abs,
        restart,
    )

    # Read likwid params
    if likwid:
        # if likwid_inp is None and likwid_input_abs is None:
        #     # use default likwid parameters
        likwid_command = ["likwid-mpirun", "-n", str(mpi), "-g", group, "-mpi", "openmpi"]
        if nperdomain:
            likwid_command += ["-nperdomain", nperdomain]
        if stats:
            likwid_command += ["-stats"]
        if marker:
            likwid_command += ["-marker"]

    # command parts
    cmd_python = ["python3"]
    cmd_main = [f"{libpath}/main.py"]
    if model is not None:
        cmd_main += [model]
    cmd_main += [
        "-i",
        input_abs,
        "-o",
        output_abs,
        "--runtime",
        str(runtime),
        "-s",
        str(save_step),
        "--sort-step",
        str(sort_step),
        "--nclones",
        str(nclones),
    ]
    if time_trace:
        cmd_main += [
            "--time-trace",
            "--sample-duration",
            str(sample_duration),
            "--sample-interval",
            str(sample_interval),
        ]
    if verbose:
        cmd_main += ["-v"]

    cmd_cprofile = ["-m", "cProfile", "-o", os.path.join(output_abs, "profile_tmp"), "-s", "time"]

    # run in normal or debug mode
    if batch_abs is None:
        if likwid:
            command = likwid_command + cmd_python + cprofile * cmd_cprofile + cmd_main + ["--likwid"]
        else:
            print("\nLaunching main() in normal mode ...")
            command = ["mpirun", "-n", str(mpi)] + cmd_python + cprofile * cmd_cprofile + cmd_main

        # add restart flag
        if restart:
            command += ["-r"]

        print(f"\nCprofile turned {'on' if cprofile else 'off'}.")

        # run command as subprocess
        if likwid:
            subp_run(command, cwd=None)
        else:
            subp_run(command)

    # run in batch mode
    else:
        cleanup_batch_environment(output_abs)

        # copy batch script to output folder
        batch_abs_new = os.path.join(output_abs, "batch_script.sh")
        shutil.copy2(batch_abs, batch_abs_new)

        # delete srun command from batch script
        with open(batch_abs_new, "r") as f:
            lines = f.readlines()
            if "srun" in lines[-1]:
                lines = lines[:-2]

        with open(batch_abs_new, "w") as f:
            for line in lines:
                f.write(line)
            f.write("# Run command added by Struphy\n")

            command = cmd_python + cprofile * cmd_cprofile + cmd_main
            if restart:
                command += ["-r"]

            if likwid:
                command = likwid_command + command + ["--likwid"]

                print(f"Running with likwid with {likwid_repetitions =}")
                f.write(f"# Launching likwid {likwid_repetitions} times with likwid-mpirun\n")
                for i in range(likwid_repetitions):
                    f.write(f"\n\n# Run number {i + 1:03}\n")
                    f.write(" ".join(command) + " > " + os.path.join(output_abs, f"struphy_likwid_{i:03}.out"))
            else:
                print("Running with srun")
                command = ["srun"] + command
                f.write(" ".join(command) + " > " + os.path.join(output_abs, "struphy.out"))

        # submit batch script in output folder
        print("\nLaunching main() in batch mode ...")
        cmd = ["sbatch", "batch_script.sh"]
        subp_run(cmd, cwd=output_abs)
    return command


def generate_absolute_io_paths(
    model,
    inp,
    input_abs,
    output,
    output_abs,
    batch,
    batch_abs,
    restart,
):
    # Read struphy state file
    state = utils.read_state()

    i_path, o_path, b_path = utils.get_paths(state=state)

    if input_abs is None:
        if inp is None:
            if model is None:
                print("You have to either specify a struphy model or a parameter file which contains the model.")
                sys.exit(0)
            default_yml = os.path.join(i_path, f"params_{model}.yml")
            if os.path.isfile(default_yml):
                print(f"\nNo input file specified, running with default: {default_yml}")
                input_abs = default_yml
            else:
                # load model class
                from struphy.models import fluid, hybrid, kinetic, toy

                objs = [fluid, kinetic, hybrid, toy]
                for obj in objs:
                    try:
                        model_class = getattr(obj, model)
                    except AttributeError:
                        pass

                params = model_class.generate_default_parameter_file()
                sys.exit(0)
        else:
            input_abs = os.path.join(i_path, inp)

    if output_abs is None:
        output_abs = os.path.join(o_path, output)

    if batch_abs is None:
        if batch is not None:
            batch_abs = os.path.join(b_path, batch)

    # take existing parameter file for restart
    if restart:
        input_abs = os.path.join(output_abs, "parameters.yml")

    return input_abs, output_abs, batch_abs


def cleanup_batch_environment(output_abs):
    # create output folder if it does not exit
    if not os.path.exists(output_abs):
        os.mkdir(output_abs)
        os.mkdir(os.path.join(output_abs, "data/"))

    # remove sim.out file
    file = os.path.join(output_abs, "sim.out")
    if os.path.exists(file):
        os.remove(file)
        print("Removed file " + file)

    # remove sim.err file
    file = os.path.join(output_abs, "sim.err")
    if os.path.exists(file):
        os.remove(file)
        print("Removed file " + file)

    # remove old batch script
    file = os.path.join(output_abs, "batch_script.sh")
    if os.path.exists(file):
        os.remove(file)
        print("Removed file " + file)

    # remove struphy.out file
    file = os.path.join(output_abs, "sim.out")
    if os.path.exists(file):
        os.remove(file)
        print("Removed file " + file)
