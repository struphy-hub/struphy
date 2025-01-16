import subprocess

import struphy


def struphy_run(
    model,
    inp=None,
    input_abs=None,
    output="sim_1",
    output_abs=None,
    batch=None,
    batch_abs=None,
    batch_auto=None,
    runtime=300,
    save_step=1,
    sort_step=0,
    restart=False,
    mpi=1,
    debug=False,
    cprofile=False,
    verbose=False,
    likwid=False,
    likwid_inp=None,
    likwid_input_abs=None,
    likwid_repetitions=1,
    group="MEM_DP",
):
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

    verbose : bool
        Show full screen output.

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

    import os
    import shutil
    import subprocess

    import yaml

    import struphy
    import struphy.utils.utils as utils
    from struphy.console.utils import generate_batch_script, save_batch_script

    libpath = struphy.__path__[0]

    # Read struphy state file
    state = utils.read_state()

    # Struphy paths
    i_path = state["i_path"]
    o_path = state["o_path"]
    b_path = state["b_path"]

    assert os.path.exists(i_path), f"The path '{i_path}' does not exist. Set path with `struphy --set-i PATH`"
    if batch is not None or batch_abs is not None:
        assert os.path.exists(b_path), f"The path '{b_path}' does not exist. Set path with `struphy --set-b PATH`"

    # create absolute i/o paths
    if input_abs is None:
        if inp is None:
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
        input_abs = os.path.join(output_abs, "parameters.yml")

    # Read likwid params
    if likwid:
        if likwid_inp is None and likwid_input_abs is None:
            # use default likwid parameters
            likwid_command = ["likwid-mpirun", "-n", str(mpi), "-g", group, "-stats", "-marker"]
        else:
            if likwid_inp is not None:
                likwid_input_abs = os.path.join(i_path, likwid_inp)

            with open(likwid_input_abs, "r") as file:
                config = yaml.safe_load(file)

            # Get the command from the configuration
            command_base = config.get("command", None)
            if not command_base:
                print("Missing required configuration: 'command'")
                exit(1)

            likwid_config = config.get(command_base, {})

            # Get the options list
            options = likwid_config.get("options", [])

            # Flatten the options list
            flattened_options = ["-np", str(mpi)]
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
    cmd_python = ["python3"]
    cmd_main = [
        "main.py",
        model,
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
    ]
    if verbose:
        cmd_main += ["-v"]

    cmd_cprofile = ["-m", "cProfile", "-o", os.path.join(output_abs, "profile_tmp"), "-s", "time"]

    # run in normal or debug mode
    if batch_abs is None:
        if debug:
            print("\nLaunching main() in Cobra debug mode ...")
            command = (
                [
                    "srun",
                    "-n",
                    str(mpi),
                    "-p",  # interactive commands
                    "interactive",
                    "--time",
                    "119",
                    "--mem",
                    "2000",
                ]
                + cmd_python
                + cprofile * cmd_cprofile
                + cmd_main
            )
        elif likwid:
            cmd_main[0] = f"{libpath}/{cmd_main[0]}"
            command = likwid_command + cmd_python + cprofile * cmd_cprofile + cmd_main + ["--likwid"]
        else:
            print("\nLaunching main() in normal mode ...")
            command = ["mpirun", "-n", str(mpi)] + cmd_python + cprofile * cmd_cprofile + cmd_main

        # add restart flag
        if restart:
            command += ["-r"]

        if cprofile:
            print("\nCprofile turned on.")
        else:
            print("\nCprofile turned off.")

        # run command as subprocess
        if likwid:
            subp_run(command, cwd=None)
        else:
            subp_run(command)

    # run in batch mode
    else:
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

        # copy batch script to output folder
        batch_abs_new = os.path.join(output_abs, "batch_script.sh")
        if batch_auto:
            batch_auto = "raven"
            sbatch_params = {
                "raven": {
                    "ntasks_per_node": 8,
                    "module_setup": "module load anaconda/3/2023.03 gcc/12 openmpi/4.1 likwid/5.2",
                    "likwid": likwid,
                },
                "cobra": {
                    "ntasks_per_node": 72,
                    "likwid": likwid,
                },
                "viper": {
                    "ntasks_per_node": 4,
                    "likwid": likwid,
                },
            }

            batch_script = generate_batch_script(**sbatch_params[batch_auto])
            save_batch_script(batch_script, "test_script.sh")
            exit()
            with open(batch_abs_new, "w") as f:
                f.write(batch_script)
        else:
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

            command = cmd_python + cprofile * cmd_cprofile + [f"{libpath}/{' '.join(cmd_main)}"]
            if restart:
                command += ["-r"]

            if likwid:
                command = likwid_command + command + ["--likwid"]

            if likwid:
                print(f"Running with likwid with {likwid_repetitions = }")
                f.write(f"# Launching likwid {likwid_repetitions} times with likwid-mpirun\n")
                for i in range(likwid_repetitions):
                    f.write(f"\n\n# Run number {i:03}\n")
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


def subp_run(cmd, cwd="libpath", check=True):
    """Call subprocess.run and print run command."""

    if cwd == "libpath":
        cwd = struphy.__path__[0]

    print(f"\nRunning the following command as a subprocess:\n{' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)
