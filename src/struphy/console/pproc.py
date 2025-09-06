from struphy.utils.utils import subp_run


def struphy_pproc(
    dirs,
    dir_abs=None,
    step=1,
    celldivide=1,
    physical=False,
    guiding_center=False,
    classify=False,
    no_vtk=False,
    time_trace=False,
):
    """Post process data from finished Struphy runs.

    Parameters
    ----------
    dirs : str
        Paths of simulation output folders relative to <struphy_path>/io/out.

    dir_abs : str
        Absolute path to the simulation output folder.

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    celldivide : int, optional
        Number of grid point in each cell used to create vtk files (default=1).

    physical : bool
        Wether to do post-processing into push-forwarded physical (xyz) components of fields.

    guiding_center : bool
        Compute guiding-center coordinates (only from Particles6D).

    classify : bool
        Classify guiding-center trajectories (passing, trapped or lost).
    """
    import os

    import struphy
    import struphy.utils.utils as utils

    # Read struphy state file
    libpath = struphy.__path__[0]
    state = utils.read_state(libpath)

    o_path = state["o_path"]

    use_state_o_path = True
    if dir_abs is not None:
        dirs = [dir_abs]
        use_state_o_path = False

    for dir in dirs:
        # create absolute path
        if use_state_o_path:
            path_to_simulation = os.path.join(o_path, dir)
        else:
            path_to_simulation = dir

        print(f"Post processing data in {path_to_simulation}")

        command = [
            "python3",
            "post_processing/pproc_struphy.py",
            path_to_simulation,
            "-s",
            str(step),
            "--celldivide",
            str(celldivide),
        ]

        if physical:
            command += ["--physical"]

        if guiding_center:
            command += ["--guiding-center"]

        if classify:
            command += ["--classify"]

        # Whether vtk files should be created
        if no_vtk:
            command += ["--no-vtk"]

        if time_trace:
            command += ["--time-trace"]

        subp_run(command)
