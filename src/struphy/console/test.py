from struphy.utils.utils import subp_run


def struphy_test(
    group: str,
    *,
    mpi: int = 1,
    with_desc: bool = False,
    vrbose: bool = False,
    show_plots: bool = False,
    nclones: int = 1,
):
    """
    Run Struphy unit and/or model tests.

    Parameters
    ----------
    group : str
        Test identifier: "unit", "models", "fluid", "kinetic", "hybrid", "toy", "verification" or a model name.

    mpi : int
        Number of MPI processes used in tests (default=1).

    with_desc : bool
        Whether to include DESC equilibrium in unit tests (mem consuming).

    Tend : float
        If group is a model name, simulation end time in units of the model (default=0.015 with dt=0.005). Data is only saved at Tend if set.

    vrbose : bool
        Show full screen output.

    show_plots : bool
        Show plots of tests.
    """

    if "unit" in group:
        if mpi > 1:
            cmd = [
                "mpirun",
                "-n",
                str(mpi),
                "pytest",
                "-k",
                "not _models and not _tutorial and not pproc",
                "--with-mpi",
            ]
        else:
            cmd = [
                "pytest",
                "-k",
                "not _models and not _tutorial and not pproc",
            ]

        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        if show_plots:
            cmd += ["--show-plots"]

        subp_run(cmd)

    elif group in {"models", "fluid", "kinetic", "hybrid", "toy"}:
        if mpi > 1:
            cmd = [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(mpi),
                "pytest",
                "-k",
                "_models",
                "-m",
                group,
                "-s",
                "--with-mpi",
            ]
        else:
            cmd = [
                "pytest",
                "-k",
                "_models",
                "-m",
                group,
                "-s",
            ]

        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

    elif "verification" in group:
        if mpi > 1:
            cmd = [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(mpi),
                "pytest",
                "-k",
                "_verif_",
                "-s",
                "--with-mpi",
            ]
        else:
            cmd = [
                "pytest",
                "-k",
                "_verif_",
                "-s",
            ]

        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

    else:
        cmd = [
            "mpirun",
            "--oversubscribe",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            "_models",
            "-m",
            "single",
            "-s",
            "--with-mpi",
            "--model-name",
            group,
        ]
        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)
