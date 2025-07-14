from struphy.utils.utils import subp_run


def struphy_test(
    group: str,
    *,
    mpi: int = 4,
    fast: bool = False,
    with_desc: bool = False,
    Tend: float = None,
    vrbose: bool = False,
    verification: bool = False,
    show_plots: bool = False,
    nclones: int = 1,
):
    """
    Run Struphy unit and/or model tests.

    Parameters
    ----------
    group : str
        Test identifier: "unit", "models", "fluid", "kinetic", "hybrid", "toy" or a model name.

    mpi : int
        Number of MPI processes used in tests (must be >1, default=4).

    fast : bool
        Whether to test models just in slab geometry.

    with_desc : bool
        Whether to include DESC equilibrium in unit tests (mem consuming).

    Tend : float
        If group is a model name, simulation end time in units of the model (default=0.015 with dt=0.005). Data is only saved at Tend if set.

    vrbose : bool
        Show full screen output.

    verification : bool
        Whether to run verification tests specified in io/inp/tests.

    show_plots : bool
        Show plots of tests.
    """

    if "unit" in group:
        # first run only tests that require single process
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

        # now run parallel unit tests
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            "not _models and not _tutorial and not pproc",
            "--with-mpi",
        ]
        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

    elif "models" in group:
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            "_models",
            "-s",
            "--with-mpi",
        ]
        if fast:
            cmd += ["--fast"]
        if vrbose:
            cmd += ["--vrbose"]
        if verification:
            cmd += ["--verification"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

        # test post processing of models
        if not verification:
            cmd = [
                "pytest",
                "-k",
                "pproc",
                "-s",
            ]
            subp_run(cmd)

    elif group in {"fluid", "kinetic", "hybrid", "toy"}:
        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "pytest",
            "-k",
            group + "_models",
            "-s",
            "--with-mpi",
        ]
        if fast:
            cmd += ["--fast"]
        if vrbose:
            cmd += ["--vrbose"]
        if verification:
            cmd += ["--verification"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

        if not verification:
            from struphy.models.tests.test_xxpproc import test_pproc_codes

            test_pproc_codes(group=group)

    else:
        import os
        import pickle

        import struphy

        libpath = struphy.__path__[0]

        with open(os.path.join(libpath, "models", "models_message"), "rb") as fp:
            model_message, fluid_message, kinetic_message, hybrid_message, toy_message = pickle.load(
                fp,
            )

        if group in toy_message:
            mtype = "toy"
        elif group in fluid_message:
            mtype = "fluid"
        elif group in kinetic_message:
            mtype = "kinetic"
        elif group in hybrid_message:
            mtype = "hybrid"
        else:
            raise ValueError(f"{group} is not a valid model name.")

        py_file = os.path.join(libpath, "models", "tests", "util.py")

        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "python3",
            py_file,
            mtype,
            group,
            str(Tend),
            str(fast),
            str(vrbose),
            str(verification),
            str(nclones),
            str(show_plots),
        ]
        subp_run(cmd)

        if not verification:
            from struphy.models.tests.test_xxpproc import test_pproc_codes

            test_pproc_codes(group)
