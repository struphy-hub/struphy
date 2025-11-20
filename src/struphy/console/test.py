import os

import struphy
from struphy.utils.utils import subp_run

LIBPATH = struphy.__path__[0]


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
        
        list_of_tests = [
            f"{LIBPATH}/bsplines/tests/",
            f"{LIBPATH}/console/tests/",
            f"{LIBPATH}/feec/tests/",
            # f"{LIBPATH}/fields_background/tests/",
            # f"{LIBPATH}/geometry/tests/",
            # f"{LIBPATH}/initial/tests/",
            # f"{LIBPATH}/kinetic_background/tests/",
            # f"{LIBPATH}/linear_algebra/tests/",
            # f"{LIBPATH}/ode/tests/",
            # f"{LIBPATH}/pic/tests/",
            # f"{LIBPATH}/polar/tests/",
            # f"{LIBPATH}/post_processing/tests/",
            # f"{LIBPATH}/propagators/tests/",
            ]
        
        if mpi > 1:
            cmd = [
                "mpirun",
                "-n",
                str(mpi),
                "pytest",
                # "--testmon",
                "--with-mpi",
            ] + list_of_tests
        else:
            cmd = [
                "pytest",
                "--testmon",
            ] + list_of_tests

        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        if show_plots:
            cmd += ["--show-plots"]

        subp_run(cmd)

    elif group in {"models", "fluid", "kinetic", "hybrid", "toy"}:
        
        list_of_tests = [
            f"{LIBPATH}/models/tests/default_params/",
            ]
        
        if mpi > 1:
            cmd = [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(mpi),
                "pytest",
                "-m",
                group,
                "-s",
                # "--testmon",
                "--with-mpi",
            ] + list_of_tests
        else:
            cmd = [
                "pytest",
                "--testmon",
                # "-m",
                # group,
                # "--testmon-forceselect",
                # "-s",
            ] + list_of_tests

        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]

        subp_run(cmd)

    elif "verification" in group:
        
        list_of_tests = [
            f"{LIBPATH}/models/tests/verification/",
            ]
        
        if mpi > 1:
            cmd = [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(mpi),
                "pytest",
                "-s",
                # "--testmon",
                "--with-mpi",
            ] + list_of_tests
        else:
            cmd = [
                "pytest",
                "--testmon",
            ] + list_of_tests

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
