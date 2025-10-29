import os
import pytest
import struphy
from struphy.utils.utils import subp_run

LIBPATH = struphy.__path__[0]

import pytest
from pathlib import Path

def collect_changed_tests(test_path):
    """Collect changed tests using pytest-testmon, returning full absolute nodeids."""

    collected = []

    class Collector:
        def pytest_collection_modifyitems(self, session, config, items):
            for item in items:
                abs_path = Path(item.fspath).resolve()
                # Keep the test name and parameters after '::'
                node_suffix = item.nodeid.split("::", 1)[1] if "::" in item.nodeid else ""
                if node_suffix:
                    full_nodeid = f"{abs_path}::{node_suffix}"
                else:
                    full_nodeid = str(abs_path)
                collected.append(full_nodeid)
            # Stop after collection phase
            raise pytest.UsageError("COLLECT_ONLY")

    args = ["--testmon", "--collect-only", "-q", str(test_path)]
    print(args)

    try:
        pytest.main(args, plugins=[Collector()])
    except pytest.UsageError:
        print('exit')
        pass  # Expected exit after collection

    return collected

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

        tests = collect_changed_tests(
            f"{LIBPATH}/tests/unit/bsplines/"
        )

        print(f"{tests = }")

        if len(tests) == 0:
            print("No changed tests detected!")
            return

        if mpi > 1:
            cmd = [
                "mpirun",
                "-n",
                str(mpi),
                "pytest",
                # "--testmon",
                "--with-mpi",
            ]
        else:
            cmd = [
                "pytest",
                "--testmon",
            ]

        cmd += tests

        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        if show_plots:
            cmd += ["--show-plots"]

        # Run in the current directory
        cwd = os.getcwd()
        subp_run(cmd)

    elif group in {"models", "fluid", "kinetic", "hybrid", "toy"}:
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
                f"{LIBPATH}/tests/models/",
            ]
        else:
            cmd = [
                "pytest",
                "-m",
                group,
                "-s",
                "--testmon",
                f"{LIBPATH}/tests/models/",
            ]

        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]

        # Run in the current directory
        cwd = os.getcwd()
        subp_run(cmd)

    elif "verification" in group:
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
                f"{LIBPATH}/tests/verification/",
            ]
        else:
            cmd = [
                "pytest",
                "-s",
                "--testmon",
                f"{LIBPATH}/models/tests/verification/",
            ]

        if vrbose:
            cmd += ["--vrbose"]
        if nclones > 1:
            cmd += ["--nclones", f"{nclones}"]
        if show_plots:
            cmd += ["--show-plots"]

        # Run in the current directory
        cwd = os.getcwd()
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
            # "--testmon",
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

        # Run in the current directory
        cwd = os.getcwd()
        subp_run(cmd)
