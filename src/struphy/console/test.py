import importlib
import os
import shutil
import subprocess

import pyccel
import yaml

import struphy
import struphy.utils.utils as utils
from struphy.console.run import subp_run
LIBPATH = struphy.__path__[0]


def struphy_test(
    group: str,
    *,
    mpi: int = 2,
    fast: bool = False,
    with_desc: bool = False,
    Tend: float = None,
    vrbose: bool = False,
    verification: bool = False,
    show_plots: bool = False,
    likwid: bool = False,
    batch: bool = False,
):
    """
    Run Struphy unit and/or model tests.

    Parameters
    ----------
    group : str
        Test identifier: "unit", "models", "fluid", "kinetic", "hybrid", "toy" or a model name.

    mpi : int
        Number of MPI processes used in tests (must be >1, default=2).

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
    if likwid:
        cmd_mpirun = ["likwid-mpirun", "-n", str(mpi), "-g", "MEM_DP", "-stats", "-marker", '-mpi', 'openmpi']
    else:
        cmd_mpirun = ["mpirun", "-n", str(mpi)]
    if "unit" in group:
        # first run only tests that require single process
        cmd = [
            "pytest",
            "-k",
            "not _models and not _tutorial and not pproc and not performance",
        ]
        if with_desc:
            cmd += ["--with-desc"]
        if vrbose:
            cmd += ["--vrbose"]
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

        # now run parallel unit tests
        cmd = cmd_mpirun
        cmd += ["pytest",
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
        cmd = cmd_mpirun
        cmd += ["pytest",
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
        cmd = cmd_mpirun
        cmd += ["pytest",
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
        if show_plots:
            cmd += ["--show-plots"]
        subp_run(cmd)

        if not verification:
            from struphy.models.tests.test_xxpproc import test_pproc_codes

        test_pproc_codes(group=group)
    elif "performance" in group:
        # Make sure likwid-mpirun and pylikwid works
        if shutil.which("likwid-mpirun") is None:
            message = """
Error: 'likwid-mpirun' not found. Please ensure LIKWID is installed and in your PATH."

On Raven/Viper:
module load gcc/14 likwid/5.3
LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib
"""
            raise RuntimeError(message)

        try:
            import pylikwid
        except ImportError:
            raise ImportError("Error: 'pylikwid' is not installed.\nPlease install it via pip:\npip install pylikwid\n")

        if batch:
            i_path, o_path, b_path = utils.get_paths()
            batch_abs = os.path.join(b_path, batch)

            # TODO: After refactoring struphy_run, we can build the output directory with one line
            # Run all models
            command = ["sbatch", batch_abs]
            subprocess.run(command, check=True)

        else:
            # Run all the models
            command = cmd_mpirun + ["python3", f"{LIBPATH}/models/tests/test_performance.py"]
            subprocess.run(command, check=True)

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
            test_mod = "test_toy_models.py"
        elif group in fluid_message:
            test_mod = "test_fluid_models.py"
        elif group in kinetic_message:
            test_mod = "test_kinetic_models.py"
        elif group in hybrid_message:
            test_mod = "test_hybrid_models.py"
        else:
            raise ValueError(f"{group} is not a valid model name.")

        py_file = os.path.join(libpath, "models", "tests", test_mod)

        cmd = [
            "mpirun",
            "-n",
            str(mpi),
            "python3",
            py_file,
            group,
            str(Tend),
            str(fast),
            str(vrbose),
            str(verification),
            str(show_plots),
        ]
        subp_run(cmd)

        if not verification:
            from struphy.models.tests.test_xxpproc import test_pproc_codes

            test_pproc_codes(group)
