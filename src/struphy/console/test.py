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

def struphy_test(group, mpi=2, fast=False, with_desc=False, Tend=None, vrbose=False, batch=False):
    """
    Run Struphy unit and/or code tests.

    Parameters
    ----------
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
    """

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
        subp_run(cmd)

        # test post processing of models
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
        subp_run(cmd)

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
            batch_abs = os.path.join(state["b_path"], batch)

            # TODO: After refactoring struphy_run, we can build the output directory with one line
            # Run all models
            command = ["sbatch", batch_abs]
            subprocess.run(command, check=True)

        else:
            # Run all the models
            likwid_cmd = ["likwid-mpirun", "-n", str(mpi), "-g", "MEM_DP", "-stats", "-marker"]  # ['']
            command = likwid_cmd + ["python3", f"{LIBPATH}/models/tests/test_performance.py"]
            subprocess.run(command, check=True)

    else:
        from struphy.models.tests import test_fluid_models, test_hybrid_models, test_kinetic_models, test_toy_models
        from struphy.models.tests.test_xxpproc import test_pproc_codes

        objs = [
            test_toy_models.test_toy,
            test_fluid_models.test_fluid,
            test_kinetic_models.test_kinetic,
            test_hybrid_models.test_hybrid,
        ]
        for obj in objs:
            try:
                obj(("Cuboid", "HomogenSlab"), fast, vrbose, model=group, Tend=Tend)
                if fast:
                    print(
                        f"Fast is enabled, mappings other than Cuboid are skipped ...",
                    )
                else:
                    obj(
                        ("HollowTorus", "AdhocTorus"),
                        fast,
                        vrbose,
                        model=group,
                        Tend=Tend,
                    )
                    obj(
                        ("Tokamak", "EQDSKequilibrium"),
                        fast,
                        vrbose,
                        model=group,
                        Tend=Tend,
                    )
            except AttributeError:
                pass

        test_pproc_codes(group)
