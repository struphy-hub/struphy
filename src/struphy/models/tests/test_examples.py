import logging
import shutil
from pathlib import Path
import subprocess
import sys
import os

import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy.io.setup import import_parameters_py

logger = logging.getLogger("struphy")

EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples"

PARAMS_MODULES = sorted(
    p for p in EXAMPLES_DIR.rglob("*.py") if p.name != "__init__.py" and p.name.startswith("params_")
)


@pytest.mark.examples
@pytest.mark.parametrize("params_path", PARAMS_MODULES)
def test_examples(params_path: Path):
    """Run a full simulation for each example parameter file found in the examples/ directory.

    The test loads the parameter file, runs the simulation, and then
    executes post-processing Companion scripts are discovered by replacing the
    ``params_`` prefix with ``pproc_`` (post-processing).

    Parameters
    ----------
    params_path : Path
        Absolute path to a ``params_*.py`` file inside the examples directory,
        injected by the ``pytest.mark.parametrize`` decorator.
    """
    rel = params_path.with_suffix("")
    example_name = rel.parts[-1]

    pproc_name = example_name.replace("params_", "pproc_")
    pproc_path = Path(*rel.parts[:-1]) / f"{pproc_name}.py"

    regress_name = example_name.replace("params_", "regress_")
    regress_path = Path(*rel.parts[:-1]) / f"{regress_name}.py"

    logger.info(f"\n{MPI.COMM_WORLD.Get_rank()} Testing example: {example_name}")
    logger.info(f"{params_path = }")
    logger.info(f"{pproc_path = }")
    logger.info(f"{regress_path = }")

    params = import_parameters_py(str(params_path), name=example_name)
    params.sim.run(one_time_step=True)

    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        for key in list(env):
            if key.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_")):
                env.pop(key)
        if pproc_path.exists():

            subprocess.run(
                [sys.executable, str(pproc_path)],
                check=True,
                cwd=os.getcwd(),
                env=env,
            )

        if regress_path.exists():
            subprocess.run(
                [sys.executable, str(regress_path)],
                check=True,
                cwd=os.getcwd(),
                env=env,
            )

        shutil.rmtree(params.sim.env.path_out)
    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    logger.info(f"{PARAMS_MODULES = }")
    test_examples(params_path=PARAMS_MODULES[4])
