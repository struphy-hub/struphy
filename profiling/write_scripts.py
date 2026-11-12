"""Write the SLURM scripts of a profiling case (or of a sweep) without submitting them.

Runs a submit script (``examples/<Model>/<case>/submit_*.py``) or a sweep script (``examples/<Model>/sweep_*.py``)
with its own arguments, in the mode ``STRUPHY_PROFILING_SCRIPT_ONLY=1`` of the driver: every ``launch`` writes the SLURM
script of its run into a new run folder of the profiling folder (``/ptmp/<user>/struphy_profiling`` on Raven, see
``clusters.SCRATCH_DIRS``) and prints its path; nothing is compiled, submitted, waited on, packaged or uploaded, so
compile the kernels first (``struphy compile``), and runs submitted this way are not packaged by the driver. The
preset is that of the detected cluster, or the one given to the submit script with ``--slurm-preset`` (needed off the
clusters, e.g. on a laptop). The scripts activate the current venv (``VIRTUAL_ENV`` must be set) and contain the paths
and the ``module`` lines of the writing machine: write them on the cluster to submit them (inspect, edit, ``sbatch``);
written elsewhere, they are for inspection only.

Usage (from anywhere, with the struphy venv activated)::

    python profiling/write_scripts.py examples/LinearMHD/itpa_tae_benchmark/submit_LinearMHD_scaling.py \\
        --variant hdiv_stiffness_24x88x8 [--ranks 16] [--steps 50] [--slurm-preset raven]
    python profiling/write_scripts.py examples/LinearMHD/sweep_LinearMHD_scaling.py
"""

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    os.environ["STRUPHY_PROFILING_SCRIPT_ONLY"] = "1"
    script = Path(sys.argv[1]).resolve()
    # the driver modules (profiling_job, sweep, ...) are imported from this folder
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
