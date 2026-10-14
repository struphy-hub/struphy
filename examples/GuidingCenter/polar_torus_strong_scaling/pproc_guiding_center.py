"""Post-process the GuidingCenter polar torus strong scaling case: the long-time equilibrium diagnostics.

Run as ``python pproc_guiding_center.py [sim_folder] [variant flags of params_guiding_center.py]``. The variant flags must be those
of the run; the run parameters are taken from ``params_guiding_center.py``, which reads them. The
diagnostics of ``examples/particle_diagnostics.py`` are written into ``<sim_folder>/results``, where the profiling
packaging picks them up and uploads them with the run (``submit_guiding_center_polar_torus_strong_scaling.py`` calls this script after each simulation).
"""

import argparse
import sys
from pathlib import Path

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# the shared post-processing of the particle profiling cases lives in examples/
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
from particle_diagnostics import save_equilibrium_diagnostics  # noqa: E402

# The variant flags of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_guiding_center import parser  # noqa: E402


def main(path_out=DEFAULT_OUTPUT):
    out = Output(path_out)
    save_equilibrium_diagnostics(out, "kinetic_ions", scalars=("en_tot", "n_lost_particles"))


if __name__ == "__main__":
    # the variant flags are parsed by params_guiding_center.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out)
