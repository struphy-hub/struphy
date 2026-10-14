"""Post-process the GuidingCenter tokamak_equilibrium example: the long-time equilibrium diagnostics.

Run as ``python pproc_tokamak_equilibrium.py [sim_folder] [variant flags of params_tokamak_equilibrium.py]``. The variant flags must be those
of the run; the run parameters are taken from ``params_tokamak_equilibrium.py``, which reads them. The
diagnostics of ``struphy.post_processing.particle_diagnostics`` are written into ``<sim_folder>/results``, where the profiling
packaging picks them up and uploads them with the run (``submit_tokamak_equilibrium.py`` calls this script after each simulation).
"""

import argparse
import sys
from pathlib import Path

from struphy import Output
from struphy.post_processing.particle_diagnostics import save_equilibrium_diagnostics

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"


# The variant flags of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_tokamak_equilibrium import parser  # noqa: E402


def main(path_out=DEFAULT_OUTPUT):
    out = Output(path_out)
    save_equilibrium_diagnostics(out, "kinetic_ions", scalars=("en_tot", "n_lost_particles"))


if __name__ == "__main__":
    # the variant flags are parsed by params_tokamak_equilibrium.py, the rest here
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
