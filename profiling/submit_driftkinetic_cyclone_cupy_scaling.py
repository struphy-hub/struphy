"""DriftKineticElectrostaticAdiabatic (ITG cyclone) CuPy multi-GPU/multi-rank scaling case.

Strong-scaling study (not a backend comparison, see `submit_driftkinetic_cyclone_numpy_vs_cupy.py`
for that): the same grid/marker configuration runs under `ARRAY_BACKEND=cupy` at
increasing MPI rank counts, one rank per GPU. `--ranks 1 2 4 8` (default) covers
intra-node scaling plus one cross-node step. Grid is hardcoded to `NUM_ELEMENTS` below
(not a CLI flag), so a run's grid is always readable straight from this file; `(12, 32,
4)` -- smaller than `params_cyclone.py`'s own default -- was chosen so the `direct`
solver's one-time matrix-assembly cost (see `feectools.linalg.utilities.tosparse_via_matvec`)
reliably fits `pitagora_boost_fua_dbg`'s walltime. Uses `params_cyclone.py`'s default
solver (`direct`) at every rank count, now that `DirectSolver` supports `nprocs > 1`.
`Tend` is shortened to 3 steps to leave walltime headroom for that one-time cost.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# The preset is looked up by cluster name inside `launch`, so the dict is keyed by the
# *detected* name here rather than by the preset's own name: on Pitagora detection
# always returns "pitagora_dcgp" (it cannot tell the Booster partition apart), and this
# case must still get the Booster preset. Keying on the detected name also keeps this
# working, without a KeyError, on a machine detection does not recognise (name None).
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_cyclone.py.
GPUS_PER_NODE = 4

# Grid resolution, hardcoded rather than a `--num-elements` CLI flag -- see the module
# docstring for why this specific size was chosen.
NUM_ELEMENTS = (12, 32, 4)


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="MPI rank counts to run with, one GPU per rank (default: 1 2 4 8; 8 spans 2 Booster nodes).",
    )
    parser.add_argument("--ppc", type=int, default=None, help="Markers per cell (overrides params_cyclone.py's default, 200).")
    parser.add_argument(
        "--Tend",
        type=float,
        default=0.003,
        help=(
            "End time (default: 0.003 -> 3 steps, shortened from params_cyclone.py's own "
            "0.01/10 steps to leave walltime headroom for the direct solver's one-time "
            "matrix-assembly cost at every rank count; see the module docstring)."
        ),
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "DriftKineticElectrostaticAdiabatic"
    params_source = params_dir / "params_cyclone.py"

    profiling_case = ProfilingCase(
        label="driftkinetic_cyclone_cupy_scaling",
        name="ITG cyclone: CuPy scaling",
        description=(
            "Cyclone-instability ITG turbulence (DriftKineticElectrostaticAdiabatic) on "
            "CuPy, strong-scaled across GPUs with the direct field solver."
        ),
        physics_problem="Electrostatic drift-kinetic ITG turbulence with adiabatic electrons in toroidal geometry.",
        struphy_model_used="DriftKineticElectrostaticAdiabatic",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    param_flags = [
        "--backend", "cupy",
        "--Tend", str(args.Tend),
        "--num-elements", *[str(n) for n in NUM_ELEMENTS],
    ]
    if args.ppc is not None:
        param_flags += ["--ppc", str(args.ppc)]

    # Launch one run per rank count. One node per GPUS_PER_NODE ranks -- `launch` would
    # otherwise derive the node count from `cpus_per_node`, which on a GPU partition packs
    # far more ranks per node than there are GPUs.
    for num_tasks in args.ranks:
        num_nodes = -(-num_tasks // GPUS_PER_NODE)
        profiling_case.launch(
            num_tasks,
            num_nodes=num_nodes,
            param_flags=param_flags,
            slurm_presets={cluster_name: GPU_PRESET},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
