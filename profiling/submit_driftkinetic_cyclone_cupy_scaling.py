"""DriftKineticElectrostaticAdiabatic (ITG cyclone) CuPy multi-GPU/multi-rank scaling case.

Strong-scaling study (not a backend comparison, see `submit_driftkinetic_cyclone_numpy_vs_cupy_pcg.py`
for that): the same grid/marker configuration runs under `ARRAY_BACKEND=cupy` at
increasing MPI rank counts, one rank per GPU. `--ranks 1 2 4 8` (default) covers
intra-node scaling plus one cross-node step. Grid is hardcoded to `NUM_ELEMENTS` below
(not a CLI flag), so a run's grid is always readable straight from this file.

**Solver forced to `pcg`, not `params_cyclone.py`'s own default (`direct`).**
`DirectSolver` now supports `nprocs > 1` (a replicated matrix assembly, see
`feectools.linalg.utilities.tosparse_via_matvec`), but that assembly is currently too
slow under CuPy in practice (measured ~270-300s one-time cost even at a modest grid,
dominated by per-`dot()`-call kernel-launch/sync overhead the array-transfer
optimization only dents) to be worth using in a scaling study yet -- `pcg` gives a
cleaner, apples-to-apples comparison across rank counts until that's fixed.
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

# Grid resolution, matching params_cyclone.py's own default.
NUM_ELEMENTS = (16, 64, 4)

# MPI rank counts to run with, one GPU per rank -- 1/2/4 intra-node, 8 = 2 Booster nodes.
RANKS = [1, 2, 4, 8]

# End time: 0.003 -> 3 steps, shortened from params_cyclone.py's own 0.01/10 steps to
# keep the pcg-forced sweep quick.
TEND = 0.003


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
            "CuPy, strong-scaled across GPUs. Solver forced to pcg (direct's multi-rank "
            "assembly is not fast enough yet, see module docstring)."
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
        "--solver", "pcg",
        "--Tend", str(TEND),
        "--num-elements", *[str(n) for n in NUM_ELEMENTS],
    ]

    # Launch one run per rank count. One node per GPUS_PER_NODE ranks -- `launch` would
    # otherwise derive the node count from `cpus_per_node`, which on a GPU partition packs
    # far more ranks per node than there are GPUs.
    for num_tasks in RANKS:
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
