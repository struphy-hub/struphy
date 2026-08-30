#!/usr/bin/env python3
"""Orszag--Tang CuPy multi-GPU/multi-rank scaling case.

Strong-scaling study (not a backend comparison, see
`submit_orszag_tang_numpy_vs_cupy.py` for that): the same fixed 96x96x1 problem runs
under `ARRAY_BACKEND=cupy` at increasing MPI rank counts, one rank per GPU, to measure
whether more rank+GPU pairs actually speed up a fixed-size problem. `RANKS = [1, 2, 4,
8]` covers a single-GPU baseline, intra-node scaling, and one cross-node step (8 = 2
Booster nodes x 4 GPUs).
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# Matched benchmark configuration. Keep these fixed so uploaded runs remain directly
# comparable (see params_orszag_tang_numpy_vs_cupy.py for the actual values).
NUM_ELEMENTS = 96
NUM_STEPS = 5

# `ProfilingCase.launch` picks a preset from the dict it is given by cluster name
# (`detect_machine_name`), so the dict is keyed by the *detected* name here rather than
# by the preset's own name: on Pitagora detection always returns "pitagora_dcgp" (it
# cannot tell the Booster partition apart), and this case must still get the Booster
# preset. Keying on the detected name also keeps this working, without a KeyError, on a
# machine detection does not recognise (name None).
GPU_PRESET = {
    **SLURM_PRESETS["pitagora_boost_fua_dbg"],
    "partition": "boost_fua_dbg",
    "account": "FUSIO_HLST_6",
    "mem": "16GB",
    # More than one rank means per-rank kernel compilation and MPI/NCCL setup on top
    # of the run itself, so 5 minutes (enough for the single-rank case) isn't enough --
    # both the 2- and 4-rank runs timed out here without writing any output.
    "time": "00:15:00",
}

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_orszag_tang_numpy_vs_cupy.py.
GPUS_PER_NODE = 4

# MPI rank counts to run with, one GPU per rank -- 1 is the single-GPU baseline, 2/4
# are intra-node, 8 = 2 Booster nodes.
RANKS = [1, 2, 4, 8]


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
    params_dir = script_dir / "examples" / "ViscoResistiveMHD"
    params_source = params_dir / "params_orszag_tang_numpy_vs_cupy.py"

    profiling_case = ProfilingCase(
        label="orszag_tang_cupy_scaling",
        name="Orszag-Tang: CuPy scaling",
        description=(
            f"Fixed {NUM_ELEMENTS}x{NUM_ELEMENTS}x1 Orszag--Tang run for {NUM_STEPS} "
            "timesteps on CuPy, strong-scaled across 1-8 GPUs."
        ),
        physics_problem="Regularized ideal-MHD Orszag--Tang vortex.",
        struphy_model_used="ViscoResistiveMHD",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    param_flags = ["--backend", "cupy"]

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
