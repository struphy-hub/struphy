#!/usr/bin/env python3
"""Orszag--Tang CuPy multi-GPU strong-scaling case (2, 4 and 8 GPUs).

The same fixed 288x288x1 problem runs under `ARRAY_BACKEND=cupy` at increasing MPI rank
counts, one rank per GPU, to measure whether more rank+GPU pairs actually speed up a
fixed-size problem. `RANKS = [2, 4, 8]` covers intra-node scaling (2 and 4 GPUs on one
Booster node) and one cross-node step (8 = 2 nodes x 4 GPUs).

Why 288x288x1 rather than the 96x96x1 of `submit_orszag_tang_cupy_scaling.py`: a run
carries a rank-independent fixed cost -- setup, kernel compilation, and the per-step
Python/kernel-launch overhead that does not shrink when the local array does. Measured
on Booster H100s by varying the grid on a single GPU (7.8 s at 24^2, 13.7 s at 48^2,
30.0 s at 96^2, i.e. 16x the work for 3.9x the time), that fixed cost is ~6.3 s. At
96x96x1 it is a fifth of the runtime and caps the 2->4 GPU speedup at ~1.5x however fast
communication gets; at 288x288x1 it is a few percent, so what is measured is the solver's
parallel efficiency rather than start-up overhead. 288 is also the largest size that
still fits in the two GPUs of the smallest configuration (384x384x1 runs out of memory
there), which is what sets the upper end of this study.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# Matched benchmark configuration. Keep these fixed so uploaded runs remain directly
# comparable (see params_orszag_tang_cupy_scaling.py for the actual values).
NUM_ELEMENTS = 288
NUM_STEPS = 5

# `ProfilingCase.launch` picks a preset from the dict it is given by cluster name
# (`detect_machine_name`), so the dict is keyed by the *detected* name here rather than
# by the preset's own name: on Pitagora detection always returns "pitagora_dcgp" (it
# cannot tell the Booster partition apart), and this case must still get the Booster
# preset. Keying on the detected name also keeps this working, without a KeyError, on a
# machine detection does not recognise (name None).
GPU_PRESET = {
    **SLURM_PRESETS["pitagora_boost_fua_prod"],
    "partition": "boost_fua_prod",
    "account": "FUSIO_HLST_6",
    "mem": "100GB",
    # The 2-GPU run -- the slowest of the three -- takes ~3.5 minutes of stepping at this
    # grid size, on top of per-rank kernel compilation and MPI setup. 30 minutes leaves
    # room for that without relying on the debug partition's shorter limit.
    "time": "00:30:00",
}

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_orszag_tang_cupy_scaling.py.
GPUS_PER_NODE = 4

# MPI rank counts to run with, one GPU per rank -- 2 and 4 are intra-node, 8 = 2 Booster
# nodes, which is the step that also exercises inter-node communication.
RANKS = [2, 4, 8]


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
    params_source = params_dir / "params_orszag_tang_cupy_scaling.py"

    profiling_case = ProfilingCase(
        label="orszag_tang_cupy_gpu_scaling",
        name="Orszag-Tang: CuPy GPU scaling",
        description=(
            f"Fixed {NUM_ELEMENTS}x{NUM_ELEMENTS}x1 Orszag--Tang run for {NUM_STEPS} "
            "timesteps on CuPy, strong-scaled across 2-8 GPUs."
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
