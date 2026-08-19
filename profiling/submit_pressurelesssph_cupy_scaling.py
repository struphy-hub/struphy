"""PressureLessSPH CuPy multi-GPU/multi-rank scaling case.

Companion to submit_guidingcenter_cupy_scaling.py and submit_vlasovampere_cupy_scaling.py:
a model with no FEEC field solve at all, the low-per-marker-compute end of the three.
Same total marker count, run with `ARRAY_BACKEND=cupy` at increasing MPI rank counts
(one rank per GPU); `--ranks 2 4 8` (default) covers intra-node scaling plus one
cross-node step.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# `ProfilingCase.launch` picks a preset from the dict it is given by cluster name
# (`detect_machine_name`), so the dict is keyed by the *detected* name here rather than
# by the preset's own name: on Pitagora detection always returns "pitagora_dcgp" (it
# cannot tell the Booster partition apart), and this case must still get the Booster
# preset. Keying on the detected name also keeps this working, without a KeyError, on a
# machine detection does not recognise (name None).
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_PressureLessSPH_scaling.py.
GPUS_PER_NODE = 4


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
        default=[2, 4, 8],
        help="MPI rank counts to run with, one GPU per rank (default: 2 4 8; 8 spans 2 Booster nodes).",
    )
    parser.add_argument(
        "--Np",
        type=int,
        default=None,
        help="Total marker count, overriding params_PressureLessSPH_scaling.py's default (10,000,000).",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "PressureLessSPH"
    params_source = params_dir / "params_PressureLessSPH_scaling.py"

    profiling_case = ProfilingCase(
        label="pressurelesssph_cupy_scaling",
        name="PressureLessSPH: CuPy scaling",
        description="PressureLessSPH particles (Np=10,000,000) on CuPy, strong-scaled across GPUs. No FEEC field solve.",
        physics_problem="SPH-discretized pressureless Euler flow with external forcing; a position push plus a velocity push against a background force field, no field solve.",
        struphy_model_used="PressureLessSPH",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    param_flags = ["--backend", "cupy"]
    if args.Np is not None:
        param_flags += ["--Np", str(args.Np)]

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
