"""Guiding-centre NumPy-vs-CuPy profiling case.

This file defines the guiding-centre backend-comparison profiling case (the `ProfilingCase`)
and submits it: the same simulation is run twice, once with `ARRAY_BACKEND=numpy` on a
CPU partition and once with `ARRAY_BACKEND=cupy` on a GPU partition, so the two runs can
be compared directly. For each run, `ProfilingCase.launch` builds and submits a SLURM
script, or, without a batch system, runs directly on this machine. `finalize_run` then
packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_GuidingCenter.py`
directly (its `__main__` block is the worker), with `--backend numpy` or `--backend cupy`.

`GuidingCenter` is used here rather than `LinearMHDDriftkineticCC` because its runtime is
actually dominated by the particle kernels this comparison is meant to measure. Its whole
propagator stack is CUDA-ported and it has no FEEC field solve, so the backend difference
shows up in the total wall clock. `LinearMHDDriftkineticCC` is dominated by MHD field
propagators and one-off setup instead, which masks any particle-kernel speedup.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# Which SLURM preset each backend runs under. `ProfilingCase.launch` picks a preset from
# the dict it is given by cluster name (`detect_machine_name`), so the dict is keyed by
# the *detected* name here rather than by the preset's own name: on Pitagora detection
# always returns "pitagora_dcgp" (it cannot tell the Booster partition apart), and the
# GPU run must still get the Booster preset. Keying on the detected name also keeps this
# working, without a KeyError, on a machine detection does not recognise (name None).
CPU_PRESET = SLURM_PRESETS["pitagora_dcgp"]
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

BACKEND_PRESETS = {
    "numpy": CPU_PRESET,
    "cupy": GPU_PRESET,
}

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). The CuPy
# runs are spread so that no node holds more ranks than it has GPUs.
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
        default=[1],
        help=(
            "MPI rank counts to run each backend with (default: 1). Note that the CuPy "
            "runs currently select no GPU per rank, so more than one rank per node all "
            "share device 0."
        ),
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "GuidingCenter"
    params_source = params_dir / "params_GuidingCenter.py"

    profiling_case = ProfilingCase(
        label="guidingcenter_numpy_vs_cupy",
        name="Guiding-centre particles on cube, NumPy vs CuPy",
        description="5D guiding-centre test particles in a homogeneous slab on a 3D cube, run with the NumPy and the CuPy array backend. Runtime is dominated by the (CUDA-ported) particle pushers.",
        physics_problem="Guiding-centre drift-kinetic particle motion; the particle-push hot loop common to all PIC/drift-kinetic models.",
        struphy_model_used="GuidingCenter",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    # Launch one run per (rank count, backend) pair.
    for num_tasks in args.ranks:
        for backend, preset in BACKEND_PRESETS.items():
            if backend == "cupy":
                # One node per `GPUS_PER_NODE` ranks. `launch` would otherwise derive the
                # node count from `cpus_per_node`, which on a GPU partition packs far more
                # ranks per node than there are GPUs.
                num_nodes = -(-num_tasks // GPUS_PER_NODE)
            else:
                # Let `launch` derive the node count from the cluster's CPU count.
                num_nodes = None

            profiling_case.launch(
                num_tasks,
                num_nodes=num_nodes,
                param_flags=["--backend", backend],
                slurm_presets={cluster_name: preset},
            )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
