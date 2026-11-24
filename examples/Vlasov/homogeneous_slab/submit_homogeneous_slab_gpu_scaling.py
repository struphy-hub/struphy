"""Vlasov in a homogeneous slab: strong scaling on 1, 2 and 4 GPUs of a Pitagora booster node.

This file defines the profiling case (the `ProfilingCase`) and submits its three runs of the same
problem as `submit_homogeneous_slab_cpu_vs_gpu.py` (16 million markers by default, `--n-markers`, PushVxB and
PushEta in a constant background magnetic field, 20 steps, no field solve), on the CuPy backend with
1, 2 and 4 MPI ranks (`--gpus`), one GPU per rank (feectools binds each rank to the GPU of its node-local rank),
one thread per rank. Each run uses the `pitagora_booster` preset (one node, its 4 GPUs).
Without a batch system (or with STRUPHY_PROFILING_LOCAL=1) the runs execute directly on this machine.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_homogeneous_slab.py`
directly (its `__main__` block is the worker), and then writes the scope-profiler report and the cost
summary with `pproc_homogeneous_slab.py`.

The runs are profiled with scope-profiler (CUDA-event timing per region) and labelled
"GPU (cupy): N MPI ranks x 1 thread, N GPUs". With pseudo-random loading every rank draws the same
global sequence and keeps the markers of its domain, so all three runs push the same markers.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402


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
        "--language",
        default="fortran",
        choices=["fortran", "c"],
        help="Pyccel language to compile the Struphy kernels with.",
    )
    parser.add_argument(
        "--compiler",
        default="GNU",
        help='Pyccel compiler family ("GNU", "intel", "PGI", "nvidia" or "LLVM").',
    )
    parser.add_argument(
        "--n-markers",
        type=int,
        default=16_000_000,
        help="Total number of markers (the same on every GPU count).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        choices=[1, 2, 3, 4],
        help="GPU counts (MPI ranks, one GPU each) to run, on one booster node.",
    )
    args = parser.parse_args()
    gpus = sorted(set(args.gpus))
    gpus_text = ", ".join(str(n) for n in gpus[:-1]) + (" and " if len(gpus) > 1 else "") + str(gpus[-1])
    millions = f"{args.n_markers / 1e6:g}M"

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        # the default size keeps the label of earlier runs; other sizes are separate cases
        label="vlasov_homogeneous_slab_gpu_scaling" + ("" if args.n_markers == 16_000_000 else f"_{millions}"),
        name=f"Vlasov in a homogeneous slab, {millions} markers: strong scaling on {gpus_text} GPUs",
        description=f"The Vlasov model ({millions} markers pushed in a constant background magnetic field, PushVxB and PushEta, 20 steps, no field solve) on {gpus_text} H100 GPUs of a booster node (CuPy, one MPI rank per GPU).",
        physics_problem="Charged particles gyrating and streaming in a constant magnetic field (pure particle pushing).",
        struphy_model_used="Vlasov",
        params_source=script_dir / "params_homogeneous_slab.py",
        pproc_source=script_dir / "pproc_homogeneous_slab.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # 1, 2 and 4 MPI ranks, one GPU each, on one booster node
    param_flags = ["--n-markers", str(args.n_markers)]
    for num_tasks in gpus:
        profiling_case.launch(
            num_tasks, num_nodes=1, param_flags=param_flags, backend="cupy", slurm_preset="pitagora_booster"
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
