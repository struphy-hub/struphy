"""Poisson on a 3D cube: strong scaling on 1, 2 and 4 GPUs of a Pitagora booster node.

This file defines the profiling case (the `ProfilingCase`) and submits its three runs of the same
problem as `submit_gaussian_cube_cpu_vs_gpu.py` (manufactured Gaussian solution on a Cuboid,
degree (1, 2, 3), unpreconditioned CG, 5 solves), by default on 192 x 192 x 192 cells (`--cells`), on the
CuPy backend with 1, 2 and 4 MPI ranks, one GPU per rank (feectools binds each rank to the GPU of its
node-local rank), one thread per rank. `--solver pcg_sr` selects the CG variant with one global reduction
per iteration. The runs write only their profiling data (`--no-output` of the params file), so there is no
post-processing. Each run uses the `pitagora_booster` preset (one node, its 4 GPUs).
Without a batch system (or with STRUPHY_PROFILING_LOCAL=1) the runs execute directly on this machine.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_gaussian_cube.py`
directly (its `__main__` block is the worker).

The runs are profiled with scope-profiler (CUDA-event timing per region) and labelled
"GPU: N MPI ranks, N GPUs".
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
        "--cells",
        type=int,
        default=192,
        help="Number of cells in each direction (192^3 still fits on one H100).",
    )
    parser.add_argument(
        "--solver",
        default="pcg",
        choices=["pcg", "pcg_sr"],
        help="CG variant: 'pcg_sr' does one global reduction per iteration instead of two.",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label="poisson_cube_gpu_scaling",
        name="Poisson on 3D cube: strong scaling on 1, 2 and 4 GPUs",
        description=f"The Poisson model with manufactured solution on a 3D cube (Cuboid, {args.cells}^3 cells, degree (1, 2, 3)), 5 solves with unpreconditioned CG ('{args.solver}'), on 1, 2 and 4 H100 GPUs of a booster node (CuPy, one MPI rank per GPU).",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=script_dir / "params_gaussian_cube.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # 5 solves, as in submit_gaussian_cube_cpu_vs_gpu.py; only the profiling data is written
    cells = [str(args.cells)] * 3
    param_flags = ["--num-elements", *cells, "--steps", "5", "--solver", args.solver, "--gpu-timing", "--no-output"]

    # 1, 2 and 4 MPI ranks, one GPU each, on one booster node
    for num_tasks in (1, 2, 4):
        profiling_case.launch(
            num_tasks,
            num_nodes=1,
            param_flags=param_flags,
            backend="cupy",
            slurm_preset="pitagora_booster",
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
