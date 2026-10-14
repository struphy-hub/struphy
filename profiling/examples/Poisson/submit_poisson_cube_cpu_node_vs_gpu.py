"""Poisson on a 3D cube: 28 CPU MPI ranks vs 1 GPU, on the same kind of Pitagora booster node.

This file defines the profiling case (the `ProfilingCase`) and submits its two runs of the
problem of `submit_poisson_cube_strong_scaling.py` (manufactured Gaussian solution on a Cuboid,
degree (1, 2, 3), unpreconditioned CG), at 128 x 128 x 128 cells and with 5 time steps, i.e. 5
Poisson solves with the mass matrices assembled once in the setup, as in a time-dependent simulation:

- CPU: 28 MPI ranks with one thread each (OMP_NUM_THREADS=1), NumPy backend;
- GPU: 1 MPI rank with one thread on one H100, CuPy backend.

Every rank gets one CPU core (`cpus_per_task=1` of the preset), and the params file sets
OMP_NUM_THREADS=1 (and OPENBLAS/MKL_NUM_THREADS=1) before NumPy is imported.

Both runs use the `pitagora_booster` preset, so they run on identical hardware; the CPU run does not
touch the GPUs. Without a batch system (or with STRUPHY_PROFILING_LOCAL=1) the runs execute directly
on this machine, where the 28-rank run is skipped unless there are that many cores.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `cube_cpu_node_vs_gpu/params_poisson.py`
directly (its `__main__` block is the worker).

Why this size: the GPU's advantage grows with the grid (the CG solve is bound by memory bandwidth, the
CPU's by its 28 cores); at 128^3 one solve takes 3.3 s on the GPU vs ~66 s on 28 login-node ranks,
and the setup (mass-matrix assembly in CUDA) 1.4 s vs ~15 s.

The runs are profiled with scope-profiler, the CuPy run with CUDA-event timing per region; the
runs are labelled "CPU (numpy): 28 MPI ranks x 1 thread" and "GPU (cupy): 1 MPI rank x 1 thread, 1 GPU".
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/, two levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "cube_cpu_node_vs_gpu"

    profiling_case = ProfilingCase(
        label="poisson_cube_cpu_node_vs_gpu",
        name="Poisson on 3D cube: 28 CPU ranks vs 1 GPU",
        description="The Poisson model with manufactured solution on a 3D cube (Cuboid, 128^3 cells, degree (1, 2, 3)), 5 solves with unpreconditioned CG, on 28 MPI ranks x 1 thread of a booster node (NumPy) vs 1 MPI rank x 1 thread on 1 H100 GPU (CuPy).",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=params_dir / "params_poisson.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # 128^3 cells, 5 solves (see the module docstring); the CPU run takes a few minutes
    param_flags = ["--cells", "128", "--steps", "5"]
    slurm_options = {"time": "00:30:00"}

    # CPU: 28 MPI ranks x 1 thread on a booster node, NumPy backend
    profiling_case.launch(
        28,
        num_nodes=1,
        param_flags=param_flags,
        backend="numpy",
        slurm_preset="pitagora_booster",
        slurm_options=slurm_options,
    )
    # GPU: 1 MPI rank x 1 thread on one H100 (the rank binds to GPU 0 of the node), CuPy backend
    profiling_case.launch(
        1,
        num_nodes=1,
        param_flags=param_flags,
        backend="cupy",
        slurm_preset="pitagora_booster",
        slurm_options=slurm_options,
    )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
