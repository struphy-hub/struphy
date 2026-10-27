"""Vlasov in a homogeneous slab: 28 CPU MPI ranks vs 1 GPU, on the same kind of Pitagora booster node.

This file defines the profiling case (the `ProfilingCase`) and submits its two runs of the same
problem, a pure particle-pushing benchmark with the Vlasov model (16 million markers, PushVxB and
PushEta in a constant background magnetic field, 20 steps, no field solve):

- CPU: 28 MPI ranks with one thread each (OMP_NUM_THREADS=1), NumPy backend;
- GPU: 1 MPI rank with one thread on one H100, CuPy backend.

Every rank gets one CPU core (`cpus_per_task=1` of the preset), and the params file sets
OMP_NUM_THREADS=1 (and OPENBLAS/MKL_NUM_THREADS=1) before NumPy is imported.

Both runs use the `pitagora_booster` preset, so they run on identical hardware; the CPU run does not
touch the GPUs. Without a batch system (or with STRUPHY_PROFILING_LOCAL=1) the runs execute directly
on this machine, where the 28-rank run is skipped unless there are that many cores.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_homogeneous_slab.py`
directly (its `__main__` block is the worker), and then writes the scope-profiler report and the cost
summary with `pproc_homogeneous_slab.py`.

The runs are profiled with scope-profiler, the CuPy run with CUDA-event timing per region; the
runs are labelled "CPU (numpy): 28 MPI ranks x 1 thread" and "GPU (cupy): 1 MPI rank x 1 thread, 1 GPU".
The resolution (28 x 16 x 16 cells, 16M markers, 20 steps) is set in `params_homogeneous_slab.py`.
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
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label="vlasov_homogeneous_slab_cpu_node_vs_gpu",
        name="Vlasov in a homogeneous slab: 28 CPU ranks vs 1 GPU",
        description="The Vlasov model (16M markers pushed in a constant background magnetic field, PushVxB and PushEta, 20 steps, no field solve) on 28 MPI ranks x 1 thread of a booster node (NumPy) vs 1 MPI rank x 1 thread on 1 H100 GPU (CuPy).",
        physics_problem="Charged particles gyrating and streaming in a constant magnetic field (pure particle pushing).",
        struphy_model_used="Vlasov",
        params_source=script_dir / "params_homogeneous_slab.py",
        pproc_source=script_dir / "pproc_homogeneous_slab.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # CPU: 28 MPI ranks x 1 thread on a booster node, NumPy backend
    profiling_case.launch(28, num_nodes=1, backend="numpy", slurm_preset="pitagora_booster")
    # GPU: 1 MPI rank x 1 thread on one H100 (the rank binds to GPU 0 of the node), CuPy backend
    profiling_case.launch(1, num_nodes=1, backend="cupy", slurm_preset="pitagora_booster")

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
