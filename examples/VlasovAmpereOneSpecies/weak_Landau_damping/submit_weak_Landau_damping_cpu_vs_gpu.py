"""Vlasov-Ampere weak Landau damping profiling case: one CPU node vs one GPU node.

This file defines the profiling case (the `ProfilingCase`) and submits its two runs of the same
problem: one full CPU node (256 MPI ranks, NumPy backend) and one full GPU node (4 MPI ranks, one per
GPU, CuPy backend). The two runs go to different SLURM partitions, so each `ProfilingCase.launch`
names its own preset from `clusters.SLURM_PRESETS`. Without a batch system the runs execute directly
on this machine, where the 256-rank run is skipped unless there are that many cores.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_weak_Landau_damping.py`
directly (its `__main__` block is the worker) with `--setup profiling`, and then writes the
scope-profiler report and cost summary with `pproc_weak_Landau_damping.py --save-results`.

The runs are profiled with scope-profiler, the CuPy run with CUDA-event timing per region. The
resolution (1024 cells, 1000 markers per cell, 20 steps) is the profiling setup of
`params_weak_Landau_damping.py`, shared with the strong-scaling case of `submit_weak_Landau_damping.py`.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

# flags of params_weak_Landau_damping.py (and pproc_weak_Landau_damping.py) for the profiling setup
PARAM_FLAGS = ["--setup", "profiling", "--save-results"]

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
        label="vlasov_ampere_landau_damping_cpu_node_vs_gpu_node",
        name="Vlasov-Ampere weak Landau damping: CPU node vs GPU node",
        description="The VlasovAmpereOneSpecies model on weak Landau damping (periodic box, k = 0.5, 1e-3 density perturbation, 1024 cells, about 1M markers) on one full CPU node (256 MPI ranks, NumPy) vs one full GPU node (4 MPI ranks on 4 GPUs, CuPy).",
        physics_problem="Landau damping of an electrostatic wave in a collisionless electron plasma.",
        struphy_model_used="VlasovAmpereOneSpecies",
        params_source=script_dir / "params_weak_Landau_damping.py",
        pproc_source=script_dir / "pproc_weak_Landau_damping.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # One CPU node: 256 ranks on Pitagora's DCGP partition
    profiling_case.launch(256, num_nodes=1, param_flags=PARAM_FLAGS, backend="numpy", slurm_preset="pitagora_dcgp")
    # One GPU node: one rank per GPU on Pitagora's booster partition
    profiling_case.launch(4, num_nodes=1, param_flags=PARAM_FLAGS, backend="cupy", slurm_preset="pitagora_booster")

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
