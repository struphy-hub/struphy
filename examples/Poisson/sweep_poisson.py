"""Run all Poisson profiling cases of examples/Poisson as one sweep (see profiling/sweep.py).

- CPU strong scaling (1 to 256 MPI ranks), every variant of every case:
  gaussian_cube, gaussian_hollow_cylinder, gaussian_hollow_torus (preconditioners),
  gaussian_on_axis_cylinder, gaussian_on_axis_torus (tensor-product vs polar splines, preconditioners).
  The variants are read from the VARIANTS dict of each submit script, so new variants are picked up.
- GPU cases of gaussian_cube: 28 CPU ranks vs 1 GPU (128^3), and GPU strong scaling at 192^3 (1, 2, 4 GPUs) and
  256^3 (2, 4 GPUs; 256^3 does not fit on one GPU).

All jobs are submitted first (SLURM runs them as the partitions allow); the runs are then packaged, and with
--upload pushed, one at a time as their jobs finish. Keep this process running until it is done (e.g. in tmux).

Usage (from anywhere, with the struphy venv activated):

    python examples/Poisson/sweep_poisson.py [--upload] [--cpu-only | --gpu-only] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "profiling"))

from sweep import run_sweep, variant_entries  # noqa: E402

CPU_CASES = (
    "gaussian_cube",
    "gaussian_hollow_cylinder",
    "gaussian_hollow_torus",
    "gaussian_on_axis_cylinder",
    "gaussian_on_axis_torus",
)


def entries(cpu: bool = True, gpu: bool = True) -> list:
    """The submit commands of the sweep, as (script, arguments)."""
    out = []
    if cpu:
        for case in CPU_CASES:
            out += variant_entries(HERE / case / f"submit_{case}.py")
    if gpu:
        cube = HERE / "gaussian_cube"
        out += [
            (cube / "submit_gaussian_cube_cpu_vs_gpu.py", []),
            (cube / "submit_gaussian_cube_gpu_scaling.py", ["--cells", "192", "--gpus", "1", "2", "4"]),
            (cube / "submit_gaussian_cube_gpu_scaling.py", ["--cells", "256", "--gpus", "2", "4"]),
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Poisson profiling cases as one sweep.")
    parser.add_argument("--upload", action="store_true", help="Push the packaged runs to the profiling-data repo.")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--cpu-only", action="store_true", help="Only the CPU strong scaling cases.")
    which.add_argument("--gpu-only", action="store_true", help="Only the GPU cases of gaussian_cube.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between polls of the SLURM queue.")
    args = parser.parse_args()

    n_failed = run_sweep(
        entries(cpu=not args.gpu_only, gpu=not args.cpu_only),
        upload=args.upload,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
    )
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
