"""Run all LinearMHD ITPA TAE strong-scaling cases as one sweep (see profiling/sweep.py).

Every variant (space of the magnetic field, preconditioner, grid) of
``itpa_tae_benchmark/submit_LinearMHD_scaling.py``, 4 to 64 MPI ranks on one Raven CPU node (32 for the toroidal
Fourier preconditioner). The variants are read from the VARIANTS dict of the submit script, so new variants are picked
up.

All jobs are submitted first (SLURM runs them as the partitions allow); the runs are then packaged, and with
--upload pushed, one at a time as their jobs finish. Keep this process running until it is done (e.g. in tmux).

Usage (from anywhere, with the struphy venv activated):

    python examples/LinearMHD/sweep_LinearMHD_scaling.py [--upload] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "profiling"))

from sweep import run_sweep, variant_entries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all LinearMHD ITPA TAE strong-scaling cases as one sweep.")
    parser.add_argument("--upload", action="store_true", help="Push the packaged runs to the profiling-data repo.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands.")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between polls of the SLURM queue.")
    parser.add_argument(
        "--mem",
        default=None,
        help="SLURM memory per node of every run, passed to each submit script (default: that of each variant).",
    )
    args = parser.parse_args()

    entries = variant_entries(HERE / "itpa_tae_benchmark" / "submit_LinearMHD_scaling.py")
    if args.mem:
        entries = [(script, [*arguments, "--mem", args.mem]) for script, arguments in entries]
    n_failed = run_sweep(
        entries,
        upload=args.upload,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
    )
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
