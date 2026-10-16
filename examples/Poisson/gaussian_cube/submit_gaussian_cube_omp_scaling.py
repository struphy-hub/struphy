"""Poisson on a 3D cube: OpenMP strong scaling with one MPI rank on a Pitagora DCGP node.

This file defines the profiling case (the `ProfilingCase`) and submits its runs: the problem of the MPI
scaling in `submit_gaussian_cube.py` (manufactured Gaussian solution on a Cuboid, 128 x 128 x 128 cells,
degree (1, 2, 3), unpreconditioned CG, one solve) with one MPI rank and 1, 2, 4, ..., 256 OpenMP threads
(`--threads`), on the NumPy backend. The kernels are compiled with OpenMP (the case's `openmp`), which
recompiles them if they were compiled without.

The threads are placed by OpenMP (OMP_PLACES=cores) on a whole node, with one placement per scan
(`--proc-bind`, default spread):
- close: the threads on neighbouring cores, filling one CCD (chiplet, 16 cores) after the other;
- spread: the threads as far apart as possible, over both sockets and all CCDs, so that each thread has as
  much memory bandwidth as possible (cf. the MPI pinning of `submit_gaussian_cube.py`).

Each generated script runs the simulation itself by invoking `params_gaussian_cube.py` directly (its
`__main__` block is the worker), and then post-processes it with `pproc_gaussian_cube.py`. The runs are
labelled e.g. "CPU: 1 MPI rank, 16 threads, spread". `finalize_run` packages (and uploads) each run as soon
as its own job finishes, and writes the report of the scan into the case folder, e.g.
`report_poisson_cube_omp_scaling_spread.html`.
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
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="OpenMP thread counts of the single MPI rank, at most the 256 cores of a node.",
    )
    parser.add_argument(
        "--proc-bind",
        choices=["close", "spread"],
        default="spread",
        help="OpenMP thread placement (OMP_PROC_BIND) of the scan.",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label="poisson_cube_omp_scaling",
        name="Poisson on 3D cube OpenMP strong scaling test, no preconditioner",
        description="OpenMP strong scaling of the Poisson model with manufactured solution on a 3D cube (Cuboid, 128^3 cells), solved with unpreconditioned CG, with one MPI rank.",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=script_dir / "params_gaussian_cube.py",
        pproc_source=script_dir / "pproc_gaussian_cube.py",
        language=args.language,
        compiler=args.compiler,
        openmp=True,
        upload=args.upload,
    )

    # Launch one run per thread count, all with one MPI rank on one node
    for num_threads in sorted(set(args.threads)):
        profiling_case.launch(1, num_nodes=1, num_threads=num_threads, omp_proc_bind=args.proc_bind)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
