"""Submit the anisotropic auto-domain-decomposition benchmark.

Each profiling run benchmarks all valid ``mpi_dims_mask`` choices for the
512-by-16 ToyDrift diocotron case and records the fastest choice relative to
the default ``(True, True, True)`` decomposition.
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload packaged profiling results to the profiling-data repository.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    benchmark = (
        script_dir
        / "examples"
        / "ToyGyrokinetic"
        / "diocotron_instability"
        / "benchmark_domain_decomposition.py"
    )

    profiling_case = ProfilingCase(
        label="domain_decomposition",
        name="Automatic domain-decomposition benchmark",
        description=(
            "Anisotropic 512x16 ToyDrift diocotron case comparing automatic "
            "MPI decomposition against (True, True, True)."
        ),
        physics_problem="Diocotron instability in a non-neutral plasma.",
        struphy_model_used="ToyDrift",
        params_source=benchmark,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    for num_tasks in (8, 16, 32):
        profiling_case.launch(num_tasks)

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
