"""Submit the hybrid Vlasov sorting-frequency benchmark.

The benchmark runs the small ``LinearMHDVlasovCC`` case and measures the
candidate sorting frequencies hardcoded in
``examples/Vlasov/clone_decomposition/benchmark_vlasov_clones.py``.  The
benchmark also records the best clone/decomposition configuration first, then
reuses it while tuning sorting frequency.
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
        / "Vlasov"
        / "clone_decomposition"
        / "benchmark_vlasov_clones.py"
    )

    profiling_case = ProfilingCase(
        label="vlasov_sorting_frequency",
        name="Hybrid Vlasov sorting-frequency benchmark",
        description=(
            "Small LinearMHDVlasovCC case comparing hardcoded particle "
            "sorting frequencies after selecting the best parallel configuration."
        ),
        physics_problem="Hybrid MHD-fluid and energetic-ion Vlasov evolution.",
        struphy_model_used="LinearMHDVlasovCC",
        params_source=benchmark,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    for num_tasks in (4, 8, 16):
        profiling_case.launch(num_tasks)

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
