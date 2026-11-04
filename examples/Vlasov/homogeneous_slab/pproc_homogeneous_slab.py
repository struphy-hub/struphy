"""Post-process the Vlasov homogeneous_slab benchmark.

Run as ``python pproc_homogeneous_slab.py [sim_folder] [flags of params_homogeneous_slab.py]``; the flags must be
those of the run. Writes the scope-profiler HTML report and a summary of the cost per step and marker into
``<sim_folder>/results``, where the profiling packaging picks them up and uploads them with the run.
``submit_homogeneous_slab_cpu_vs_gpu.py`` and ``submit_homogeneous_slab_gpu_scaling.py`` call this script after each
simulation.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cunumpy as xp
import numpy as np

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_homogeneous_slab as params  # noqa: E402


def main(path_out=DEFAULT_OUTPUT):
    from scope_profiler import create_html_report

    if not (Path(path_out) / "profiling_data.h5").exists():
        print(f"No profiling data in {path_out} (run without profiling): nothing to write.")
        return

    out = Output(path_out)
    results = out.profile.results  # the scope-profiler results of the run, all ranks merged
    results_dir = out.path_out / "results"
    results_dir.mkdir(exist_ok=True)

    # the full scope-profiler report: call tree, charts, per-rank statistics
    create_html_report(results, results_dir / "report.html")

    # cost per step and per marker, from the `model.integrate` region (the maximum over the ranks per step);
    # the first step includes one-time work (CUDA module loads, allocations of work arrays).
    # On the CuPy backend the CPU timestamps only measure kernel launches (no device sync at region ends), so
    # the CUDA-event device times are used whenever they were recorded.
    integrate = results["model.integrate"]
    per_rank = [
        integrate[rank].gpu_durations if integrate[rank].has_gpu_timing else integrate[rank].durations
        for rank in integrate.ranks
    ]
    n_calls = min(len(durations) for durations in per_rank)
    step_times = np.max([durations[:n_calls] for durations in per_rank], axis=0)
    step_median = float(statistics.median(step_times[1:] if len(step_times) > 1 else step_times))
    summary = {
        "backend": xp.get_backend(),
        "gpu_timing": params.gpu_timing,
        "step_times_from": "gpu" if integrate[integrate.ranks[0]].has_gpu_timing else "cpu",
        "mpi_ranks": results.num_ranks,
        "num_elements": list(params.sim.grid.num_elements),
        "n_markers": params.n_markers,
        "steps": params.steps,
        "dt": params.dt,
        "step_times": step_times.tolist(),
        "step_median": step_median,
        "ns_per_marker_step": 1e9 * step_median / params.n_markers,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nstep: median {1e3 * step_median:.3f} ms (first step {1e3 * step_times[0]:.3f} ms), "
        f"{summary['ns_per_marker_step']:.1f} ns per marker and step ({params.n_markers:,} markers)"
    )
    # the propagators and kernels of the run; on the CuPy backend with GPU columns
    results.print_summary(
        include=["^prop: ", "^kernel: ", "^pusher: ", "mpi_sort"],
        sort="total",
        title="propagators and kernels",
    )


if __name__ == "__main__":
    # the flags of the run are parsed by params_homogeneous_slab.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Write the profiling results of a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    args = parser.parse_args(other_flags)
    main(args.path_out)
