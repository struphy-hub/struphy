"""Post-process the Vlasov-Maxwell weak Landau damping profiling case: the scope-profiler report and cost summary.

Run as ``python pproc_weak_Landau_damping.py [sim_folder]``, on the same cunumpy backend as the run (the environment
variable CUNUMPY_BACKEND, which the profiling driver exports in the job script). The resolution of the run is taken
from ``params_weak_Landau_damping.py``, the timings from the run's ``profiling_data.h5``. The scope-profiler HTML report
and ``summary.json`` are written into ``<sim_folder>/results``, where the profiling packaging picks them up and
uploads them next to the run's profiling data (``submit_weak_Landau_damping.py`` calls this script after
each simulation).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cunumpy as xp
import numpy as np
from scope_profiler import create_html_report

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# The resolution of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_weak_Landau_damping import dt, gpu_timing, num_elements, parser, ppc, steps  # noqa: E402


def main(path_out=DEFAULT_OUTPUT):
    out = Output(path_out)

    try:
        results = out.profile.results  # the scope-profiler results of the run, all ranks merged
    except FileNotFoundError:
        # e.g. in test_examples.py, which runs the simulation without profiling
        print(f"No profiling data in {out.path_out}; run with Simulation.run(profiling_activated=True).")
        return
    results_dir = out.path_out / "results"
    results_dir.mkdir(exist_ok=True)

    # the full scope-profiler report: call tree, charts, per-rank statistics
    create_html_report(results, results_dir / "report.html")

    # cost per step and per marker, from the `model.integrate` region (the maximum over the ranks per step);
    # the first step includes one-time work (CUDA module loads, allocations of work arrays)
    integrate = results["model.integrate"]
    per_rank = [integrate[rank].durations for rank in integrate.ranks]
    n_calls = min(len(durations) for durations in per_rank)
    step_times = np.max([durations[:n_calls] for durations in per_rank], axis=0)
    step_median = float(statistics.median(step_times[1:] if len(step_times) > 1 else step_times))
    n_markers = ppc * num_elements
    summary = {
        "backend": xp.get_backend(),
        "gpu_timing": gpu_timing,
        "mpi_ranks": results.num_ranks,
        "num_elements": num_elements,
        "ppc": ppc,
        "n_markers": n_markers,
        "steps": steps,
        "dt": dt,
        "step_times": step_times.tolist(),
        "step_median": step_median,
        "ns_per_marker_step": 1e9 * step_median / n_markers,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nstep: median {1e3 * step_median:.3f} ms (first step {1e3 * step_times[0]:.3f} ms), "
        f"{summary['ns_per_marker_step']:.1f} ns per marker and step ({n_markers:,} markers)"
    )
    # the propagators and kernels of the run; on the CuPy backend with GPU columns
    results.print_summary(include=["^prop: ", "^kernel: "], sort="total", title="propagators and kernels")


if __name__ == "__main__":
    # the flags of the run are parsed by params_weak_Landau_damping.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out)
