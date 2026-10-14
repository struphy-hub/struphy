"""Post-process and plot weak Landau damping.

Run as ``python pproc_weak_Landau_damping.py [sim_folder] [flags of params_weak_Landau_damping.py]``; the flags
must be those of the run. With ``--save-results`` (passed by ``submit_weak_Landau_damping.py`` after each
profiling run) the scope-profiler HTML report and a summary of the cost per step and marker are written into
``<sim_folder>/results`` instead of showing the plots; the profiling packaging uploads them with the run.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cunumpy as xp
import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_data"

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_weak_Landau_damping as params  # noqa: E402


def plot_panels(data, x, y, n_panels, ncols, title=None):
    """Snapshots of a binned distribution at evenly spaced saved times, one panel each."""
    times = np.unique(np.linspace(0, data.sizes["t"] - 1, n_panels).round().astype(int))
    grid = data.isel(t=times).plot(x=x, y=y, col="t", col_wrap=min(ncols, len(times)))
    if title is not None:
        grid.fig.suptitle(title)
    return grid


def E_exact(t, eps=0.001):
    """Analytical electric energy of weak Landau damping, t in normalized units."""
    r = 0.3677
    omega_r = 1.4156
    omega_i = -0.1533
    phi = 0.5362
    return (4 * eps * r * xp.exp(omega_i * t) * xp.cos(omega_r * t - phi)) ** 2 * xp.pi


def save_profiling_results(path_out):
    """Write the scope-profiler HTML report and a summary of the cost per step and marker into ``results``."""
    from scope_profiler import create_html_report

    out = Output(path_out)
    results = out.profile.results  # the scope-profiler results of the run, all ranks merged
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
    n_markers = params.ppc * params.num_elements
    summary = {
        "backend": xp.get_backend(),
        "gpu_timing": params.gpu_timing,
        "mpi_ranks": results.num_ranks,
        "num_elements": params.num_elements,
        "ppc": params.ppc,
        "n_markers": n_markers,
        "steps": params.steps,
        "dt": params.dt,
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


def main(path_out=DEFAULT_OUTPUT, amplitude=0.001):
    # VTK files of the fields (fields on a 1D grid, no 3D view)
    run = Output(path_out).pproc(create_vtk=True)

    # electric field energy against the analytical damping
    energy = run.evaluate("scalars", variables="electric_energy")["electric_energy"]
    analytical = E_exact(energy.t.values, eps=amplitude)  # t is in Struphy units
    fig, ax = plt.subplots()
    ax.plot(energy.t, energy, label="numerical")
    ax.plot(energy.t, analytical, "--", label="analytical")
    ax.set(xlabel="time", yscale="log", title="Electric energy")
    ax.legend()
    plt.show()

    # full f and delta f in the e1-v1 plane at four times
    for quantity, title in (("f", "full-$f$"), ("delta_f", r"$\delta f$")):
        data = run.evaluate(f"kinetic_ions/{quantity}", dataset=f"e1_v1_density/{quantity}")
        plot_panels(data, x="eta1", y="v1", n_panels=4, ncols=4, title=title)
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_weak_Landau_damping.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_data beside this script)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.001,
        help="Initial perturbation amplitude for the analytical curve",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write the profiling results (scope-profiler report, cost summary) instead of showing the plots.",
    )
    args = parser.parse_args(other_flags)
    if args.save_results and not params.profiling:
        parser.error("--save-results needs the run's --setup profiling")
    if args.save_results:
        save_profiling_results(args.path_out)
    else:
        main(args.path_out, amplitude=args.amplitude)
