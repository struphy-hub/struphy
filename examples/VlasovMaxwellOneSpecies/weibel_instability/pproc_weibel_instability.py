"""Post-process and plot the Weibel instability.

Run as ``python pproc_weibel_instability.py [sim_folder] [flags of params_weibel_instability.py]``; the flags must
be those of the run. With ``--save-results`` (passed by ``submit_weibel_instability.py`` after each
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
import params_weibel_instability as params  # noqa: E402


def plot_panels(data, x, y, n_panels, ncols, title=None):
    """Snapshots of a binned distribution at evenly spaced saved times, one panel each."""
    times = np.unique(np.linspace(0, data.sizes["t"] - 1, n_panels).round().astype(int))
    grid = data.isel(t=times).plot(x=x, y=y, col="t", col_wrap=min(ncols, len(times)))
    if title is not None:
        grid.fig.suptitle(title)
    return grid


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


def main(path_out=DEFAULT_OUTPUT):
    # VTK files of the fields (fields on a 1D grid, no 3D view)
    run = Output(path_out).pproc(create_vtk=True).with_time_units("normalized")
    time = run.time
    Tend = run.time_opts.Tend
    algo = run.time_opts.split_algo

    # ------------------
    # Gauss law violation
    # ------------------
    if run.model.measure_gauss_law:
        gauss_error = run.evaluate("scalars", variables="gauss_error")["gauss_error"]

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(gauss_error.t, gauss_error)
        ax.set_xlim(0, Tend)
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel("gauss error")
        ax.set_title("Gauss law violation as function of time")
        ax.grid()
        plt.tight_layout()
        plt.show()

    # ------------------
    # progression of EM-field energy along different directions
    # ------------------
    e_field = run.evaluate("em_fields/e_field")
    b_field = run.evaluate("em_fields/b_field")
    spatial = ("eta1", "eta2", "eta3")
    # a periodic direction with a single element has size 1 (no repeated endpoint);
    # its cell then spans the whole unit length, contributing a factor of 1
    unit_volume = xp.prod([1 / max(e_field.sizes[dim] - 1, 1) for dim in spatial])

    def field_energy(field):
        """Energy of each component over space, as array of shape (component, t)."""
        return (field**2).sum(spatial).transpose("component", "t").values * unit_volume / 2

    electric_energy = field_energy(e_field)
    magnetic_energy = field_energy(b_field)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharex=True)
    ax.plot(time, electric_energy[0], label=r"|$E_1|^2$/2", color="blue")
    ax.plot(time, electric_energy[1], label=r"|$E_2|^2$/2", color="green")
    ax.plot(time, magnetic_energy[2], label=r"|$B_3|^2$/2", color="red")

    # determine magnetic field growth rate
    exp_func = lambda x, m, b: 10 ** (m * x + b)

    ti = time[-1] // 5
    tf = time[-1] if ti == 0.0 else 2 * ti
    print(f"{ti = }, {tf = }")

    xi = xp.abs(time - ti).argmin() + 1
    xf = xp.abs(time - tf).argmin() + 1

    fitting = xp.polyfit(time[xi:xf], xp.log10(magnetic_energy[2][xi:xf]), deg=1)
    ax.plot(
        time,
        exp_func(time, *fitting),
        label="fitted growth rate\n" + rf"$10^{{{fitting[0]:.5f}x {fitting[1]:.0f}}}$",
        color="cyan",
    )
    ax.plot(
        time,
        exp_func(time, 0.02784, fitting[1]),
        label="analytical growth rate\n" + rf"$10^{{0.02784x {fitting[1]:.0f}}}$",
        color="cyan",
        ls="--",
        alpha=0.5,
    )

    ax.set_title("Energy in EM field")
    ax.set_ylabel("Energy [a.u.]")
    ax.set_xlabel("time")
    ax.set_ylim(1e-14, 1e0)
    ax.set_xlim(0, Tend)
    ax.legend(ncol=3)
    ax.set_yscale("log")
    ax.minorticks_on()

    fig.suptitle(f"VlasovMaxwellOneSpecies simulation:\n {algo=}")
    plt.tight_layout()
    plt.show()

    # ------------------
    # Binning distribution evolution
    # ------------------
    for bin_name, x, y in (
        ("e1_v1_density", "eta1", "v1"),
        ("v1_v2_density", "v1", "v2"),
    ):
        for quantity in ("f", "delta_f"):
            data = run.evaluate(f"kinetic_ions/{quantity}", dataset=f"{bin_name}/{quantity}")
            plot_panels(data, x=x, y=y, n_panels=20, ncols=4)
            plt.show()

    # ------------------
    # EM field at selected times
    # ------------------
    def plot_EM_state(time_step: float, n_dim=3):
        electric_field = e_field.sel(t=time_step, method="nearest").isel(eta2=0, eta3=0)
        magnetic_field = b_field.sel(t=time_step, method="nearest").isel(eta2=0, eta3=0)

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), sharex=True, sharey=True)
        for i in range(n_dim):
            axs[0, i].plot(electric_field.eta1, electric_field.isel(component=i))
            axs[0, i].set_title(rf"$E_{i + 1}$")
            axs[1, i].plot(magnetic_field.eta1, magnetic_field.isel(component=i))
            axs[1, i].set_title(rf"$B_{i + 1}$")

        axs[0, 0].set_ylabel(r"Electric field value")
        axs[1, 0].set_ylabel(r"Magnetic field value")
        for i in range(n_dim):
            axs[1, i].set_xlabel(r"$\eta_1$")
        axs[0, 0].set_ylim(-5e-3, 5e-3)
        axs[1, 0].set_ylim(-5e-3, 5e-3)

        fig.suptitle(f"EM-field at time step: {float(electric_field.t):.2f}")
        plt.show()
        plt.close()

    for t in xp.linspace(0, time[-1], 2):
        plot_EM_state(t)

    # ------------------
    # Current density evolution
    # ------------------
    def current_1D(time_step: float):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), sharey=True, sharex=True)
        for i in range(3):
            for j in range(3):
                current = run.evaluate(
                    "kinetic_ions/f", dataset=f"e{i + 1}_current_{j + 1}/f", t=time_step, method="nearest"
                )
                ax[i, j].axhline(color="red", alpha=0.5)
                ax[i, j].plot(current[f"eta{i + 1}"], current)
            ax[i, 0].set_ylim(-0.01, 0.01)

        for i in range(3):
            ax[i, 0].set_ylabel(rf"$j_{i + 1}$")
        for j in range(3):
            ax[2, j].set_xlabel(rf"$\eta_{ {j + 1} }$")

        fig.suptitle(f"Current density at time {time_step:.2f}")
        plt.tight_layout()
        plt.show()

    for t in xp.linspace(0, time[-1], 2):
        current_1D(t)


if __name__ == "__main__":
    # the flags of the run are parsed by params_weibel_instability.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_data beside this script)",
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
        main(args.path_out)
