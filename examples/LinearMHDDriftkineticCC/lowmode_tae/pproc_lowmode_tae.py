"""Post-process and plot the low-mode TAE driven by energetic ions (LinearMHDDriftkineticCC).

Run as ``python pproc_lowmode_tae.py [path_out] [flags of params_lowmode_tae.py] [--show]``. The standard plots of
the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and last saved
time, together with the energy budget (MHD, magnetic and energetic-ion energies) and the (m, n) harmonics of the
radial components. The measured TAE frequency is compared with the analytical one and the growth rate is fitted to
the MHD kinetic energy; both are saved, and a frequency off by more than 10% or a non-positive growth rate fails with
an ``AssertionError`` after the figures are saved (not with ``--test-run``). Figures are written into
``<path_out>/results``. The energy and harmonics plots use plasma-plots (``pip install "struphy[pproc]"``).
With ``--save-results`` (passed by ``submit_lowmode_tae_cpu_vs_gpu.py`` after each profiling run, ``--setup profiling``)
the scope-profiler HTML report and a summary of the cost per step are written into ``<path_out>/results`` instead; the
profiling packaging uploads them with the run.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_lowmode_tae import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# standard plots: (key, label, species, variable, vector) per field
FIELDS = [
    ("u", "U", "mhd", "velocity", True),
    ("b", "B", "em_fields", "b_field", True),
    ("p", "p", "mhd", "pressure", False),
]
# (m, n) harmonics of the radial (eta1) components
HARMONICS = [("u_r", "mhd/velocity"), ("b_r", "em_fields/b_field")]
ENERGIES = ("en_U", "en_B", "en_thermal", "en_fv", "en_fB")  # energy scalars shown next to en_tot


def energy_parts(run):
    """The energy scalars of ENERGIES saved by the run (runs before the LinearEuler family saved en_p instead of
    en_thermal)."""
    if "en_thermal" in run.scalars:
        return ENERGIES
    return tuple("en_p" if name == "en_thermal" else name for name in ENERGIES)


# the TAE: toroidal mode number n (the logical mode N_LOGICAL along eta3 times tor_period of the domain) in the gap
# between the poloidal mode numbers M and M + 1, see the description of the params file
M_TAE = 1
N_LOGICAL = -1
# relative tolerance of the measured TAE frequency: the formula gives the center of the gap, whose relative width
# is about the inverse aspect ratio r_c/R0 = 0.1 at the gap
TAE_RTOL = 0.1
# the growth rate is fitted to the second half of the run (after the initial transient)
FIT_START = 0.5


def tae_frequency(run, basis):
    """Measured and expected frequency of the TAE.

    Expected: omega_TAE = n / (2m + 1) * v_A(r_c) / R0 at the radius r_c of the gap, q(r_c) = (2m + 1) / (2n), for the
    q profile q0 + (q1 - q0) r^2/a^2 of the AdhocTorus; v_A = |B| / sqrt(n) (in units of the Alfven velocity) with the
    flux-surface averages of |B| and of the number density. Measured: the dominant frequency of the (M_TAE, n)
    harmonic of the radial velocity on the flux surface r_c over the second half of the run (the growing TAE), projected
    on its phase at the start of the fit so that a standing or a propagating mode gives the same frequency.
    """
    equil, params = run.equil, run.domain.params
    q0, q1, a, R0 = (equil.params[k] for k in ("q0", "q1", "a", "R0"))
    n = abs(N_LOGICAL) * params["tor_period"]
    q_c = (2 * M_TAE + 1) / (2 * n)
    r_c = a * np.sqrt((q_c - q0) / (q1 - q0))
    eta1 = np.array([(r_c - params["a1"]) / (params["a2"] - params["a1"])])
    eta2, eta3 = np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 8, endpoint=False)
    abs_b = np.mean(equil.absB0(eta1, eta2, eta3))
    density = np.mean(equil.n0(eta1, eta2, eta3))
    expected = n / (2 * M_TAE + 1) * abs_b / np.sqrt(density) / R0

    # the radial component is the first one of a logical basis (the FEEC form for Cartesian initial conditions)
    basis = run.field_form("mhd/velocity") if basis == "physical" else basis
    u_r = run.evaluate("mhd/velocity", representation=basis, eta1=eta1, eta2=eta2, eta3=eta3, component=0)
    modes = u_r.squeeze("eta1", drop=True).plasma.analysis.mode_spectrum()
    coefficient = modes.sel(m=M_TAE, n=N_LOGICAL)
    coefficient = coefficient.isel(t=slice(int(FIT_START * coefficient.sizes["t"]), None))
    initial = complex(coefficient.isel(t=0))
    phase = initial / abs(initial) if abs(initial) > 0 else 1.0
    signal = (coefficient * np.conj(phase)).real
    # a frequency needs a time series (not e.g. the few saved steps of a test)
    if signal.sizes["t"] < 3:
        measured = float("nan")
    else:
        measured = float(signal.plasma.analysis.spectral_peaks(n_peaks=1).omega_refined[0])
    print(
        f"TAE (m = {M_TAE}, {M_TAE + 1}, n = {n}) at r_c = {r_c:.3g}: omega = {measured:.4g}, expected {expected:.4g}"
    )
    return measured, expected


def growth_rate(run):
    """Growth rate of the TAE: half the slope of ln(en_U) (the MHD kinetic energy grows like exp(2 gamma t)),
    fitted by least squares to the second half of the run."""
    en_u = run.scalars["en_U"]
    t = np.asarray(en_u["t"])
    values = np.asarray(en_u)
    fit = (t >= t[0] + FIT_START * (t[-1] - t[0])) & (values > 0)
    if fit.sum() < 3:
        return float("nan")
    slope = np.polyfit(t[fit], np.log(values[fit]), 1)[0]
    gamma = 0.5 * slope
    print(f"TAE growth rate gamma = {gamma:.4g} (fit of en_U for t >= {t[fit][0]:.4g})")
    return float(gamma)


def growth_figure(run, gamma):
    """The MHD kinetic energy on a log scale with the fitted exponential growth."""
    en_u = run.scalars["en_U"]
    t = np.asarray(en_u["t"])
    values = np.asarray(en_u)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t, values, label="en_U")
    if np.isfinite(gamma):
        t_fit = t[t >= t[0] + FIT_START * (t[-1] - t[0])]
        ref = values[np.searchsorted(t, t_fit[0])] * np.exp(2 * gamma * (t_fit - t_fit[0]))
        ax.semilogy(t_fit, ref, "--", label=f"exp(2 gamma t), gamma = {gamma:.3g}")
    ax.set_xlabel("t")
    ax.set_ylabel("MHD kinetic energy")
    ax.legend()
    fig.tight_layout()
    return fig


def minor_radius(run):
    """The minor radius of the HollowTorus as a function of eta1."""
    a1, a2 = run.domain.params["a1"], run.domain.params["a2"]
    return lambda eta1: a1 + (a2 - a1) * eta1


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


def save_profiling_results(path_out):
    """Write the scope-profiler HTML report and a summary of the cost per step into ``<path_out>/results``."""
    from scope_profiler import create_html_report

    out = Output(path_out)
    results = out.profile.results  # the scope-profiler results of the run, all ranks merged
    results_dir = Path(out.path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    # the full scope-profiler report: call tree, charts, per-rank statistics
    create_html_report(results, results_dir / "report.html")

    # cost per step from the `model.integrate` region (the maximum over the ranks per step); the first step includes
    # one-time work (CUDA module loads, allocations of work arrays), so the median leaves it out
    integrate = results["model.integrate"]
    per_rank = [integrate[rank].durations for rank in integrate.ranks]
    n_calls = min(len(durations) for durations in per_rank)
    step_times = np.max([durations[:n_calls] for durations in per_rank], axis=0)
    step_median = float(statistics.median(step_times[1:] if len(step_times) > 1 else step_times))
    summary = {
        "backend": os.environ.get("CUNUMPY_BACKEND", "numpy"),
        "mpi_ranks": results.num_ranks,
        "num_elements": list(args.num_elements),
        "n_markers": args.Np,
        "filter": args.filter,
        "steps": len(step_times),
        "dt": args.dt,
        "step_times": step_times.tolist(),
        "step_median": step_median,
        "ns_per_marker_step": 1e9 * step_median / args.Np,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nstep: median {1e3 * step_median:.3f} ms (first step {1e3 * step_times[0]:.3f} ms), "
        f"{summary['ns_per_marker_step']:.1f} ns per marker and step ({args.Np:,} markers)"
    )
    # the propagators and kernels of the run; on the CuPy backend with GPU columns
    results.print_summary(include=["^prop: ", "^kernel: "], sort="total", title="propagators and kernels")


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only
    run.pproc(create_vtk=True, physical=True, vtk_steps=snapshots)
    times = [float(run.time[n]) for n in snapshots]

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    # the basis of the components of each vector field (None for a scalar field)
    bases = {f"{s}/{v}": run.field_basis(f"{s}/{v}") if vector else None for _, _, s, v, vector in FIELDS}

    # frequency and growth rate of the TAE, compared with the expectations at the end (after the figures)
    measured, expected = tae_frequency(run, bases["mhd/velocity"])
    gamma = growth_rate(run)
    np.save(os.path.join(results_dir, "omega_tae.npy"), measured)
    np.save(os.path.join(results_dir, "omega_tae_expected.npy"), expected)
    np.save(os.path.join(results_dir, "gamma_tae.npy"), gamma)

    finish(
        {"energies": run.plot.energies(parts=energy_parts(run)).fig, "growth": growth_figure(run, gamma)},
        results_dir,
        save_figs,
    )
    for n, t in zip(snapshots, times):
        # one field at a time, saved right away, so that few figures are open at once
        for key, label, species, variable, _ in FIELDS:
            name = f"{species}/{variable}"
            field = (key, f"{label} at t = {t:.3g}", name, n, None, bases[name])
            slices = plot_physical_slices(
                run, run.domain, run.grid.num_elements, [field], components=bases[name] is not None
            )
            # e.g. u1_slices_rz_t1.5 for the first component of U in the R-z plane at t = 1.5
            finish({f"{fig_name}_t{t:.6g}": fig for fig_name, fig in slices.items()}, results_dir, save_figs)

        figs = {}
        for key, name in HARMONICS:
            # the logical radial component: in the basis of the initial condition unless it is Cartesian
            basis = bases[name] if bases[name] != "physical" else run.field_form(name)
            data = run.evaluate(name, representation=basis, t=n, component=0)
            data.name = key
            result = data.plasma.plot.mode_profiles(
                scale=(1, run.domain.params["tor_period"]),
                x_of=minor_radius(run),
                xlabel="minor radius r",
                top=4,
            )
            figs[f"{key}_harmonics_t{t:.6g}"] = result.fig
        finish(figs, results_dir, save_figs)

        # 3D views of the scalar fields and of the first (radial eta1, in a logical basis) components, saved as
        # <field>_3d_t<time>.png and <field>1_3d_t<time>.png
        for key, _, species, variable, _ in FIELDS:
            basis = bases[f"{species}/{variable}"]
            if basis is None:
                filename = os.path.join(results_dir, f"{key}_3d_t{t:.6g}.png")
                plot_field_3d(run.path_out, variable, species=species, snapshot=n, filename=filename)
            elif basis in (run.field_form(f"{species}/{variable}"), "physical"):
                # the VTK files hold the components in the FEEC form and the Cartesian ones
                filename = os.path.join(results_dir, f"{key}1_3d_t{t:.6g}.png")
                plot_field_3d(
                    run.path_out,
                    variable,
                    species=species,
                    snapshot=n,
                    component=0,
                    physical=basis == "physical",
                    filename=filename,
                )
            else:
                print(f"No 3D view of {species}/{variable}: its basis {basis!r} is not in the VTK files.")

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        assert abs(measured - expected) <= TAE_RTOL * expected, (
            f"The TAE frequency {measured:.4g} differs from the expected {expected:.4g} by more than {TAE_RTOL:.0%}."
        )
        assert gamma > 0.0, f"The TAE does not grow (gamma = {gamma:.4g})."


if __name__ == "__main__":
    # the flags of the run are parsed by params_lowmode_tae.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write the profiling results (scope-profiler report, cost summary) instead of the plots.",
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    if pproc_args.save_results and args.setup != "profiling":
        pproc_parser.error("--save-results needs the run's --setup profiling")
    if pproc_args.save_results:
        save_profiling_results(pproc_args.path_out)
    else:
        main(pproc_args.path_out, save_figs=not pproc_args.show)
