"""Post-process and plot the ITPA TAE driven by energetic ions (LinearMHDDriftkineticCC).

Run as ``python pproc_itpa_tae_B0.py`` or ``python pproc_itpa_tae_Bmax.py``, with the arguments
``[path_out] [flags of the params file] [--compare OTHER] [--fit-window T0 T1] [--show]``. The standard
plots of the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and
last saved time, together with the energy budget (MHD, magnetic and energetic-ion energies), the (m, n) harmonics
of the radial components, the energetic-ion density and delta f(r, v_par). The measured TAE frequency is compared with the analytical one and with the ideal MHD
frequency of the ITPA benchmark (Koenies et al. 2018), and the growth rate is fitted to the MHD kinetic energy; both
are saved. A frequency off by more than 10% or, with a filter, a non-positive growth rate fails with an
``AssertionError`` after the figures are saved (not with ``--test-run``). With ``--compare OTHER`` (e.g. the output
folder of the run with ``--filter none``), the MHD kinetic energy and its part outside the toroidal harmonic n = 6 of
both runs are overlaid in ``filter_comparison.png``. Figures are written into ``<path_out>/results``. The energy and
harmonics plots use plasma-plots (``pip install "struphy[pproc]"``).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line by the params file), for the default output folder.
args = None
parser = None
DEFAULT_OUTPUT = None


def configure(params):
    """Use the flags of the params file ``params`` (a module, see pproc_itpa_tae_B0.py and pproc_itpa_tae_Bmax.py)."""
    global args, parser, DEFAULT_OUTPUT
    args, parser = params.args, params.parser
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

# the TAE: toroidal mode number n (the logical mode N_LOGICAL along eta3 times tor_period of the domain) in the gap
# between the poloidal mode numbers M and M + 1, see the description of the params file
M_TAE = 10
N_LOGICAL = -1
# relative tolerance of the measured TAE frequency: the formula gives the center of the gap, whose relative width
# is about the inverse aspect ratio r_c/R0 = 0.05 at the gap
TAE_RTOL = 0.1
# ideal MHD frequency of the TAE in the ITPA benchmark: 4.01e5 rad/s (CAS3D) and 4.13e5 rad/s (GYGLES), the mean of the
# two, asserted with the same relative tolerance
OMEGA_ITPA = 4.07e5
# linear phase of the TAE for the frequency and the growth rate: after the initial transient, before the saturation
FIT_WINDOW = (100.0, 400.0)
# time of the delta f(r, v_par) plot, in the saturated phase
SATURATION = 600.0
# energetic-ion density of the benchmark, in units of the bulk density 2e19 m^-3
ITPA_N0 = 1.44131e17 / 2e19
ITPA_C = (0.49123, 0.298228, 0.198739, 0.521298)


def nearest_times(times, targets):
    """Sorted indices of the saved times closest to the target times, without duplicates."""
    return sorted({int(np.argmin(np.abs(times - target))) for target in targets})


def tae_frequency(run, basis, fit_window):
    """Measured and expected frequency of the TAE.

    Expected: omega_TAE = n / (2m + 1) * v_A(r_c) / R0 at the radius r_c of the gap, q(r_c) = (2m + 1) / (2n), for the
    q profile q0 + (q1 - q0) r^2/a^2 of the AdhocTorus; v_A = |B| / sqrt(n) (in units of the Alfven velocity) with the
    flux-surface averages of |B| and of the number density. Measured: the dominant frequency of the (M_TAE, n)
    harmonic of the radial velocity on the flux surface r_c in the time window fit_window (the growing TAE), projected
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
    coefficient = coefficient.sel(t=slice(*fit_window))
    initial = complex(coefficient.isel(t=0)) if coefficient.sizes["t"] > 0 else 0.0
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


def growth_rate(run, fit_window):
    """Growth rate of the TAE: half the slope of ln(en_U) (the MHD kinetic energy grows like exp(2 gamma t)),
    fitted by least squares in the time window fit_window."""
    en_u = run.scalars["en_U"]
    t = np.asarray(en_u["t"])
    values = np.asarray(en_u)
    fit = (t >= fit_window[0]) & (t <= fit_window[1]) & (values > 0)
    if fit.sum() < 3:
        return float("nan")
    slope = np.polyfit(t[fit], np.log(values[fit]), 1)[0]
    gamma = 0.5 * slope
    print(f"TAE growth rate gamma = {gamma:.4g} (fit of en_U for {t[fit][0]:.4g} <= t <= {t[fit][-1]:.4g})")
    return float(gamma)


def growth_figure(run, gamma, fit_window):
    """The MHD kinetic energy on a log scale with the fitted exponential growth."""
    en_u = run.scalars["en_U"]
    t = np.asarray(en_u["t"])
    values = np.asarray(en_u)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t, values, label="en_U")
    if np.isfinite(gamma):
        t_fit = t[(t >= fit_window[0]) & (t <= fit_window[1])]
        ref = values[np.searchsorted(t, t_fit[0])] * np.exp(2 * gamma * (t_fit - t_fit[0]))
        ax.semilogy(t_fit, ref, "--", label=f"exp(2 gamma t), gamma = {gamma:.3g}")
    ax.set_xlabel("t")
    ax.set_ylabel("MHD kinetic energy")
    ax.legend()
    fig.tight_layout()
    return fig


def density_profile_figure(run):
    """Energetic-ion density and -d ln n / dr at several times against the benchmark profile."""
    f = run.evaluate("energetic_ions/e1_density/f")
    t = f["t"].values
    r = minor_radius(run)(f["eta1"].values)

    n0, (c0, c1, c2, c3) = ITPA_N0 * args.nh_scale, ITPA_C
    tanh = np.tanh((r - c0) / c2)
    target = n0 * c3 * np.exp(-c2 / c1 * tanh)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(r, target, "k--", lw=1, label="ITPA")
    ax1.plot(r, (1 - tanh**2) / c1, "k--", lw=1, label="ITPA")
    for n in nearest_times(t, (0.0, 300.0, 600.0, t[-1])):
        n_r = f.isel(t=n).values
        # 3-bin smoothing against the marker noise, which the derivative amplifies
        smooth = np.convolve(n_r, np.ones(3) / 3, mode="same")
        ax0.plot(r, n_r, label=f"t = {t[n]:.4g}")
        ax1.plot(r[1:-1], -np.gradient(np.log(smooth), r)[1:-1], label=f"t = {t[n]:.4g}")
        if n == 0:
            inner = (r > 0.15) & (r < 0.95)
            gradient = (r > 0.2) & (r < 0.8)
            print(
                f"Energetic-ion density at t = {t[n]:.4g}: max |n / n_ITPA - 1| = "
                f"{np.abs(n_r / target - 1)[inner].max():.3f} for 0.15 < r < 0.95, peak -d ln n / dr = "
                f"{(-np.gradient(np.log(smooth), r))[gradient].max():.3f} (ITPA {(1 / c1):.3f})"
            )

    ax0.set_ylabel("n")
    ax0.set_title("energetic-ion density")
    ax1.set_ylabel(r"$-d\ln n / dr$")
    ax1.set_title("density gradient")
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(0.0, 4.0)
    for ax in (ax0, ax1):
        ax.set_xlabel("r")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    fig.tight_layout()
    return fig


def resonance_figure(run):
    """delta f(r, v_par) at saturation and delta f(v_par) at the gap."""
    df = run.evaluate("energetic_ions/e1_v1_density/delta_f")
    t, v = df["t"].values, df["v1"].values
    r = minor_radius(run)(df["eta1"].values)
    # v_A = |B| / sqrt(n) in units of the Alfven velocity, with the bulk density 1
    v_res = run.equil.params["B0"] / 3

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    n = nearest_times(t, (SATURATION,))[0]
    vmax = np.abs(df.isel(t=n).values).max() or 1.0
    mesh = ax0.pcolormesh(v, r, df.isel(t=n).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax0)
    ax0.set_ylabel("r")
    ax0.set_title(rf"$\delta f(r, v_\parallel)$ at t = {t[n]:.4g}")

    gap = (r >= 0.45) & (r <= 0.55)
    for n in nearest_times(t, (200.0, 400.0, 600.0, t[-1])):
        ax1.plot(v, df.isel(t=n).values[gap].sum(axis=0), label=f"t = {t[n]:.4g}")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_title(r"$\delta f(v_\parallel)$, r = 0.45 - 0.55")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    for ax in (ax0, ax1):
        ax.axvline(-v_res, color="k", ls=":", lw=0.8)
        ax.axvline(v_res, color="k", ls=":", lw=0.8)
        ax.set_xlabel(r"$v_\parallel$ (dotted: $\pm v_A/3$)")
    fig.tight_layout()
    return fig


def toroidal_energies(run, basis):
    """Squared amplitude of the radial velocity on the flux surface of the gap in the toroidal harmonic of the TAE and
    in all other toroidal harmonics (the marker noise that the Fourier filter removes), as functions of time."""
    params = run.domain.params
    eta1 = np.array([(0.5 - params["a1"]) / (params["a2"] - params["a1"])])
    eta2, eta3 = np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 8, endpoint=False)
    basis = run.field_form("mhd/velocity") if basis == "physical" else basis
    u_r = run.evaluate("mhd/velocity", representation=basis, eta1=eta1, eta2=eta2, eta3=eta3, component=0)
    power = abs(u_r.squeeze("eta1", drop=True).plasma.analysis.mode_spectrum()) ** 2
    total = power.sum(("m", "n"))
    tae = power.sel(n=[N_LOGICAL, -N_LOGICAL]).sum(("m", "n"))
    return np.asarray(power["t"]), np.asarray(tae), np.asarray(total - tae)


def comparison_figure(runs):
    """The MHD kinetic energy and the energy of the radial velocity at the gap in and outside the toroidal harmonic of
    the TAE, for each run (label -> Output), e.g. with and without the Fourier filter."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    n_tor = abs(N_LOGICAL) * next(iter(runs.values())).domain.params["tor_period"]
    for label, run in runs.items():
        en_u = run.scalars["en_U"]
        line = ax0.semilogy(np.asarray(en_u["t"]), np.asarray(en_u), label=label)[0]
        t, tae, other = toroidal_energies(run, run.field_basis("mhd/velocity"))
        ax1.semilogy(t, tae, color=line.get_color(), label=f"{label}: n = {n_tor}")
        ax1.semilogy(t, other, "--", color=line.get_color(), label=f"{label}: n != {n_tor}")
    ax0.set_xlabel("t")
    ax0.set_ylabel("MHD kinetic energy")
    ax0.legend()
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$|u_r|^2$ at r = 0.5 (sum of harmonics)")
    ax1.legend(fontsize="small")
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


def main(path_out=None, save_figs=True, compare=None, fit_window=FIT_WINDOW):
    path_out = DEFAULT_OUTPUT if path_out is None else path_out
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only
    run.pproc(create_vtk=True, physical=True, vtk_steps=snapshots)
    times = [float(run.time[n]) for n in snapshots]

    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    # the basis of the components of each vector field (None for a scalar field)
    bases = {f"{s}/{v}": run.field_basis(f"{s}/{v}") if vector else None for _, _, s, v, vector in FIELDS}

    # frequency and growth rate of the TAE, compared with the expectations at the end (after the figures)
    measured, expected = tae_frequency(run, bases["mhd/velocity"], fit_window)
    gamma = growth_rate(run, fit_window)
    # in SI units (time unit of the run in seconds)
    t_unit = float(run.units.t)
    measured_si, gamma_si = measured / t_unit, gamma / t_unit
    print(f"TAE frequency {measured_si:.4g} rad/s (ITPA benchmark, ideal MHD: {OMEGA_ITPA:.4g} rad/s)")
    print(f"TAE growth rate {gamma_si:.4g} 1/s, gamma/omega = {gamma / measured:.3g}")
    np.save(os.path.join(results_dir, "omega_tae.npy"), measured)
    np.save(os.path.join(results_dir, "omega_tae_expected.npy"), expected)
    np.save(os.path.join(results_dir, "omega_tae_si.npy"), measured_si)
    np.save(os.path.join(results_dir, "gamma_tae.npy"), gamma)
    np.save(os.path.join(results_dir, "gamma_tae_si.npy"), gamma_si)

    figs = {"energies": run.plot.energies(parts=ENERGIES).fig, "growth": growth_figure(run, gamma, fit_window)}
    # the binned distributions are not in the runs of older params files
    if "energetic_ions/e1_density/f" in run.keys():
        figs["density_profile"] = density_profile_figure(run)
    if "energetic_ions/e1_v1_density/delta_f" in run.keys():
        figs["resonance"] = resonance_figure(run)
    if compare is not None:
        other = Output(compare)
        # labelled by the folder names, or by the full paths if the names are equal
        labels = [run.path_out.name, other.path_out.name]
        if labels[0] == labels[1]:
            labels = [str(run.path_out), str(other.path_out)]
        figs["filter_comparison"] = comparison_figure(dict(zip(labels, (run, other))))
    finish(figs, results_dir, save_figs)

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
        assert abs(measured_si - OMEGA_ITPA) <= TAE_RTOL * OMEGA_ITPA, (
            f"The TAE frequency {measured_si:.4g} rad/s differs from the ITPA benchmark {OMEGA_ITPA:.4g} rad/s by "
            f"more than {TAE_RTOL:.0%}."
        )
        # without filter, the marker noise may dominate the MHD kinetic energy
        if args.filter != "none":
            assert gamma > 0.0, f"The TAE does not grow (gamma = {gamma:.4g})."


def cli():
    """Parse the command line and post-process."""
    # the flags of the run are parsed by the params file, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, beside this script)",
    )
    pproc_parser.add_argument(
        "--compare",
        default=None,
        help="Output folder of a second run to overlay in filter_comparison.png.",
    )
    pproc_parser.add_argument(
        "--fit-window",
        type=float,
        nargs=2,
        default=FIT_WINDOW,
        metavar=("T0", "T1"),
        help="Time window of the frequency and growth rate fits (default: %(default)s).",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(
        pproc_args.path_out,
        save_figs=not pproc_args.show,
        compare=pproc_args.compare,
        fit_window=tuple(pproc_args.fit_window),
    )
