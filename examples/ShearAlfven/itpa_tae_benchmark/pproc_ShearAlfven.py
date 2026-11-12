"""Post-process and plot the ITPA TAE benchmark run (ShearAlfven).

Run as ``python pproc_ShearAlfven.py [path_out] [flags of params_ShearAlfven.py] [--show]``. The standard
plots of the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and
last saved time, together with the energy budget and the (m, n) harmonics of the radial components. Vector fields
are shown by their components in the basis of their initial condition (``given_in_basis``, else their FEEC form,
see ``Output.field_basis``), as functions of the physical coordinates; the 3D views show the scalar fields and the
first components (radial eta1 in a logical basis). The measured TAE frequency is compared with the analytical one
and with the ideal MHD frequency of the ITPA benchmark (Koenies et al. 2018) and saved; a deviation of more than
10% fails with an ``AssertionError`` after the figures are saved (not with
``--test-run``). Figures are written into ``<path_out>/results``. The energy and harmonics plots use plasma-plots
(``pip install "struphy[pproc]"``).

Unless ``--no-frames``, the frames of three videos of the radial velocity u_r over the run (at most ``--max-frames``
saved time steps) are written into ``<path_out>/results/frames/``: ``u_r_rz`` (the poloidal plane), ``u_r_3d_cut``
(3D, the camera on the poloidal cross-section of the cutaway) and ``u_r_3d`` (3D, the whole torus with the
cutaway); the 3D views show the flux surface r = 0.5 of the gap, see ``struphy.post_processing.frames``. Make a video e.g. with
``ffmpeg -framerate 25 -i frame_%04d.png -pix_fmt yuv420p u_r_rz.mp4`` in a frames folder.

With ``--save-results`` (passed by ``submit_ShearAlfven_scaling.py`` after each profiling run, ``--setup profiling``)
the scope-profiler HTML report and a summary of the cost per step and of the solver iterations are written into
``<path_out>/results`` instead; the profiling packaging uploads them with the run.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.frames import write_3d_frames, write_videos
from struphy.post_processing.profiling import save_profiling_results
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_ShearAlfven import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# standard plots: (key, label, species, variable, vector) per field
FIELDS = [
    ("u", "U", "mhd", "velocity", True),
    ("b", "B", "em_fields", "b_field", True),
]
# (m, n) harmonics of the radial (eta1) components
HARMONICS = [("u_r", "mhd/velocity"), ("b_r", "em_fields/b_field")]
ENERGIES = ("en_U", "en_B")  # energy scalars shown next to en_tot

# the TAE: toroidal mode number n (the logical mode N_LOGICAL along eta3 times tor_period of the domain) in the gap
# between the poloidal mode numbers M and M + 1, see the description of the params file
M_TAE = 10
N_LOGICAL = -1
# minor radius of the gap (q = 1.75), the flux surface shown in the 3D views of the videos
R_GAP = 0.5
# relative tolerance of the measured TAE frequency: the formula gives the center of the gap, whose relative width
# is about the inverse aspect ratio r_c/R0 = 0.05 at the gap
TAE_RTOL = 0.1
# ideal MHD frequency of the TAE in the ITPA benchmark (Koenies et al. 2018): 4.01e5 rad/s (CAS3D) and 4.13e5 rad/s
# (GYGLES), the mean of the two, asserted with the same relative tolerance
OMEGA_ITPA = 4.07e5


def radial_range(domain):
    """The minor radius (r_min, r_max) at eta1 = 0 and 1: the hole and the wall of the HollowTorus, or (0, a) for the
    IGAPolarTorus of --polar (no hole)."""
    params = domain.params
    if "a1" in params:
        return params["a1"], params["a2"]
    return 0.0, params["a"]


def tae_frequency(run, basis):
    """Measured and expected frequency of the TAE.

    Expected: omega_TAE = n / (2m + 1) * v_A(r_c) / R0 at the radius r_c of the gap, q(r_c) = (2m + 1) / (2n), for the
    q profile q0 + (q1 - q0) r^2/a^2 of the AdhocTorus; v_A = |B| / sqrt(n) (in units of the Alfven velocity) with the
    flux-surface averages of |B| and of the number density. Measured: the dominant frequency of the (M_TAE, n)
    harmonic of the radial velocity on the flux surface r_c over the run, projected on its initial phase so that a
    standing or a propagating mode gives the same frequency.
    """
    equil, params = run.equil, run.domain.params
    q0, q1, a, R0 = (equil.params[k] for k in ("q0", "q1", "a", "R0"))
    n = abs(N_LOGICAL) * params["tor_period"]
    q_c = (2 * M_TAE + 1) / (2 * n)
    r_c = a * np.sqrt((q_c - q0) / (q1 - q0))
    r_min, r_max = radial_range(run.domain)
    eta1 = np.array([(r_c - r_min) / (r_max - r_min)])
    eta2, eta3 = np.linspace(0, 1, 64, endpoint=False), np.linspace(0, 1, 8, endpoint=False)
    abs_b = np.mean(equil.absB0(eta1, eta2, eta3))
    density = np.mean(equil.n0(eta1, eta2, eta3))
    expected = n / (2 * M_TAE + 1) * abs_b / np.sqrt(density) / R0

    # the radial component is the first one of a logical basis (the FEEC form for Cartesian initial conditions)
    basis = run.field_form("mhd/velocity") if basis == "physical" else basis
    u_r = run.evaluate("mhd/velocity", representation=basis, eta1=eta1, eta2=eta2, eta3=eta3, component=0)
    modes = u_r.squeeze("eta1", drop=True).plasma.analysis.mode_spectrum()
    coefficient = modes.sel(m=M_TAE, n=N_LOGICAL)
    initial = complex(coefficient.isel(t=0))
    # projected on the initial phase; without initial amplitude in this harmonic its real part
    phase = initial / abs(initial) if abs(initial) > 0 else 1.0
    signal = (coefficient * np.conj(phase)).real
    # a frequency needs a time series (not e.g. the single saved step of a one-step test)
    if signal.sizes["t"] < 3:
        measured = float("nan")
    else:
        measured = float(signal.plasma.analysis.spectral_peaks(n_peaks=1).omega_refined[0])
    print(
        f"TAE (m = {M_TAE}, {M_TAE + 1}, n = {n}) at r_c = {r_c:.3g}: omega = {measured:.4g}, expected {expected:.4g}"
    )
    return measured, expected


def minor_radius(run):
    """The minor radius as a function of eta1 (HollowTorus or, with --polar, IGAPolarTorus)."""
    a1, a2 = radial_range(run.domain)
    return lambda eta1: a1 + (a2 - a1) * eta1


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


def save_results(path_out):
    """Write the scope-profiler report and the summary of the cost per step and of the solver iterations into
    ``<path_out>/results`` (:func:`struphy.post_processing.profiling.save_profiling_results`)."""
    info = {
        "num_elements": list(args.num_elements),
        "degree": list(args.degree),
        "b_space": args.b_space,
        "projected": args.projected,
        "polar": args.polar,
        "a1": None if args.polar else args.a1,
        "precond": args.precond,
        "dt": args.dt,
    }
    summary = save_profiling_results(path_out, info=info)
    step_times = summary["step_times"]
    iterations = ", ".join(
        f"{mean:.1f} iterations per step ({name})" for name, mean in summary["solver_iterations_mean"].items()
    )
    print(
        f"\nstep: median {1e3 * summary['step_median']:.3f} ms (first step {1e3 * step_times[0]:.3f} ms); {iterations}"
    )
    Output(path_out).profile.results.print_summary(include=["^prop: "], sort="total", title="propagators")


def main(path_out=DEFAULT_OUTPUT, save_figs=True, frames=True, max_frames=500):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files of the standard snapshots only (e.g. for ParaView)
    run.pproc(create_vtk=True, physical=True, vtk_steps=snapshots)
    times = [float(run.time[n]) for n in snapshots]
    # eta1 of the flux surface r = 0.5 of the gap, shown in the 3D views
    r_min, r_max = radial_range(run.domain)
    eta1_gap = (R_GAP - r_min) / (r_max - r_min)

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    # the basis of the components of each vector field (None for a scalar field)
    bases = {f"{s}/{v}": run.field_basis(f"{s}/{v}") if vector else None for _, _, s, v, vector in FIELDS}

    # the frequency of the TAE, compared with the analytical value at the end (after the figures)
    measured, expected = tae_frequency(run, bases["mhd/velocity"])
    # in rad/s (time unit of the run in seconds)
    measured_si = measured / float(run.units.t)
    print(f"TAE frequency {measured_si:.4g} rad/s, ITPA benchmark (ideal MHD) {OMEGA_ITPA:.4g} rad/s")
    np.save(os.path.join(results_dir, "omega_tae.npy"), measured)
    np.save(os.path.join(results_dir, "omega_tae_expected.npy"), expected)
    np.save(os.path.join(results_dir, "omega_tae_si.npy"), measured_si)

    finish({"energies": run.plot.energies(parts=ENERGIES).fig}, results_dir, save_figs)
    for n, t in zip(snapshots, times):
        # one field at a time, saved right away, so that few figures are open at once
        for key, label, species, variable, _ in FIELDS:
            name = f"{species}/{variable}"
            field = (key, f"{label} at t = {t:.3g}", name, n, None, bases[name])
            slices = plot_physical_slices(run, run.domain, run.grid.num_elements, [field], components=True)
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

    # 3D views (cutaway, semi-transparent boundary, the flux surface r = 0.5 of the gap) of the scalar fields and of
    # the first (radial eta1, in a logical basis) components at the standard snapshots, as the 3D frames of the videos
    # (one color scale over the snapshots), saved as <field>_3d_t<time>.png and <field>1_3d_t<time>.png
    for key, _, species, variable, _ in FIELDS:
        name, basis = f"{species}/{variable}", bases[f"{species}/{variable}"]
        folder = os.path.join(results_dir, f"{key}_3d_snapshots")
        views = write_3d_frames(
            run,
            name,
            folder,
            component=None if basis is None else 0,
            basis=basis,
            steps=snapshots,
            view="cut",
            surface=eta1_gap,
        )
        # the frames are written on rank 0 only (an empty list on the other ranks, which must not touch the folder)
        if views:
            for view, t in zip(views, times):
                shutil.move(view, os.path.join(results_dir, f"{key}{'' if basis is None else '1'}_3d_t{t:.6g}.png"))
            shutil.rmtree(folder, ignore_errors=True)

    if frames:
        # frames of videos of the radial velocity u_r (the first component in the normalized basis, radial for the
        # circular flux surfaces): the poloidal plane and two 3D views, at most max_frames saved
        # time steps, in <path_out>/results/frames/u_r_rz, u_r_3d_cut and u_r_3d (frame_*.png); the 3D views show the
        # flux surface r = 0.5 of the gap (not the one through the largest value, which can be a continuum oscillation)
        videos = write_videos(
            run,
            "mhd/velocity",
            os.path.join(results_dir, "frames"),
            component=0,
            basis="norm",
            prefix="u_r",
            label="radial velocity u_r",
            max_frames=max_frames,
            options_3d={"surface": eta1_gap},
        )
        for video, files in videos.items():
            if files:  # (empty on the ranks other than 0)
                print(f"{len(files)} frames of {video} in {os.path.join(results_dir, 'frames', video)}")

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


if __name__ == "__main__":
    # the flags of the run are parsed by params_ShearAlfven.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_parser.add_argument("--no-frames", action="store_true", help="Do not write the frames of the videos.")
    pproc_parser.add_argument(
        "--max-frames", type=int, default=500, help="Largest number of frames per video (default: 500)."
    )
    pproc_parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write the profiling results (scope-profiler report, cost summary) instead of the plots.",
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    if pproc_args.save_results and args.setup != "profiling":
        pproc_parser.error("--save-results needs the run's --setup profiling")
    if pproc_args.save_results:
        save_results(pproc_args.path_out)
    else:
        main(
            pproc_args.path_out,
            save_figs=not pproc_args.show,
            frames=not pproc_args.no_frames,
            max_frames=pproc_args.max_frames,
        )
