"""Post-process and plot the ITPA TAE benchmark run (LinearMHD).

Run as ``python pproc_TAE_benchmark.py [path_out] [flags of params_TAE_benchmark.py] [--show]``. The standard
plots of the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and
last saved time, together with the energy budget and the (m, n) harmonics of the radial components. Vector fields
are shown by their components in the basis of their initial condition (``given_in_basis``, else their FEEC form,
see ``Output.field_basis``), as functions of the physical coordinates; the 3D views show the scalar fields and the
first components (radial eta1 in a logical basis). The measured TAE frequency is compared with the analytical one, saved and a
warning issued for a deviation of more than 10% (not with ``--test-run``). Figures are written into ``<path_out>/results``. The energy and harmonics plots use plasma-plots
(``pip install "struphy[pproc]"``).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_TAE_benchmark import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# standard plots: (key, label, species, variable, vector) per field
FIELDS = [
    ("u", "U", "mhd", "velocity", True),
    ("b", "B", "em_fields", "b_field", True),
    ("n", "n", "mhd", "density", False),
    ("p", "p", "mhd", "pressure", False),
]
# (m, n) harmonics of the radial (eta1) components
HARMONICS = [("u_r", "mhd/velocity"), ("b_r", "em_fields/b_field")]
ENERGIES = ("en_U", "en_B", "en_thermal")  # energy scalars shown next to en_tot

# the TAE: toroidal mode number n (the logical mode N_LOGICAL along eta3 times tor_period of the domain) in the gap
# between the poloidal mode numbers M and M + 1, see the description of the params file
M_TAE = 10
N_LOGICAL = -1
# relative tolerance of the measured TAE frequency: the formula gives the center of the gap, whose relative width
# is about the inverse aspect ratio r_c/R0 = 0.05 at the gap
TAE_RTOL = 0.1


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
    """The minor radius of the HollowTorus as a function of eta1."""
    a1, a2 = run.domain.params["a1"], run.domain.params["a2"]
    return lambda eta1: a1 + (a2 - a1) * eta1


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


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

    # the frequency of the TAE, compared with the analytical value (not for the few steps of --test-run); only
    # reported until a full-resolution run has confirmed the measurement (then: assert)
    measured, expected = tae_frequency(run, bases["mhd/velocity"])
    np.save(os.path.join(results_dir, "omega_tae.npy"), measured)
    np.save(os.path.join(results_dir, "omega_tae_expected.npy"), expected)
    if not args.test_run and abs(measured - expected) > TAE_RTOL * expected:
        warnings.warn(
            f"The TAE frequency {measured:.4g} differs from the expected {expected:.4g} by more than {TAE_RTOL:.0%}."
        )

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


if __name__ == "__main__":
    # the flags of the run are parsed by params_TAE_benchmark.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
