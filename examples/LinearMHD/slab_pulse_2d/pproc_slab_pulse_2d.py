"""Post-process, plot and compare runs of the 2D slab pulse (LinearMHD models).

Run as ``python pproc_slab_pulse_2d.py [path_out ...] [flags of params_slab_pulse_2d.py] [--show]``. For each run, the
pressure and the out-of-plane velocity u_z are plotted at the first, middle and last saved time, with the analytic
fronts overlaid (Friedrichs diagrams of the fast and slow waves, the shear Alfven pulses at +- v_A t b0), together with
the energy budget. The reference solution is exact: in the periodic, homogeneous slab each Fourier mode of the linear
MHD system (velocity, magnetic field and pressure) evolves with the matrix exponential of its 7x7 system, applied to
the saved initial state of the run. Measured at the last saved time: the relative L2 errors of the pressure and of u_z
with respect to this solution, and the maximal relative change of the perturbed energy en_tot over the run. They are
printed, saved to ``<path_out>/results/metrics.json`` and asserted (errors below ERROR_TOL, energy conservation for
LinearMHD and LinearMHDCurlCurlwithProj up to the solver tolerances) after the figures are saved (not with
``--test-run``). With several output folders, the cuts across and along B0 and the energies of all runs are also
compared in ``<first path_out>/results/compare_runs.png``. The energy plots use plasma-plots
(``pip install "struphy[pproc]"``).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import expm

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_slab_pulse_2d import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

ENERGIES = ("en_U", "en_B", "en_thermal")  # energy scalars shown next to en_tot
GAMMA = 5 / 3
NUM_EVAL = 128  # evaluation points per direction
# relative L2 errors of p and u_z at the last saved time (default grid 64x64, dt = 0.01: at most 1.3% and 3.7%)
ERROR_TOL = {"p": 0.03, "u_z": 0.06}
# relative change of en_tot for the energy conserving models (5e-6 with solver tolerance 1e-8); en_tot is the
# conserved energy only in a uniform equilibrium with periodic boundaries, as here
ENERGY_RTOL = 1e-4
CONSERVING = ("LinearMHD", "LinearMHDCurlCurlwithProj")


def equilibrium(run):
    """Magnetic field B0, number density n0 and pressure p0 of the homogeneous equilibrium."""
    params = run.equil.params
    b0 = np.array([params["B0x"], params["B0y"], params["B0z"]])
    return b0, params["n0"], params["beta"] * (b0 @ b0) / 2


def speeds(run):
    """Alfven velocity, sound speed and unit vector b0 (in the x-y plane) of the homogeneous equilibrium."""
    b0, n0, p0 = equilibrium(run)
    abs_b = np.linalg.norm(b0)
    return abs_b / np.sqrt(n0), np.sqrt(GAMMA * p0 / n0), b0[:2] / abs_b


def friedrichs(v_a, c_s, b0, t, num=721):
    """Fronts (x, y) of a point disturbance after the time t: group velocities of the fast and slow waves times t."""
    phi = np.linspace(0, 2 * np.pi, num)
    # wave vector at the angle phi to b0, and its derivative with respect to phi
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    k_hat, dk_hat = np.einsum("ijn,j->in", rot, b0), np.einsum("ijn,j->in", rot, [-b0[1], b0[0]])
    sum2, cos2 = v_a**2 + c_s**2, np.cos(phi) ** 2
    root = np.sqrt(np.maximum(1 - 4 * v_a**2 * c_s**2 * cos2 / sum2**2, 0.0))
    fronts = {}
    for name, sign in (("fast", 1.0), ("slow", -1.0)):
        v = np.sqrt(sum2 / 2 * (1 + sign * root))
        dv = np.gradient(v, phi)
        fronts[name] = (v * k_hat + dv * dk_hat) * t
    return fronts


def model_name(run):
    return type(run.model).__name__


def evaluate(run, n):
    """Coordinates and the fields on the x-y plane at the saved step n: velocity and magnetic field (Cartesian
    components, first index) and pressure."""
    eta = (np.arange(NUM_EVAL) + 0.5) / NUM_EVAL
    params = run.domain.params
    x = params["l1"] + (params["r1"] - params["l1"]) * eta
    y = params["l2"] + (params["r2"] - params["l2"]) * eta
    kwargs = dict(t=n, eta1=eta, eta2=eta, eta3=[0.5])
    fields = {"p": run.evaluate("mhd/pressure", **kwargs).squeeze().to_numpy()}
    for key, name in (("u", "mhd/velocity"), ("b", "em_fields/b_field")):
        fields[key] = np.stack([run.evaluate(name, component=c, **kwargs).squeeze().to_numpy() for c in range(3)])
    return x, y, fields


def exact(run, x, y, initial, t):
    """Exact solution at the time t of the linear MHD system for the initial fields (on the periodic grid x, y):
    per Fourier mode k,

        n0 du/dt = -i k p + (i k x b) x B0,   db/dt = i k x (u x B0),   dp/dt = -gamma p0 i k . u.
    """
    b0, n0, p0 = equilibrium(run)
    kx = 2 * np.pi * np.fft.fftfreq(x.size, x[1] - x[0])
    ky = 2 * np.pi * np.fft.fftfreq(y.size, y[1] - y[0])
    k = np.stack(np.meshgrid(kx, ky, np.zeros(1), indexing="ij"), axis=-1)[:, :, 0]

    def cross_matrix(a):
        """Matrices of a x . for a field of vectors a (last index)."""
        out = np.zeros(a.shape + (3,), dtype=a.dtype)
        out[..., 0, 1], out[..., 0, 2], out[..., 1, 2] = -a[..., 2], a[..., 1], -a[..., 0]
        out[..., 1, 0], out[..., 2, 0], out[..., 2, 1] = a[..., 2], -a[..., 1], a[..., 0]
        return out

    ik = 1j * k
    ik_cross, b0_cross = cross_matrix(ik), cross_matrix(np.broadcast_to(b0, k.shape).astype(complex))
    M = np.zeros(k.shape[:2] + (7, 7), dtype=complex)
    M[..., 0:3, 3:6] = -b0_cross @ ik_cross / n0  # (ik x b) x B0 = -B0 x (ik x b)
    M[..., 0:3, 6] = -ik / n0
    M[..., 3:6, 0:3] = -ik_cross @ b0_cross  # ik x (u x B0) = -ik x (B0 x u)
    M[..., 6, 0:3] = -GAMMA * p0 * ik

    state = np.concatenate([initial["u"], initial["b"], initial["p"][None]])
    hat = np.fft.fft2(state, axes=(1, 2))
    hat = np.einsum("xyij,jxy->ixy", expm(M * t), hat)
    out = np.fft.ifft2(hat, axes=(1, 2)).real
    return {"u": out[0:3], "b": out[3:6], "p": out[6]}


def cut(x, y, values, direction, r):
    """Values along the line r * direction through the origin."""
    interp = RegularGridInterpolator((x, y), values, bounds_error=False, fill_value=None)
    return interp(np.stack([r * direction[0], r * direction[1]], axis=-1))


def relative_error(values, reference):
    return float(np.linalg.norm(values - reference) / np.linalg.norm(reference))


def metrics(run):
    """Relative L2 errors of p and u_z at the last saved time and maximal relative change of en_tot."""
    n = run.standard_snapshots()[-1]
    x, y, initial = evaluate(run, 0)
    _, _, fields = evaluate(run, n)
    reference = exact(run, x, y, initial, float(run.time[n]))
    en_tot = run.scalars["en_tot"].to_numpy()
    return {
        "model": model_name(run),
        "t": float(run.time[n]),
        "p_error": relative_error(fields["p"], reference["p"]),
        "u_z_error": relative_error(fields["u"][2], reference["u"][2]),
        "energy_change": float(np.max(np.abs(en_tot - en_tot[0])) / en_tot[0]),
    }


def plot_snapshots(run, results_dir, save_figs):
    """Pressure and u_z at the standard snapshots, with the analytic fronts."""
    v_a, c_s, b0 = speeds(run)
    for n in run.standard_snapshots():
        t = float(run.time[n])
        x, y, fields = evaluate(run, n)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        # the color scale of p saturates at 30% of its maximum (after t = 0), to show the weaker fast front
        scales = (0.3 if t > 0 else 1.0, 1.0)
        for ax, values, label, scale in zip(
            axes, (fields["p"], fields["u"][2]), ("pressure p", "velocity u_z"), scales
        ):
            vmax = scale * np.max(np.abs(values)) or 1.0
            im = ax.pcolormesh(x, y, values.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.plot(*(np.outer([-2, 2], b0).T), "k:", lw=0.8, label="B0 direction")
            ax.set(title=f"{label}, t = {t:.3g}", xlabel="x", ylabel="y", aspect="equal")
        if t > 0:
            for name, style in (("fast", "k--"), ("slow", "g--")):
                axes[0].plot(*friedrichs(v_a, c_s, b0, t)[name], style, lw=0.9, label=f"{name} front")
            axes[1].plot(*np.outer([v_a * t, -v_a * t], b0).T, "ko", mfc="none", label="+- v_A t b0")
        for ax in axes:
            ax.legend(loc="lower right", fontsize=7)
        finish({f"p_u_z_t{t:.6g}": fig}, results_dir, save_figs)


def plot_comparison(runs, results_dir, save_figs):
    """Cuts of p across B0 and of u_z along B0 at the last saved time with the exact solution, and the energies, of
    all runs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    r = np.linspace(-1.9, 1.9, 400)
    for run in runs:
        _, _, b0 = speeds(run)
        perp = np.array([-b0[1], b0[0]])
        n = run.standard_snapshots()[-1]
        t = float(run.time[n])
        x, y, fields = evaluate(run, n)
        axes[0].plot(r, cut(x, y, fields["p"], perp, r), label=model_name(run))
        axes[1].plot(r, cut(x, y, fields["u"][2], b0, r), label=model_name(run))
        en_tot = run.scalars["en_tot"]
        axes[2].plot(en_tot.t, en_tot / en_tot[0], label=model_name(run))
    # the exact solution of the initial state of the first run
    x, y, initial = evaluate(runs[0], 0)
    reference = exact(runs[0], x, y, initial, t)
    axes[0].plot(r, cut(x, y, reference["p"], perp, r), "k--", lw=0.9, label="exact")
    axes[1].plot(r, cut(x, y, reference["u"][2], b0, r), "k--", lw=0.9, label="exact")
    axes[0].set(title=f"pressure across B0, t = {t:.3g}", xlabel="distance from the pulse center")
    axes[1].set(title=f"u_z along B0, t = {t:.3g}", xlabel="distance from the pulse center")
    axes[2].set(title="perturbed energy en_tot / en_tot(0)", xlabel="t")
    for ax in axes:
        ax.legend(fontsize=7)
    finish({"compare_runs": fig}, results_dir, save_figs)


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


def main(paths_out=(DEFAULT_OUTPUT,), save_figs=True):
    runs, results = [], []
    for path_out in paths_out:
        run = Output(path_out)
        runs.append(run)
        results_dir = os.path.join(run.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        plot_snapshots(run, results_dir, save_figs)
        finish({"energies": run.plot.energies(parts=ENERGIES).fig}, results_dir, save_figs)

        result = metrics(run)
        with open(os.path.join(results_dir, "metrics.json"), "w") as file:
            json.dump(result, file, indent=2)
        results.append(result)

    print(f"{'model':<28}{'t':>6}{'p error':>10}{'u_z error':>11}{'d en_tot':>11}")
    for res in results:
        print(
            f"{res['model']:<28}{res['t']:>6.3g}{res['p_error']:>10.3g}{res['u_z_error']:>11.3g}"
            f"{res['energy_change']:>11.3g}"
        )

    if len(runs) > 1:
        plot_comparison(runs, os.path.join(runs[0].path_out, "results"), save_figs)

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        for res in results:
            for key, tol in ERROR_TOL.items():
                assert res[f"{key}_error"] <= tol, (
                    f"{res['model']}: relative L2 error {res[f'{key}_error']:.3g} of {key} above {tol}."
                )
            if res["model"] in CONSERVING:
                assert res["energy_change"] <= ENERGY_RTOL, (
                    f"{res['model']}: relative change {res['energy_change']:.3g} of en_tot above {ENERGY_RTOL:.0e}."
                )


if __name__ == "__main__":
    # the flags of the run are parsed by params_slab_pulse_2d.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process, plot and compare saved simulation runs.")
    pproc_parser.add_argument(
        "paths_out",
        nargs="*",
        default=[DEFAULT_OUTPUT],
        help="Simulation output folder(s) (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.paths_out, save_figs=not pproc_args.show)
