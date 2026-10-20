"""Post-process and plot the slab drift wave.

Run as ``python pproc_slab_drift_wave.py [sim_folder] [flags of params_slab_drift_wave.py] [--show]``; the flags
must be those of the run. The figures are written into ``<sim_folder>/results`` (``--show`` shows them instead):
the amplitude and phase of the initial (k_y, k_z) mode against time with the measured and predicted frequencies,
the field energy, and phi in the (x, y) plane at the first, middle and last saved time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.special import ive, wofz

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_slab_drift_wave as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def plasma_Z(zeta):
    """Plasma dispersion function."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def drift_wave_roots(n_harmonics=8):
    """Roots of the local kinetic dispersion relations (quasi-neutral, eta_i = 0) of the full-orbit model and of
    the drift-kinetic model, for the wave vector of the initial perturbation; see kinetic_drift_waves.md."""
    eps, Te, Ti, kap = params.epsilon, params.Te, params.Ti, params.kappa_n
    kperp2 = params.kx**2 + params.ky**2
    b = eps**2 * kperp2 * Ti
    w_si = -eps * params.ky * Ti * kap
    s = np.sqrt(2) * abs(params.kz) * np.sqrt(Ti)

    def D_vlasov(w):
        tot = 1.0 + 0j
        for n in range(-n_harmonics, n_harmonics + 1):
            tot += ive(n, b) * (w - w_si) / s * plasma_Z((w - n / eps) / s)
        return 1 / Te + tot / Ti

    def D_dk(w):
        return b / Ti + 1 / Te + (1 + (w - w_si) / s * plasma_Z(w / s)) / Ti

    def root(D):
        sol = fsolve(lambda x: [D(x[0] + 1j * x[1]).real, D(x[0] + 1j * x[1]).imag], [-w_si * Te / Ti, 0.0])
        return sol[0] + 1j * sol[1]

    return root(D_vlasov), root(D_dk)


def mode_coefficient(run, n_points=32):
    """Fourier coefficient c(t) of phi on the plane x = Lx/2 for the initial mode cos(k_y y + k_z z), which is one
    period in eta2 and eta3; for a wave cos(k_y y + k_z z - omega t), c(t) = A/2 exp(-i omega t)."""
    e = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phi = run.evaluate("em_fields/phi", eta1=0.5, eta2=e, eta3=e)
    data = np.asarray(phi.transpose("t", "eta2", "eta3").values)
    phase = np.exp(-2j * np.pi * (e[:, None] + e[None, :]))
    return phi.t.values, np.mean(data * phase[None], axis=(1, 2))


def measured_frequency(t, coeff):
    """Angular frequency from a linear fit of the unwrapped phase of c(t) = A/2 exp(-i omega t)."""
    if len(t) < 3:
        return np.nan
    return -np.polyfit(t, np.unwrap(np.angle(coeff)), 1)[0]


def saver(run, save_figs):
    """A function saving figures as ``<name>.png`` into ``<path_out>/results`` and closing them (``save_figs``)."""
    results_dir = Path(run.path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    def save(figs):
        if save_figs:
            for name, fig in figs.items():
                fig.savefig(results_dir / f"{name}.png")
                plt.close(fig)

    return save


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    save = saver(run, save_figs)

    # the initial (k_y, k_z) mode on the plane x = Lx/2
    t, coeff = mode_coefficient(run)
    omega_num = measured_frequency(t, coeff)
    omega_v, omega_dk = drift_wave_roots()
    print(f"measured omega = {omega_num:.4f}, full orbit: {omega_v:.4f}, drift-kinetic: {omega_dk:.4f}")

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(t, np.abs(coeff))
    axs[0].set(ylabel="mode amplitude", yscale="log")
    axs[1].plot(t, np.unwrap(np.angle(coeff)), label="measured")
    axs[1].plot(t, np.angle(coeff[0]) - omega_v.real * t, "--", label="full orbit")
    axs[1].plot(t, np.angle(coeff[0]) - omega_dk.real * t, ":", label="drift-kinetic")
    axs[1].set(xlabel="time", ylabel="mode phase")
    axs[1].legend()
    fig.suptitle(
        rf"$\omega$ measured {omega_num:.3f}, full orbit {omega_v.real:.3f}, drift-kinetic {omega_dk.real:.3f}"
    )
    save({"mode": fig})

    energy = run.evaluate("scalars", variables="en_phi")["en_phi"]
    fig, ax = plt.subplots()
    ax.plot(energy.t, energy)
    ax.set(xlabel="time", yscale="log", title="Field energy en_phi")
    save({"en_phi": fig})

    # phi in the (x, y) plane at the first, middle and last saved time
    plane = run.evaluate("em_fields/phi", t=list(snapshots), eta3=0.0)
    grid = plane.plot(x="eta2", y="eta1", col="t")
    save({"phi_xy": grid.fig})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_slab_drift_wave.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, beside this script)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    args = parser.parse_args(other_flags)
    main(args.path_out, save_figs=not args.show)
