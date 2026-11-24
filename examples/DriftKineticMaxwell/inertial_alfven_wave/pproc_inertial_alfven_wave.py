"""Post-process the inertial Alfvén wave example: wave frequency from the magnetic energy and energy conservation.

Run as ``python pproc_inertial_alfven_wave.py [--lam L] [--exb-current]`` (same flags as the params file) or with
the path of the output folder. Writes ``energies.png`` into ``<sim_folder>/results`` and prints the fitted frequency.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from params_inertial_alfven_wave import EPSILON, KAPPA, LX, LZ, args  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder


def dispersion_roots(lam, exb_current):
    """Positive real roots omega of the determinant of the cold-slab dispersion matrix
    omega^2 (lam^2 E + chi E_perp) - c1 (k^2 E - k k.E) - P E_z e_z + i omega H E x e_z
    (polynomial of degree 6 in omega, interpolated exactly from 7 values)."""
    c1, c2, c3 = 1.0, KAPPA**2 * EPSILON, 1.0 / EPSILON
    chi, P, H = (EPSILON * KAPPA) ** 2, c2 * c3, c2 if exb_current else 0.0
    k = np.array([2 * np.pi / LX, 0.0, 2 * np.pi / LZ])
    cross_z = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def det(w):
        mat = (
            w**2 * (lam**2 * np.eye(3) + chi * np.diag([1.0, 1.0, 0.0]))
            - c1 * (k @ k * np.eye(3) - np.outer(k, k))
            - P * np.diag([0.0, 0.0, 1.0])
            + 1j * w * H * cross_z
        )
        return np.linalg.det(mat).real

    nodes = np.linspace(-1.0, 1.0, 7)
    roots = np.roots(np.polyfit(nodes, [det(w) for w in nodes], 6))
    return np.sort(roots[(abs(roots.imag) < 1e-8) & (roots.real > 1e-8)].real)


def main(path_out=DEFAULT_OUTPUT):
    out = Output(path_out)
    scalars = out.scalars
    t = scalars["en_B"].t.values
    en_B = scalars["en_B"].values
    en_tot = scalars["en_tot"].values

    results_dir = Path(path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    omega_exact = dispersion_roots(args.lam, args.exb_current)[0]
    # en_B ~ cos^2(omega t) = (1 + cos(2 omega t)) / 2 (plus the upper branch for lam > 0, filtered by the fit)
    omega = None
    if len(t) > 10:
        popt, _ = curve_fit(
            lambda t, a, b, c, w: a * np.cos(2 * w * t) + b * np.sin(2 * w * t) + c,
            t,
            en_B / en_B[0],
            p0=(0.5, 0.0, 0.5, omega_exact),
        )
        omega = abs(popt[3])
        print(
            f"fitted omega = {omega:.5f}, analytical omega = {omega_exact:.5f} (rel. error {omega / omega_exact - 1:.2%})"
        )
    else:
        print("Too few saved time steps for a frequency fit.")
    print(f"max relative change of en_tot: {np.max(np.abs(en_tot - en_tot[0])) / abs(en_tot[0]):.2e}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(t, en_B, label="en_B")
    axs[0].plot(t, scalars["en_E"].values, label="en_E")
    if omega is not None:
        axs[0].plot(t, en_B[0] * np.cos(omega_exact * t) ** 2, "r:", label=f"cos^2, omega = {omega_exact:.4f}")
    axs[0].set_xlabel("t")
    axs[0].legend()
    axs[1].plot(t, (en_tot - en_tot[0]) / abs(en_tot[0]))
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("(en_tot - en_tot(0)) / en_tot(0)")
    fig.suptitle(f"Inertial Alfven wave, {out.path_out.name}")
    fig.tight_layout()
    fig.savefig(results_dir / "energies.png")
    plt.close(fig)


if __name__ == "__main__":
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="Output folder of the run.")
    pproc_args, _ = pproc_parser.parse_known_args()
    main(pproc_args.path_out)
