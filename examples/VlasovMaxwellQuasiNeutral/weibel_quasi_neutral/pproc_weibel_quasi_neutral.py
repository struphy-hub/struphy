"""Post-process the quasi-neutral Weibel example: growth rate of the magnetic energy and energy conservation.

Run as ``python pproc_weibel_quasi_neutral.py [--lam L]`` (same flags as the params file) or with the path of the
output folder. Writes ``energies.png`` into ``<sim_folder>/results`` and prints the fitted growth rate.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from params_weibel_quasi_neutral import args  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder
GAMMA_EXACT = 0.02784


def main(path_out=DEFAULT_OUTPUT):
    out = Output(path_out)
    scalars = out.scalars
    t = scalars["en_B"].t.values
    en_B = scalars["en_B"].values
    en_tot = scalars["en_tot"].values

    results_dir = Path(path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    # growth rate from the linear phase, Tend/5 < t < 2 Tend/5: en_B ~ exp(2 gamma t)
    mask = (t > t[-1] / 5) & (t < 2 * t[-1] / 5)
    if np.count_nonzero(mask) > 2:
        slope, offset = np.polyfit(t[mask], np.log(en_B[mask]), 1)
        gamma = slope / 2
        print(
            f"fitted gamma = {gamma:.5f}, analytical gamma = {GAMMA_EXACT} (rel. error {gamma / GAMMA_EXACT - 1:.2%})"
        )
    else:
        gamma = None
        print("Too few saved time steps for a growth-rate fit.")
    print(f"max relative change of en_tot: {np.max(np.abs(en_tot - en_tot[0])) / abs(en_tot[0]):.2e}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].semilogy(t, en_B, label="en_B")
    axs[0].semilogy(t, scalars["en_E"].values, label="en_E")
    if gamma is not None:
        axs[0].semilogy(t[mask], np.exp(offset + slope * t[mask]), "k--", label=f"fit, gamma = {gamma:.4f}")
        axs[0].semilogy(
            t, en_B[mask][0] * np.exp(2 * GAMMA_EXACT * (t - t[mask][0])), "r:", label=f"gamma = {GAMMA_EXACT}"
        )
    axs[0].set_ylim(en_B.min() / 10, en_B.max() * 10)
    axs[0].set_xlabel("t")
    axs[0].legend()
    axs[1].plot(t, (en_tot - en_tot[0]) / abs(en_tot[0]))
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("(en_tot - en_tot(0)) / en_tot(0)")
    fig.suptitle(f"Weibel instability, {out.path_out.name}")
    fig.tight_layout()
    fig.savefig(results_dir / "energies.png")
    plt.close(fig)


if __name__ == "__main__":
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="Output folder of the run.")
    pproc_args, _ = pproc_parser.parse_known_args()
    main(pproc_args.path_out)
