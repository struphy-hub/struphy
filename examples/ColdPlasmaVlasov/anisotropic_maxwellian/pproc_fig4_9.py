"""Thesis Figure 4.9: energy partition and total energy error of params_fig4_9.py (Lie-Trotter, no control variate).

Run as ``python pproc_fig4_9.py [path_out] [--show]``. The figure is written into ``<path_out>/results``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_fig4_9"

# linear growth rate of the anisotropy-driven instability (thesis)
GAMMA = 0.0447

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.7,
}


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    scalars = out.scalars
    t = scalars["t"].to_numpy()
    en_B = scalars["en_B"].to_numpy()
    en_E = scalars["en_E"].to_numpy()
    en_J = scalars["en_J"].to_numpy()
    en_f = scalars["en_f"].to_numpy()
    en_tot = scalars["en_tot"].to_numpy()

    relative_error = np.abs(en_tot - en_tot[0]) / abs(en_tot[0])

    tg = np.linspace(0.0, min(120.0, t[-1]), 100)
    growth = 5e-6 * np.exp(2.0 * GAMMA * tg)

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.2))
        ax1.semilogy(t, en_B / en_tot[0], color="darkorange", lw=1.0, label=r"$\widetilde{\mathcal{E}}_B$")
        ax1.semilogy(t, en_E / en_tot[0], color="purple", lw=1.0, label=r"$\widetilde{\mathcal{E}}_E$")
        ax1.semilogy(t, en_J / en_tot[0], color="sienna", ls="--", lw=1.0, label=r"$\mathcal{E}_\mathrm{c}$")
        ax1.semilogy(t, en_f / en_tot[0], color="royalblue", lw=1.0, label=r"$\mathcal{E}_\mathrm{h}$")
        ax1.semilogy(tg, growth, color="black", ls="--", lw=1.0, label="growth")
        ax1.set_title("Partition of energy", pad=3)
        ax1.set_xlabel(r"$t|\Omega_\mathrm{ce}|$", fontsize=7, labelpad=1)
        ax1.set_ylabel(r"$\mathcal{E}/\mathcal{E}(0)$", fontsize=7, labelpad=1)
        ax1.set_xlim(0, t[-1])
        ax1.set_ylim(1e-8, 1e1)
        ax1.tick_params(axis="both", labelsize=7, width=0.8, length=3)
        ax1.legend(
            loc="lower right",
            ncol=2,
            frameon=True,
            framealpha=0.8,
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.8,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        ax2.semilogy(t[1:], relative_error[1:], color="purple", lw=1.0)
        ax2.set_title("Relative error in total energy", pad=3)
        ax2.set_xlabel(r"$t|\Omega_\mathrm{ce}|$", fontsize=7, labelpad=1)
        ax2.set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$", fontsize=7, labelpad=1)
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(1e-10, 1e-1)
        ax2.tick_params(axis="both", labelsize=7, width=0.8, length=3)

        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.82, wspace=0.38)

    if save_figs:
        results_dir = Path(out.path_out) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "Figure_4_9.png", dpi=400)
        fig.savefig(results_dir / "Figure_4_9.pdf")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot thesis Figure 4.9 from a run of params_fig4_9.py.")
    parser.add_argument("path_out", nargs="?", default=str(DEFAULT_OUTPUT), help="Output folder of the run.")
    parser.add_argument("--show", action="store_true", help="Show the figure instead of saving it.")
    args = parser.parse_args()
    main(args.path_out, save_figs=not args.show)
