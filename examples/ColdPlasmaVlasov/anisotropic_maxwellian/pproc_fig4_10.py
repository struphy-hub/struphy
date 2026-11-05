"""Thesis Figure 4.10: energy partition of params_fig4_10.py (Lie-Trotter, control variate) and its total energy
error compared with params_fig4_9.py (no control variate).

Run as ``python pproc_fig4_10.py [path_out] [--reference REF] [--show]``. The comparison curve is drawn only if the
reference run (default: the output of params_fig4_9.py) exists. The figure is written into ``<path_out>/results``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "sim_fig4_10"
DEFAULT_REFERENCE = HERE / "sim_fig4_9"

# linear growth rate of the anisotropy-driven instability (thesis)
GAMMA = 0.0447

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
}


def relative_energy_error(scalars):
    en_tot = scalars["en_tot"].to_numpy()
    return np.abs(en_tot - en_tot[0]) / abs(en_tot[0])


def main(path_out=DEFAULT_OUTPUT, reference=DEFAULT_REFERENCE, save_figs=True):
    out = Output(path_out)
    scalars = out.scalars
    t = scalars["t"].to_numpy()
    en_tot0 = scalars["en_tot"].to_numpy()[0]

    ref_scalars = None
    if reference is not None and (Path(reference) / "data").is_dir():
        ref_scalars = Output(reference).scalars

    tg = np.linspace(0.0, min(120.0, t[-1]), 200)
    growth = 1e-8 * np.exp(2.0 * GAMMA * tg)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.2))
        for key, color, ls, label in (
            ("en_B", "darkorange", "-", r"$\mathcal{E}_{\tilde B}$"),
            ("en_E", "purple", "-", r"$\mathcal{E}_{\tilde E}$"),
            ("en_J", "sienna", "--", r"$\mathcal{E}_{c}$"),
            ("en_f", "royalblue", "-", r"$\mathcal{E}_{h}$"),
        ):
            ax[0].semilogy(t, scalars[key].to_numpy() / en_tot0, color=color, ls=ls, lw=0.9, label=label)
        ax[0].semilogy(tg, growth, color="black", ls="--", lw=0.9, label="growth")
        ax[0].set_title("Partition of energy")
        ax[0].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
        ax[0].set_ylabel(r"$\mathcal{E}/\mathcal{E}(0)$")
        ax[0].set_xlim(0, t[-1])
        ax[0].set_ylim(1e-8, 1e1)
        ax[0].legend(
            loc="lower right",
            frameon=True,
            framealpha=0.8,
            ncol=2,
            fontsize=6,
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.8,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        if ref_scalars is not None:
            t_ref = ref_scalars["t"].to_numpy()
            ax[1].semilogy(t_ref[1:], relative_energy_error(ref_scalars)[1:], color="purple", lw=0.9, label="No CV")
        ax[1].semilogy(t[1:], relative_energy_error(scalars)[1:], color="darkorange", lw=0.9, label="With CV")
        ax[1].set_title("Relative error in total energy (Lie-Trotter)")
        ax[1].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
        ax[1].set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$")
        ax[1].set_xlim(0, t[-1])
        ax[1].set_ylim(1e-10, 1e-1)
        ax[1].legend(loc="upper left", frameon=False)

        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.86, wspace=0.34)

    if save_figs:
        results_dir = Path(out.path_out) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "Figure_4_10.png", dpi=400)
        fig.savefig(results_dir / "Figure_4_10.pdf")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot thesis Figure 4.10 from a run of params_fig4_10.py.")
    parser.add_argument("path_out", nargs="?", default=str(DEFAULT_OUTPUT), help="Output folder of the run.")
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="Output folder of the run without control variate (params_fig4_9.py).",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure instead of saving it.")
    args = parser.parse_args()
    main(args.path_out, reference=args.reference, save_figs=not args.show)
