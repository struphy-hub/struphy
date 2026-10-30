"""Thesis Figure 4.11: magnetic energy and total energy error of params_fig4_11.py (Strang splitting) compared with
params_fig4_9.py (Lie-Trotter splitting).

Run as ``python pproc_fig4_11.py [path_out] [--reference REF] [--show]``. The Lie-Trotter curves are drawn only if
the reference run (default: the output of params_fig4_9.py) exists. The figure is written into
``<path_out>/results``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "sim_fig4_11"
DEFAULT_REFERENCE = HERE / "sim_fig4_9"

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
}


def main(path_out=DEFAULT_OUTPUT, reference=DEFAULT_REFERENCE, save_figs=True):
    out = Output(path_out)
    runs = [("Geometric (Strang)", "sienna", out.scalars)]
    if reference is not None and (Path(reference) / "data").is_dir():
        runs.insert(0, ("Geometric (Lie-Trotter)", "purple", Output(reference).scalars))
    t_end = out.scalars["t"].to_numpy()[-1]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(2, 1, figsize=(3.55, 4.35))
        for label, color, scalars in runs:
            t = scalars["t"].to_numpy()
            en_tot = scalars["en_tot"].to_numpy()
            err = np.abs(en_tot - en_tot[0]) / abs(en_tot[0])
            ax[0].semilogy(t, scalars["en_B"].to_numpy() / en_tot[0], color=color, lw=0.9, label=label)
            ax[1].semilogy(t[1:], err[1:], color=color, lw=0.9, label=label)

        ax[0].set_title("Magnetic field energy")
        ax[0].set_ylabel(r"$\widetilde{\mathcal{E}}_B/\mathcal{E}(0)$")
        ax[0].set_xlim(0, t_end)
        ax[0].set_ylim(1e-8, 1e1)
        ax[0].legend(loc="upper right", frameon=False)

        ax[1].set_title("Relative error in total energy")
        ax[1].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
        ax[1].set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$")
        ax[1].set_xlim(0, t_end)
        ax[1].set_ylim(1e-12, 1e-3)
        ax[1].legend(loc="upper right", frameon=False)

        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.94, hspace=0.42)

    if save_figs:
        results_dir = Path(out.path_out) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "Figure_4_11.png", dpi=400)
        fig.savefig(results_dir / "Figure_4_11.pdf")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot thesis Figure 4.11 from a run of params_fig4_11.py.")
    parser.add_argument("path_out", nargs="?", default=str(DEFAULT_OUTPUT), help="Output folder of the run.")
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="Output folder of the Lie-Trotter run (params_fig4_9.py).",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure instead of saving it.")
    args = parser.parse_args()
    main(args.path_out, reference=args.reference, save_figs=not args.show)
