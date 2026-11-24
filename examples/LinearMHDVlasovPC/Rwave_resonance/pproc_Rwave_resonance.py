"""Plot the R-wave resonance (LinearMHDVlasovPC).

Run as ``python pproc_Rwave_resonance.py [path_out] [flags of params_Rwave_resonance.py] [--show]``.

Two figures are saved in ``<path_out>/diagnostics``:
- ``magnetic_energy.png``: magnetic energy with an exponential fit in the linear phase,
- ``f_vpar.png``: f(v_par) at t = 0 and at saturation, and f(v_par, t).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from maybempi import MPI

from struphy import Output

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params_Rwave_resonance import args, k, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

TITLE = "pressure coupling"

# linear growth phase in units of 1/Omega_c (used for the fit)
FIT_WINDOW = (10.0, 90.0)

# resonance omega - k v_par = -Omega_c with omega = k v_A
V_RES = (k + 1.0) / k


def plot_magnetic_energy(run):
    en_B = run.scalars["en_B"]
    t = en_B["t"].values
    e = en_B.values

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(t, e, label=r"$|\widetilde B|^2$")

    # |B|^2 ~ exp(2 gamma t)
    mask = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1]) & (e > 0)
    if mask.sum() >= 2:
        slope, offset = np.polyfit(t[mask], np.log(e[mask]), 1)
        gamma = slope / 2
        t_unit = en_B["t_seconds"].values[-1] / t[-1]  # seconds per 1/Omega_c
        print(f"gamma = {gamma:.4g} Omega_c = {gamma / t_unit:.3e} 1/s")

        ax.semilogy(t, np.exp(offset + slope * t), "--", label=rf"$\gamma = {gamma / t_unit:.2e}/s$")
        ax.axvspan(*FIT_WINDOW, color="0.9", label="Linear growth")

    ax.set_ylim(1e-10, 1e-1)
    ax.set_xlabel(r"time [$1/\Omega_c$]")
    ax.set_ylabel("magnetic energy")
    ax.set_title(TITLE)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_distribution(run):
    f = run.evaluate("energetic_ions/v3_density/f")
    v = f["v3"].values
    f = f / np.trapezoid(f.isel(t=0), v)  # normalize to 1 at t = 0

    # saturation = maximum of the magnetic energy
    en_B = run.scalars["en_B"]
    t_sat = en_B["t"].values[np.argmax(en_B.values)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    ax0.plot(v, f.isel(t=0), label="$t = 0$")
    ax0.plot(v, f.sel(t=t_sat, method="nearest"), "--", label=r"$t = t_\mathrm{sat}$")
    ax0.axvline(V_RES, color="k", ls="--", label=rf"$v_\parallel = (\omega + \Omega_c)/k = {V_RES:.2f}\,V_A$")
    ax0.set_ylim(0.0, 1.3 * float(f.max()))
    ax0.set_xlabel(r"$v_\parallel$ [$V_A$]")
    ax0.set_ylabel(r"$f_\mathrm{normalized}$")
    ax0.set_title(TITLE)
    ax0.legend(loc="upper right", fontsize="small")

    mesh = ax1.pcolormesh(v, f["t"].values, f.values, cmap="inferno", shading="auto")
    ax1.axvline(V_RES, color="w", ls="--")
    fig.colorbar(mesh, ax=ax1)
    ax1.set_xlabel(r"$v_\parallel$ [$V_A$]")
    ax1.set_ylabel(r"time [$1/\Omega_c$]")
    ax1.set_title(r"$f(v_\parallel,\, t)$")

    fig.tight_layout()
    return fig


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)

    # evaluate on all ranks, but only rank 0 saves the figures
    figs = {
        "magnetic_energy": plot_magnetic_energy(run),
        "f_vpar": plot_distribution(run),
    }
    if MPI.COMM_WORLD.Get_rank() != 0:
        plt.close("all")
        return

    if not save_figs:
        plt.show()
        return

    diagnostics = os.path.join(run.path_out, "diagnostics")
    os.makedirs(diagnostics, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(os.path.join(diagnostics, f"{name}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    # flags of the params file are handled there, the rest here
    _, other_flags = parser.parse_known_args()

    pproc_parser = argparse.ArgumentParser(description="Plot the R-wave resonance.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="simulation output folder")
    pproc_parser.add_argument("--show", action="store_true", help="show the figures instead of saving them")
    pproc_args = pproc_parser.parse_args(other_flags)

    main(pproc_args.path_out, save_figs=not pproc_args.show)
