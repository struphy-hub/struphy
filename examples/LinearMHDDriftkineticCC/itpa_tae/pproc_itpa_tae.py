"""Plot the ITPA TAE runs (LinearMHDDriftkineticCC).

Run as ``python pproc_itpa_tae.py [path_out] [--show]``; the default ``path_out`` is ``sim_fourier/`` beside this
file (the output of params_itpa_tae_fourier.py), use ``sim_hybrid`` for params_itpa_tae_hybrid.py.

Five figures are saved in ``<path_out>/diagnostics``:
- ``energy.png``: |B~|^2 and |U~|^2 with an exponential fit of |B~|^2 in the linear phase,
- ``harmonics.png``: radial profiles of the m = 9..12 (n = 6) harmonics of U~_r and B~_r at the end of the linear phase,
- ``poloidal.png``: U~_r, U~_theta, B~_r and B~_theta in the poloidal plane at the same time,
- ``resonance.png``: delta f(r, v_par) at saturation and delta f(v_par) at the gap, with v_par = +-v_A/3,
- ``profile.png``: energetic-ion density n(r) and -d ln n/dr at several times, against the ITPA profile.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from maybempi import MPI

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_fourier"

# linear growth phase (used for the fit; its end is the time of the harmonics and poloidal plots)
FIT_WINDOW = (100.0, 400.0)
# time of the delta f(r, v_par) plot, in the saturated phase
SATURATION = 600.0

# poloidal harmonics of the n = 6 TAE, which is the logical mode -1 along eta3 (tor_period = 6)
MS = (9, 10, 11, 12)
N_DIRECTION = -1


def nearest(times, t):
    """Index of the saved time closest to t."""
    return int(np.argmin(np.abs(times - t)))


def minor_radius(run, eta1):
    """Minor radius r of the HollowTorus at eta1."""
    a1, a2 = run.domain.params["a1"], run.domain.params["a2"]
    return a1 + (a2 - a1) * eta1


def plot_energy(run):
    en_U, en_B = run.scalars["en_U"], run.scalars["en_B"]
    t = en_U["t"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t, en_B.values, label=r"$|\widetilde B|^2$")
    ax.semilogy(t, en_U.values, label=r"$|\widetilde U|^2$")

    # |B~|^2 ~ exp(gamma t): gamma is the growth rate of the energy
    mask = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1]) & (en_B.values > 0)
    if mask.sum() >= 2:
        gamma, offset = np.polyfit(t[mask], np.log(en_B.values[mask]), 1)
        t_unit = en_B["t_seconds"].values[-1] / t[-1]  # seconds per time unit
        print(f"{run.path_out}: gamma = {gamma:.4g} = {gamma / t_unit:.3e} 1/s")
        ax.semilogy(t, np.exp(offset + gamma * t), "--", label=rf"$\gamma = {gamma / t_unit:.2e}/s$")
        ax.axvspan(*FIT_WINDOW, color="0.9")

    ax.set_ylim(1e-8, 1e-4)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlabel(r"$t\ [T_A]$")
    ax.set_ylabel("energy")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_harmonics(run, n):
    n2, n3 = run.grid.num_elements[1], run.grid.num_elements[2]
    eta1 = np.linspace(0.0, 1.0, 101)
    eta2 = np.linspace(0.0, 1.0, 2 * n2, endpoint=False)
    eta3 = np.linspace(0.0, 1.0, 2 * n3, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fields = (("mhd/velocity", r"$\widetilde U_r$", 10), ("em_fields/b_field", r"$\widetilde B_r$", 11))
    for ax, (name, label, dominant) in zip(axes, fields):
        # eta1 component of the 2-form (the radial flux), Fourier transformed in (eta2, eta3)
        f_r = run.evaluate(name, eta1=eta1, eta2=eta2, eta3=eta3, component=0, t=n).values
        c = np.abs(np.fft.fft2(f_r, axes=(1, 2)))
        norm = c[:, dominant, N_DIRECTION].max() or 1.0
        for m in MS:
            ax.plot(minor_radius(run, eta1), c[:, m, N_DIRECTION] / norm, label=f"m = {m}")

        ax.set_title(label)
        ax.set_xlabel("r")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("normalized FFT amplitude")
    fig.suptitle(f"t = {run.time[n]:g}")
    fig.tight_layout()
    return fig


def plot_poloidal(run, n):
    eta1 = np.linspace(0.0, 1.0, 64)
    eta2 = np.linspace(0.0, 1.0, 129)
    eta3 = np.array([0.0])
    r = minor_radius(run, eta1)[:, None, None]
    da = run.domain.params["a2"] - run.domain.params["a1"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (name, label) in enumerate((("mhd/velocity", "U"), ("em_fields/b_field", "B"))):
        # radial and poloidal components from the contravariant ones (r = a1 + (a2 - a1) eta1, theta = 2 pi eta2)
        v = run.evaluate(name, representation="v", eta1=eta1, eta2=eta2, eta3=eta3, t=n)
        X, Z = v.coords["X"].values[..., 0], v.coords["Z"].values[..., 0]
        components = ((da * v.values[0], "r"), (2 * np.pi * r * v.values[1], r"\theta"))
        for ax, (field, comp) in zip(axes[2 * i : 2 * i + 2], components):
            vmax = np.abs(field).max() or 1.0
            mesh = ax.pcolormesh(X, Z, field[..., 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            fig.colorbar(mesh, ax=ax)
            ax.set_title(rf"$\widetilde {label}_{comp}$")
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle(f"t = {run.time[n]:g}")
    fig.tight_layout()
    return fig


def plot_resonance(run):
    df = run.evaluate("energetic_ions/e1_v1_density/delta_f")
    t, v = df["t"].values, df["v1"].values
    r = minor_radius(run, df["eta1"].values)
    # velocity unit: v_A at B = 1 T and bulk density 1, so v_A = B0 here
    v_res = run.equil.params["B0"] / 3

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    n = nearest(t, SATURATION)
    vmax = np.abs(df.isel(t=n).values).max() or 1.0
    mesh = ax0.pcolormesh(v, r, df.isel(t=n).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax0)
    ax0.set_ylabel("r")
    ax0.set_title(rf"$\delta f(r, v_\parallel)$ at t = {t[n]:g}")

    gap = (r >= 0.45) & (r <= 0.55)
    for n in sorted({nearest(t, tt) for tt in (200, 400, 600, 1000)}):
        ax1.plot(v, df.isel(t=n).values[gap].sum(axis=0), label=f"t = {t[n]:g}")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_title(r"$\delta f(v_\parallel)$, r = 0.45 - 0.55")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    for ax in (ax0, ax1):
        ax.axvline(-v_res, color="k", ls=":", lw=0.8)
        ax.axvline(v_res, color="k", ls=":", lw=0.8)
        ax.set_xlabel(r"$v_\parallel$ (dotted: $\pm v_A/3$)")
    fig.tight_layout()
    return fig


def plot_profile(run):
    f = run.evaluate("energetic_ions/e1_density/f")
    t = f["t"].values
    r = minor_radius(run, f["eta1"].values)

    # ITPA profile (Koenies et al. 2018)
    n0, c = 0.00720652, (0.49123, 0.298228, 0.198739, 0.521298)
    r_itpa = np.linspace(r[0], r[-1], 200)
    tanh = np.tanh((r_itpa - c[0]) / c[2])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(r_itpa, n0 * c[3] * np.exp(-c[2] / c[1] * tanh), "k--", lw=1, label="ITPA")
    ax1.plot(r_itpa, (1 - tanh**2) / c[1], "k--", lw=1, label="ITPA")
    for n in sorted({nearest(t, tt) for tt in (0, 300, 600, 1000)}):
        n_r = f.isel(t=n).values
        # 3-bin smoothing against the marker noise, which the derivative amplifies
        smooth = np.convolve(n_r, np.ones(3) / 3, mode="same")
        ax0.plot(r, n_r, label=f"t = {t[n]:g}")
        ax1.plot(r[1:-1], -np.gradient(np.log(smooth), r)[1:-1], label=f"t = {t[n]:g}")

    ax0.set_ylabel("n")
    ax0.set_title("energetic-ion density")
    ax1.set_ylabel(r"$-d\ln n / dr$")
    ax1.set_title("density gradient")
    ax1.set_xlim(0.2, 0.8)
    ax1.set_ylim(0.0, 4.0)
    for ax in (ax0, ax1):
        ax.set_xlabel("r")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    fig.tight_layout()
    return fig


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    n = nearest(run.time, FIT_WINDOW[1])

    # evaluate on all ranks, but only rank 0 saves the figures
    figs = {
        "energy": plot_energy(run),
        "harmonics": plot_harmonics(run, n),
        "poloidal": plot_poloidal(run, n),
        "resonance": plot_resonance(run),
        "profile": plot_profile(run),
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
    pproc_parser = argparse.ArgumentParser(description="Plot an ITPA TAE run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="simulation output folder (default: sim_fourier beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="show the figures instead of saving them")
    pproc_args = pproc_parser.parse_args()

    main(pproc_args.path_out, save_figs=not pproc_args.show)
