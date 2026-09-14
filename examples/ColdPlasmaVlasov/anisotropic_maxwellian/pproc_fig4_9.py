from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "thesis_run1_full"
H5_FILE = RUN / "data" / "data_proc0.hdf5"
OUTDIR = RUN / "post_processing"
OUTDIR.mkdir(parents=True, exist_ok=True)

NP = 100000
gamma = 0.0447

with h5py.File(H5_FILE, "r") as f:
    t = np.asarray(f["time/value"][:])
    B = np.asarray(f["scalar/en_B"][:])
    E = np.asarray(f["scalar/en_E"][:])
    C = np.asarray(f["scalar/en_J"][:])
    H = NP * np.asarray(f["scalar/en_f"][:])

Etot = B + E + C + H
relative_error = np.abs(Etot - Etot[0]) / abs(Etot[0])

tg = np.linspace(0.0, 120.0, 100)
growth = 5e-6 * np.exp(2.0 * gamma * tg)

plt.rcParams.update({
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
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.2))
ax1.semilogy(t, B / Etot[0], color="darkorange", lw=1.0, label=r"$\widetilde{\mathcal{E}}_B$")
ax1.semilogy(t, E / Etot[0], color="purple", lw=1.0, label=r"$\widetilde{\mathcal{E}}_E$")
ax1.semilogy(t, C / Etot[0], color="sienna", ls="--", lw=1.0, label=r"$\mathcal{E}_\mathrm{c}$")
ax1.semilogy(t, H / Etot[0], color="royalblue", lw=1.0, label=r"$\mathcal{E}_\mathrm{h}$")
ax1.semilogy(tg, growth, color="black", ls="--", lw=1.0, label="growth")
ax1.set_title("Partition of energy", fontsize=7, pad=3)
ax1.set_xlabel(r"$t|\Omega_\mathrm{ce}|$", fontsize=7, labelpad=1)
ax1.set_ylabel(r"$\mathcal{E}/\mathcal{E}(0)$", fontsize=7, labelpad=1)
ax1.set_xlim(0, 200)
ax1.set_ylim(1e-8, 1e1)
ax1.tick_params(axis="both", labelsize=7, width=0.8, length=3)
ax1.legend(
    loc="lower right",
    ncol=2,
    fontsize=6,
    frameon=True,
    framealpha=0.8,
    borderpad=0.3,
    labelspacing=0.25,
    handlelength=1.8,
    handletextpad=0.4,
    columnspacing=0.8,
)

ax2.semilogy(t, relative_error, color="purple", lw=1.0)
ax2.set_title("Relative error in total energy", fontsize=7, pad=3)
ax2.set_xlabel(r"$t|\Omega_\mathrm{ce}|$", fontsize=7, labelpad=1)
ax2.set_ylabel(
    r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$",
    fontsize=7,
    labelpad=1,
)
ax2.set_xlim(0, 200)
ax2.set_ylim(1e-10, 1e-1)
ax2.tick_params(axis="both", labelsize=7, width=0.8, length=3)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.82, wspace=0.38)

png = OUTDIR / "Figure_4_9.png"
pdf = OUTDIR / "Figure_4_9.pdf"
fig.savefig(png, dpi=400)
fig.savefig(pdf)
