from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN_NOCV = ROOT / "thesis_run1_full"
RUN_CV = ROOT / "thesis_run1_fig4_10"
OUTDIR = RUN_CV / "post_processing"
OUTDIR.mkdir(parents=True, exist_ok=True)

NP = 100000
gamma = 0.0447


def load_run(run):
    with h5py.File(run / "data" / "data_proc0.hdf5", "r") as f:
        t = np.asarray(f["time/value"][:])
        B = np.asarray(f["scalar/en_B"][:])
        E = np.asarray(f["scalar/en_E"][:])
        C = np.asarray(f["scalar/en_J"][:])
        H = NP * np.asarray(f["scalar/en_f"][:])
    total = B + E + C + H
    return t, B, E, C, H, total


t0, B0, E0, C0, H0, T0 = load_run(RUN_NOCV)
t1, B1, E1, C1, H1, T1 = load_run(RUN_CV)

if not np.allclose(t0, t1):
    raise RuntimeError("No-CV and CV time grids do not match.")

t = t0

B = B1 / T1[0]
E = E1 / T1[0]
C = C1 / T1[0]
H = H1 / T1[0]

err_nocv = np.abs(T0 - T0[0]) / abs(T0[0])
err_cv = np.abs(T1 - T1[0]) / abs(T1[0])

tg = np.linspace(0.0, 120.0, 200)
growth = 1e-8 * np.exp(2.0 * gamma * tg)

plt.rcParams.update({
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
})

fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.2))

ax[0].semilogy(t, B, color="darkorange", lw=0.9, label=r"$\mathcal{E}_{\tilde B}$")
ax[0].semilogy(t, E, color="purple", lw=0.9, label=r"$\mathcal{E}_{\tilde E}$")
ax[0].semilogy(t, C, color="sienna", ls="--", lw=0.9, label=r"$\mathcal{E}_{c}$")
ax[0].semilogy(t, H, color="royalblue", lw=0.9, label=r"$\mathcal{E}_{h}$")
ax[0].semilogy(tg, growth, color="black", ls="--", lw=0.9, label="growth")
ax[0].set_title("Partition of energy")
ax[0].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
ax[0].set_ylabel(r"$\mathcal{E}/\mathcal{E}(0)$")
ax[0].set_xlim(0, 200)
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

ax[1].semilogy(t, err_nocv, color="purple", lw=0.9, label="No CV")
ax[1].semilogy(t, err_cv, color="darkorange", lw=0.9, label="With CV")
ax[1].set_title("Relative error in total energy (Lie-Trotter)")
ax[1].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
ax[1].set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$")
ax[1].set_xlim(0, 200)
ax[1].set_ylim(1e-10, 1e-1)
ax[1].legend(loc="upper left", frameon=False)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.86, wspace=0.34)

png = OUTDIR / "Figure_4_10.png"
pdf = OUTDIR / "Figure_4_10.pdf"
fig.savefig(png, dpi=400)
fig.savefig(pdf)
