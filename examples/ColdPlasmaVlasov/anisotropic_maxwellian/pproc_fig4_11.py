from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN_LIE = ROOT / "thesis_run1_full"
RUN_STRANG = ROOT / "thesis_run1_fig4_11"
OUTDIR = RUN_STRANG / "post_processing"
OUTDIR.mkdir(parents=True, exist_ok=True)

NP = 100000


def load_run(run):
    with h5py.File(run / "data" / "data_proc0.hdf5", "r") as f:
        t = np.asarray(f["time/value"][:])
        B = np.asarray(f["scalar/en_B"][:])
        E = np.asarray(f["scalar/en_E"][:])
        C = np.asarray(f["scalar/en_J"][:])
        H = NP * np.asarray(f["scalar/en_f"][:])
    total = B + E + C + H
    return t, B, total


t_lie, B_lie, T_lie = load_run(RUN_LIE)
t_strang, B_strang, T_strang = load_run(RUN_STRANG)

if not np.allclose(t_lie, t_strang):
    raise RuntimeError("Lie-Trotter and Strang time grids do not match.")

t = t_lie

err_lie = np.abs(T_lie - T_lie[0]) / abs(T_lie[0])
err_strang = np.abs(T_strang - T_strang[0]) / abs(T_strang[0])

B_lie_n = B_lie / T_lie[0]
B_strang_n = B_strang / T_strang[0]

standard_name = "data_T=200_N=32_dt=0.0125_p=1_Np=1e5_amp=1e-4_NoCV.txt"
standard_candidates = [
    ROOT / standard_name,
    ROOT / "data" / standard_name,
    ROOT / "thesis_run1_full" / standard_name,
]

standard_file = next((p for p in standard_candidates if p.is_file()), None)
standard_available = standard_file is not None

if standard_available:
    data = np.loadtxt(standard_file)
    E_std = data[1:, -5]
    B_std = data[1:, -4]
    C_std = data[1:, -3]
    H_std = data[1:, -2]
    T_std = B_std + E_std + C_std + H_std

    Nt = len(T_std) - 1
    dt_std = 0.05 / 4
    t_std = np.linspace(0.0, Nt * dt_std, Nt + 1)

    B_std_n = B_std / T_std[0]
    err_std = np.abs(T_std - T_std[0]) / abs(T_std[0])

plt.rcParams.update({
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
})

fig, ax = plt.subplots(2, 1, figsize=(3.55, 4.35))

ax[0].semilogy(
    t,
    B_lie_n,
    color="purple",
    lw=0.9,
    label="Geometric (Lie-Trotter)",
)

ax[0].semilogy(
    t,
    B_strang_n,
    color="sienna",
    lw=0.9,
    label="Geometric (Strang)",
)

if standard_available:
    ax[0].semilogy(
        t_std,
        B_std_n,
        color="darkorange",
        lw=0.9,
        label="Standard",
    )

ax[0].set_title("Magnetic field energy")
ax[0].set_ylabel(r"$\widetilde{\mathcal{E}}_B/\mathcal{E}(0)$")
ax[0].set_xlim(0, 200)
ax[0].set_ylim(1e-8, 1e1)
ax[0].legend(loc="upper right", frameon=False)

ax[1].semilogy(
    t,
    err_lie,
    color="purple",
    lw=0.9,
    label="Geometric (Lie-Trotter)",
)

ax[1].semilogy(
    t,
    err_strang,
    color="sienna",
    lw=0.9,
    label="Geometric (Strang)",
)

if standard_available:
    ax[1].semilogy(
        t_std,
        err_std,
        color="darkorange",
        lw=0.9,
        label="Standard",
    )

ax[1].set_title("Relative error in total energy")
ax[1].set_xlabel(r"$t|\Omega_\mathrm{ce}|$")
ax[1].set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$")
ax[1].set_xlim(0, 200)
ax[1].set_ylim(1e-12, 1e-3)
ax[1].legend(loc="upper right", frameon=False)

fig.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.94, hspace=0.42)

png = OUTDIR / "Figure_4_11.png"
pdf = OUTDIR / "Figure_4_11.pdf"
fig.savefig(png, dpi=400)
fig.savefig(pdf)
