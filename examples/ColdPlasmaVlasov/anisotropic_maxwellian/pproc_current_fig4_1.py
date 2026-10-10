import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Current geometric/FEM STRUPHY counterpart of thesis Fig. 4.1.
# Reads the completed current run; no simulation is performed.

RUN = "thesis_run1_fig4_9"
H5 = os.path.join(RUN, "data", "data_proc0.hdf5")
OUT = os.path.join(RUN, "figures")
os.makedirs(OUT, exist_ok=True)

with h5py.File(H5, "r") as f:
    t = np.asarray(f["time/value"][:], dtype=float)
    en_B = np.asarray(f["scalar/en_B"][:], dtype=float)
    en_E = np.asarray(f["scalar/en_E"][:], dtype=float)
    en_J = np.asarray(f["scalar/en_J"][:], dtype=float)
    en_f = np.asarray(f["scalar/en_f"][:], dtype=float)
    en_tot = np.asarray(f["scalar/en_tot"][:], dtype=float)

# en_f is already the complete particle-energy contribution.
decomp = en_B + en_E + en_J + en_f
err = np.max(np.abs(en_tot - decomp))
if err > 1e-12 * max(1.0, np.max(np.abs(en_tot))):
    raise RuntimeError(f"Energy decomposition failed: max abs error = {err:.6e}")

E0 = en_tot[0]
B = en_B / E0
E = en_E / E0
C = en_J / E0
H = en_f / E0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(3.4, 2.2))

ax.semilogy(t, B, color="darkorange", lw=0.9,
            label=r"$\widetilde{\mathcal {E}}_B$")
ax.semilogy(t, E, color="purple", lw=0.9,
            label=r"$\widetilde{\mathcal {E}}_E$")
ax.semilogy(t, C, color="sienna", lw=0.9, ls="--",
            label=r"$\mathcal {E}_c$")
ax.semilogy(t, H, color="royalblue", lw=0.9,
            label=r"$\mathcal {E}_h$")

# Thesis-style analytical growth guide.
gamma = 0.0447
growth = 1e-6 * np.exp(2.0 * gamma * t)
ax.semilogy(t, growth, color="black", lw=0.7, ls="--",
            label=r"$\propto e^{2\gamma t}$")

ax.set_title("Partition of energy", pad=3)
ax.set_xlabel(r"$t|\Omega_{\mathrm{ce}}|$")
ax.set_ylabel(r"$\mathrm{E}/\mathrm{E}(0)$")
ax.set_xlim(0, 200)
ax.set_ylim(1e-8, 1e1)

ax.legend(loc="lower right", ncol=2, frameon=True,
          handlelength=1.7, columnspacing=0.8,
          borderpad=0.4, handletextpad=0.4)

fig.subplots_adjust(left= .20, right=.985, bottom=.22, top=.82)
fig.savefig(os.path.join(OUT, "Figure_4_1_current.png"), dpi=400)
fig.savefig(os.path.join(OUT, "Figure_4_1_current.pdf"))
plt.close(fig)

print("CURRENT FIG. 4.1 GENERATED")
print(f"Run: {RUN}")
print(f"Points: {len(t)}")
print(f"Energy decomposition max abs error: {err:.6e}")
print(f"E(0): {E0:.12e}")
print(f"Output: {os.path.join(OUT, 'Figure_4_1_current.png')}")
