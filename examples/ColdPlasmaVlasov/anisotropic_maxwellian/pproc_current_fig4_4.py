import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Current geometric/FEM STRUPHY counterpart of thesis Fig. 4.4.
# Uses the completed current run; no simulation is performed.

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

# Independent consistency check.
decomp = en_B + en_E + en_J + en_f
decomp_err = np.max(np.abs(en_tot - decomp))
if decomp_err > 1e-12 * max(1.0, np.max(np.abs(en_tot))):
    raise RuntimeError(f"Energy decomposition failed: {decomp_err:.6e}")

E0 = en_tot[0]
rel_err = np.abs(en_tot - E0) / abs(E0)

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
ax.semilogy(t, rel_err, color="purple", lw=0.9)

ax.set_title("Relative error in total energy", pad=3)
ax.set_xlabel(r"$t|\Omega_{\mathrm{ce}}|$")
ax.set_ylabel(r"$|\mathcal{E}(t)-\mathcal{E}(0)|/\mathcal{E}(0)$")
ax.set_xlim(0, 200)
ax.set_ylim(1e-10, 1e1)

fig.subplots_adjust(left=.20, right=.985, bottom=.22, top=.82)
fig.savefig(os.path.join(OUT, "Figure_4_4_current.png"), dpi=400)
fig.savefig(os.path.join(OUT, "Figure_4_4_current.pdf"))
plt.close(fig)

imax = np.argmax(rel_err)
print("CURRENT FIG. 4.4 GENERATED")
print(f"Run: {RUN}")
print(f"Energy decomposition max abs error: {decomp_err:.6e}")
print(f"Initial total energy: {E0:.12e}")
print(f"Maximum relative total-energy error: {rel_err[imax]:.6e}")
print(f"Time of maximum error: {t[imax]:.6f}")
print(f"Final relative total-energy error: {rel_err[-1]:.6e}")
print(f"Output: {os.path.join(OUT, 'Figure_4_4_current.png')}")
