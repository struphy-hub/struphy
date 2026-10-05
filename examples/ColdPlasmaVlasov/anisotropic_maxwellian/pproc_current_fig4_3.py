#!/usr/bin/env python3
"""
Current-method Figure 4.3 post-processing.

Reads the already completed geometric-FEEC Run-1 HDF5 output.
NO simulation is run.

Important:
- e3_density is the electric-field-component diagnostic.
- v3_density is the parallel velocity distribution.
- v1_v3_density is the 2-D perpendicular-component/parallel
  velocity distribution.

The script therefore uses the velocity diagnostics directly from
data_proc0.hdf5 rather than assuming that post-processing .npy files
have already been materialized.
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "thesis_run1_full"
H5 = RUN / "data" / "data_proc0.hdf5"
OUT = RUN / "post_processing" / "current_fig4_3"
OUT.mkdir(parents=True, exist_ok=True)

OMEGA_R = 0.4742
K = 2.0
V_RESONANCE = abs((OMEGA_R - 1.0) / K)

if not H5.is_file():
    raise FileNotFoundError(f"Missing completed simulation output:\n{H5}")

def centers(ds, axis):
    key = f"bin_centers_{axis}"
    if key not in ds.attrs:
        raise KeyError(f"Missing HDF5 attribute {key!r} on {ds.name}")
    return np.asarray(ds.attrs[key], dtype=float)

print("=" * 72)
print("CURRENT FIG. 4.3 — VELOCITY-DISTRIBUTION POST-PROCESSING")
print("=" * 72)
print("Run:", RUN)
print("Input:", H5)
print("No simulation will be run.")

with h5py.File(H5, "r") as f:
    t = np.asarray(f["time/value"][:], dtype=float)

    ds_v3 = f["kinetic/hot_elec/f/v3_density"]
    ds_v1v3 = f["kinetic/hot_elec/f/v1_v3_density"]

    f_v3_all = np.asarray(ds_v3[:], dtype=float)
    f_v1v3_all = np.asarray(ds_v1v3[:], dtype=float)

    v3 = centers(ds_v3, 1)
    v1 = centers(ds_v1v3, 1)
    v3_2d = centers(ds_v1v3, 2)

if f_v3_all.ndim != 2:
    raise RuntimeError(f"Expected v3_density to be 2-D, got {f_v3_all.shape}")
if f_v1v3_all.ndim != 3:
    raise RuntimeError(
        f"Expected v1_v3_density to be 3-D, got {f_v1v3_all.shape}"
    )

if not np.allclose(v3, v3_2d):
    raise RuntimeError("v3 grids from 1-D and 2-D diagnostics do not match.")

if f_v3_all.shape[0] != len(t) or f_v1v3_all.shape[0] != len(t):
    raise RuntimeError("Distribution-history length does not match time history.")

f_v3_0 = f_v3_all[0]
f_v3_f = f_v3_all[-1]

F0 = f_v1v3_all[0]
Ff = f_v1v3_all[-1]

dv1 = float(np.mean(np.diff(v1)))
dv3 = float(np.mean(np.diff(v3)))

# Integrate the 2-D v1-v3 distribution over v3 to obtain the
# available current-method v1 marginal.
f_v1_0 = np.sum(F0, axis=1) * dv3
f_v1_f = np.sum(Ff, axis=1) * dv3

# Integrate over v1 as a consistency check against the v3 diagnostic.
f_v3_from_2d_0 = np.sum(F0, axis=0) * dv1
f_v3_from_2d_f = np.sum(Ff, axis=0) * dv1

print(f"Time range: {t[0]:.8g} -> {t[-1]:.8g}")
print("v3 grid:", v3.shape, float(v3.min()), float(v3.max()))
print("v1 grid:", v1.shape, float(v1.min()), float(v1.max()))
print("v1-v3 history:", f_v1v3_all.shape)

# Basic consistency diagnostics.
norm_v3_0 = np.sum(f_v3_0) * dv3
norm_v3_f = np.sum(f_v3_f) * dv3
norm_2d_0 = np.sum(F0) * dv1 * dv3
norm_2d_f = np.sum(Ff) * dv1 * dv3

marginal_err_0 = np.linalg.norm(f_v3_from_2d_0 - f_v3_0) / max(
    np.linalg.norm(f_v3_0), 1e-300
)
marginal_err_f = np.linalg.norm(f_v3_from_2d_f - f_v3_f) / max(
    np.linalg.norm(f_v3_f), 1e-300
)

print(f"v3 integral at t=0   : {norm_v3_0:.8e}")
print(f"v3 integral at t=T   : {norm_v3_f:.8e}")
print(f"2-D integral at t=0  : {norm_2d_0:.8e}")
print(f"2-D integral at t=T  : {norm_2d_f:.8e}")
print(f"v3 marginal relative error at t=0: {marginal_err_0:.8e}")
print(f"v3 marginal relative error at t=T: {marginal_err_f:.8e}")

# ------------------------------------------------------------------
# Plotting style kept compact and thesis-like.
# ------------------------------------------------------------------
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

fig, ax = plt.subplots(2, 2, figsize=(5.8, 4.35))

# Parallel distribution.
ax[0, 0].plot(v3, f_v3_0, color="darkorange", lw=0.9, label="t = 0")
ax[0, 0].plot(v3, f_v3_f, color="purple", lw=0.9, label=f"t = {t[-1]:g}")
ax[0, 0].axvline(+V_RESONANCE, color="black", ls="--", lw=0.7)
ax[0, 0].axvline(-V_RESONANCE, color="black", ls="--", lw=0.7)
ax[0, 0].set_xlabel(r"$v_3$")
ax[0, 0].set_ylabel(r"$f_h(v_3)$")
ax[0, 0].set_title("Parallel velocity distribution")
ax[0, 0].legend(frameon=False)

# Perpendicular component marginal.
ax[0, 1].plot(v1, f_v1_0, color="darkorange", lw=0.9, label="t = 0")
ax[0, 1].plot(v1, f_v1_f, color="purple", lw=0.9, label=f"t = {t[-1]:g}")
ax[0, 1].set_xlabel(r"$v_1$")
ax[0, 1].set_ylabel(r"$f_h(v_1)$")
ax[0, 1].set_title("Perpendicular-component marginal")
ax[0, 1].legend(frameon=False)

# Change in parallel distribution.
ax[1, 0].plot(v3, f_v3_f - f_v3_0, color="purple", lw=0.9)
ax[1, 0].axhline(0.0, color="black", lw=0.6)
ax[1, 0].axvline(+V_RESONANCE, color="black", ls="--", lw=0.7)
ax[1, 0].axvline(-V_RESONANCE, color="black", ls="--", lw=0.7)
ax[1, 0].set_xlabel(r"$v_3$")
ax[1, 0].set_ylabel(r"$\Delta f_h(v_3)$")
ax[1, 0].set_title("Parallel-distribution change")

# Change in perpendicular-component marginal.
ax[1, 1].plot(v1, f_v1_f - f_v1_0, color="purple", lw=0.9)
ax[1, 1].axhline(0.0, color="black", lw=0.6)
ax[1, 1].set_xlabel(r"$v_1$")
ax[1, 1].set_ylabel(r"$\Delta f_h(v_1)$")
ax[1, 1].set_title("Perpendicular-marginal change")

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.11,
    top=0.91,
    wspace=0.30,
    hspace=0.38,
)

png = OUT / "Figure_4_3_current.png"
pdf = OUT / "Figure_4_3_current.pdf"
npz = OUT / "fig4_3_current_data.npz"

fig.savefig(png, dpi=400)
fig.savefig(pdf)
plt.close(fig)

np.savez(
    npz,
    time=t,
    v3=v3,
    v1=v1,
    f_v3_initial=f_v3_0,
    f_v3_final=f_v3_f,
    f_v1_initial=f_v1_0,
    f_v1_final=f_v1_f,
    v1_v3_initial=F0,
    v1_v3_final=Ff,
    resonance_velocity=V_RESONANCE,
)

print("=" * 72)
print("FIGURE 4.3 CURRENT-METHOD POST-PROCESSING COMPLETE")
print("=" * 72)
print("Saved:", png)
print("Saved:", pdf)
print("Saved:", npz)
