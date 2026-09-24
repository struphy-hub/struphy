#!/usr/bin/env python3
"""
FINAL historical Figure 4.3 postprocessor.

This version intentionally uses the historical particle-distribution dataset
used by the original Standard-FEM plotting notebook:

    thesis_fig4_3_particles/run1_standard_cv.npz

The file was generated with CONTROL_VARIATE=0.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# The completed 39-minute run is NOT touched.
DATA = ROOT.parents[2] / "thesis_fig4_3_particles" / "run1_standard_cv.npz"
OUT = ROOT / "folder_fig4_3"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Run-1 parameters
# ---------------------------------------------------------------------
wpar = 0.20
wperp = 0.53
nu_h = 0.06
wpe = 2.0
wce = -1.0
k = 2.0
Lz = np.pi
omega_r = 0.4742
Np = 100000

nh = nu_h * wpe**2

# CORRECT resonance:
# v_R = |(omega_r + Omega_ce)/k|
#     = |(0.4742 - 1)/2|
#     = 0.2629 c
vR = abs((omega_r + wce) / k)

# ---------------------------------------------------------------------
# Load the completed NO-CV particle run
# ---------------------------------------------------------------------
if not DATA.exists():
    raise FileNotFoundError(
        f"\nCannot find:\n  {DATA}\n\n"
        "Use the existing thesis_fig4_3_particles directory containing "
        "the completed no-control-variate Run-1 simulation."
    )

d = np.load(DATA)
particles0 = np.asarray(d["particles_initial"], dtype=float)
particlesf = np.asarray(d["particles_final"], dtype=float)

# If metadata exists, verify that this really is the no-CV run.
if "parameters" in d:
    par = d["parameters"]
    control = int(par[-2])
    if control != 0:
        raise RuntimeError(
            f"Expected the historical CV-off particle run, but metadata "
            f"reports control_variate={control}."
        )
    Np = int(par[-3])

# ---------------------------------------------------------------------
# EXACT historical histogram definitions
# ---------------------------------------------------------------------
Nbin_par = 128
Nbin_perp = 32
Lv_par = 6.0
Lv_perp = 4.0

dv_par = Lv_par / Nbin_par
dv_perp = Lv_perp / Nbin_perp

vpar_edges = np.linspace(-Lv_par / 2.0, Lv_par / 2.0, Nbin_par + 1)
vperp_edges = np.linspace(0.0, Lv_perp, Nbin_perp + 1)

vpar = vpar_edges[:-1] + dv_par / 2.0
vperp = vperp_edges[:-1] + dv_perp / 2.0

# ---------------------------------------------------------------------
# Analytic initial anisotropic Maxwellian marginals
# ---------------------------------------------------------------------
def fh0(vx, vy, vz):
    return (
        nh / ((2.0 * np.pi) ** 1.5 * wpar * wperp**2)
        * np.exp(
            -vz**2 / (2.0 * wpar**2)
            - (vx**2 + vy**2) / (2.0 * wperp**2)
        )
    )

# Integrate the 3-D Maxwellian analytically over the other velocity
# coordinates, exactly as in the historical plotting procedure.
fpar0 = fh0(0.0, 0.0, vpar) * 2.0 * np.pi * wperp**2
fperp0 = (
    fh0(vperp / np.sqrt(2.0), vperp / np.sqrt(2.0), 0.0)
    * np.sqrt(2.0 * np.pi) * wpar
)

# ---------------------------------------------------------------------
# Historical particle marginals
# ---------------------------------------------------------------------
def particle_marginals(p):
    vperp_particles = np.sqrt(p[:, 1]**2 + p[:, 2]**2)

    bpar = np.digitize(p[:, 3], vpar_edges) - 1
    bperp = np.digitize(vperp_particles, vperp_edges) - 1

    # This is the historical normalization:
    #
    #   bincount(weights=particles[:,4])
    #       /(Np * dv * Lz)
    #
    # and for perpendicular velocity the cylindrical Jacobian
    # 2*pi*v_perp.
    fpar = np.bincount(
        bpar,
        weights=p[:, 4],
        minlength=Nbin_par,
    )[:Nbin_par] / (Np * dv_par * Lz)

    fperp = np.bincount(
        bperp,
        weights=p[:, 4],
        minlength=Nbin_perp,
    )[:Nbin_perp] / (
        Np * dv_perp * Lz * vperp * 2.0 * np.pi
    )

    return fpar, fperp


fparf, fperpf = particle_marginals(particlesf)

# Historical differences: final - initial.
dpar = fparf - fpar0
dperp = fperpf - fperp0

# ---------------------------------------------------------------------
# Typography: match the thesis, NOT the previous oversized version.
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

ORANGE = "darkorange"
PURPLE = "purple"
BLACK = "black"

# ---------------------------------------------------------------------
# Individual panels -- useful for checking against the thesis source.
# ---------------------------------------------------------------------
def save_panel(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# Top-left
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ax.plot(vpar, fpar0, color=ORANGE, lw=1.0)
ax.plot(vpar, fparf, color=PURPLE, lw=1.0)

top = float(np.max(fpar0))
ax.plot(
    [vR, vR], [0, top],
    color=BLACK, ls="--", lw=0.55, label=r"$v_{\mathrm{R}}$"
)
ax.plot([-vR, -vR], [0, top], color=BLACK, ls="--", lw=0.55)

ax.set_xlim(-1, 1)
ax.set_ylim(0, 0.5)
ax.set_xlabel(r"$v_\parallel/c$")
ax.set_ylabel(r"$f_{\mathrm{h}}(v_\parallel)c^3/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Initial and final parallel distribution")
ax.legend(loc="upper right", frameon=True)
save_panel(fig, "parallel")


# Top-right
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ax.plot(vperp, fperp0, color=ORANGE, lw=1.0, label=r"$t=0$")
ax.plot(vperp, fperpf, color=PURPLE, lw=1.0, label=r"$t=200\,|\Omega_{\mathrm{ce}}|$")

ax.set_xlim(0, 2)
ax.set_ylim(0, 0.2)
ax.set_xlabel(r"$v_\perp/c$")
ax.set_ylabel(r"$f_{\mathrm{h}}(v_\perp)c^2/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Initial and final perp. distribution")
ax.legend(loc="upper right", frameon=True)
save_panel(fig, "perpendicular")


# Bottom-left
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ax.axhline(0, color=BLACK, ls="--", lw=0.55)
ax.plot(vpar, dpar, color=PURPLE, lw=1.0)

ax.plot(
    [vR, vR], [-0.05, 0],
    color=BLACK, ls="--", lw=0.55, label=r"$v_{\mathrm{R}}$"
)
ax.plot([-vR, -vR], [-0.05, 0], color=BLACK, ls="--", lw=0.55)

ax.set_xlim(-1, 1)
ax.set_ylim(-0.05, 0.05)
ax.set_xlabel(r"$v_\parallel/c$")
ax.set_ylabel(r"$\delta f_{\mathrm{h}}(v_\parallel)c^3/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Difference parallel")
ax.legend(loc="lower right", frameon=True)
save_panel(fig, "delta_parallel")


# Bottom-right
fig, ax = plt.subplots(figsize=(2.8, 2.2))
ax.axhline(0, color=BLACK, ls="--", lw=0.55)
ax.plot(vperp, dperp, color=PURPLE, lw=1.0)

ax.set_xlim(0, 2)
ax.set_ylim(-0.005, 0.005)
ax.set_xlabel(r"$v_\perp/c$")
ax.set_ylabel(r"$\delta f_{\mathrm{h}}(v_\perp)c^2/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Difference perpendicular")
save_panel(fig, "delta_perpendicular")


# ---------------------------------------------------------------------
# Combined Figure 4.3 -- final comparison target
# ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.35))

# 1
ax = axs[0, 0]
ax.plot(vpar, fpar0, color=ORANGE, lw=1.0)
ax.plot(vpar, fparf, color=PURPLE, lw=1.0)
ax.plot([vR, vR], [0, top], "k--", lw=0.55, label=r"$v_{\mathrm{R}}$")
ax.plot([-vR, -vR], [0, top], "k--", lw=0.55)
ax.set_xlim(-1, 1)
ax.set_ylim(0, 0.5)
ax.set_xlabel(r"$v_\parallel/c$")
ax.set_ylabel(r"$f_{\mathrm{h}}(v_\parallel)c^3/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Initial and final parallel distribution")
ax.legend(loc="upper right", frameon=True)

# 2
ax = axs[0, 1]
ax.plot(vperp, fperp0, color=ORANGE, lw=1.0, label=r"$t=0$")
ax.plot(vperp, fperpf, color=PURPLE, lw=1.0, label=r"$t=200\,|\Omega_{\mathrm{ce}}|$")
ax.set_xlim(0, 2)
ax.set_ylim(0, 0.2)
ax.set_xlabel(r"$v_\perp/c$")
ax.set_ylabel(r"$f_{\mathrm{h}}(v_\perp)c^2/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Initial and final perp. distribution")
ax.legend(loc="upper right", frameon=True)

# 3
ax = axs[1, 0]
ax.axhline(0, color=BLACK, ls="--", lw=0.55)
ax.plot(vpar, dpar, color=PURPLE, lw=1.0)
ax.plot([vR, vR], [-0.05, 0], "k--", lw=0.55, label=r"$v_{\mathrm{R}}$")
ax.plot([-vR, -vR], [-0.05, 0], "k--", lw=0.55)
ax.set_xlim(-1, 1)
ax.set_ylim(-0.05, 0.05)
ax.set_xlabel(r"$v_\parallel/c$")
ax.set_ylabel(r"$\delta f_{\mathrm{h}}(v_\parallel)c^3/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Difference parallel")
ax.legend(loc="lower right", frameon=True)

# 4
ax = axs[1, 1]
ax.axhline(0, color=BLACK, ls="--", lw=0.55)
ax.plot(vperp, dperp, color=PURPLE, lw=1.0)
ax.set_xlim(0, 2)
ax.set_ylim(-0.005, 0.005)
ax.set_xlabel(r"$v_\perp/c$")
ax.set_ylabel(r"$\delta f_{\mathrm{h}}(v_\perp)c^2/|\Omega_{\mathrm{ce}}|$")
ax.set_title("Difference perpendicular")

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.11,
    top=0.945,
    wspace=0.30,
    hspace=0.42,
)

fig.savefig(OUT / "fig4_3_combined.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "fig4_3_combined.pdf", bbox_inches="tight")
plt.close(fig)

# Save arrays for auditability.
np.savez(
    OUT / "fig4_3_data.npz",
    vpar=vpar,
    vperp=vperp,
    fpar0=fpar0,
    fparf=fparf,
    fperp0=fperp0,
    fperpf=fperpf,
    dpar=dpar,
    dperp=dperp,
    vR=vR,
)

print("============================================================")
print("Historical Figure 4.3 post-processing complete")
print("============================================================")
print(f"Input: {DATA}")
print(f"Np: {Np}")
print(f"v_R = {vR:.6f} c")
print(f"Output: {OUT / 'fig4_3_combined.png'}")
print("")
print("Figure was generated and saved onto folder_fig4_3")
