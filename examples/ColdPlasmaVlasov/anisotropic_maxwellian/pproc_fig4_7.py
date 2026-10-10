from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionPatch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "thesis_fig4_7_run3" / "run3_fig4_7.npz"
OUT = ROOT / "thesis_fig4_7_run3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Historical utility
sys.path.insert(0, str(ROOT))
import Utilitis_HybridCode as utils


# ----------------------------------------------------------------------
# Load saved Run-3 simulation
# ----------------------------------------------------------------------
d = np.load(DATA)

time = d["time"]
z = d["z"]
bx_coeff = d["Bx"]
par = d["parameters"]

T = float(time[-1])
dt = float(time[1] - time[0])
Lz = float(z[-1] - z[0])
Nz = len(z) - 1

# Historical Run-3 parameters
c = 1.0
wce = -1.0
wpe = 2.0
p = 3


print("================================================")
print("POST-PROCESSING THESIS FIGURE 4.7")
print("================================================")
print(f"Time samples : {len(time)}")
print(f"Spatial pts  : {len(z)}")
print(f"Lz           : {Lz}")
print(f"Nz           : {Nz}")
print(f"dt           : {dt}")
print(f"Tend         : {T}")
print("================================================")


# ----------------------------------------------------------------------
# Reconstruct historical FEM basis
# createBasis returns:
#     bsp, N, quad_points, weights
#
# The historical notebook then evaluates each Bx coefficient vector
# onto the element grid:
#
#     Bx[i] = utils.evaluation(bx_save[i], bsp, zj, zj)
#
# Here we explicitly construct zj because the current createBasis()
# returns N as its second object.
# ----------------------------------------------------------------------
bsp, Nbasis, quad_points, weights = utils.createBasis(Lz, Nz, p)
zj = np.linspace(0.0, Lz, Nz + 1)

print("Evaluating FEM Bx coefficients on spatial grid...")

Bx = np.empty_like(bx_coeff)

for i in range(len(time)):
    if i % 1000 == 0:
        print(f"  spatial evaluation: {i}/{len(time)}")

    Bx[i] = utils.evaluation(
        bx_coeff[i],
        bsp,
        zj,
        z,
    )

print("Bx evaluated.")
print("Evaluated Bx range:", Bx.min(), Bx.max())
print("Evaluated Bx std  :", Bx.std())


# ----------------------------------------------------------------------
# HISTORICAL FOURIER AXES
# ----------------------------------------------------------------------
Nt = int(T / dt)

w = np.linspace(0, Nt, Nt + 1) - Nt / 2
w = 2.0 * np.pi / T * w

ks = np.linspace(0, Nz, Nz + 1) - Nz / 2
ks = 2.0 * np.pi / Lz * ks


# ----------------------------------------------------------------------
# HISTORICAL 2-D FOURIER TRANSFORM
# ----------------------------------------------------------------------
print("Computing 2-D FFT...")

Bxkw = np.fft.fft2(Bx)

K, W = np.meshgrid(ks, w)

Bkw = np.fft.fftshift(Bxkw)
Bkw_plot = np.abs(Bkw)

Bkw_norm = Bkw_plot / Bkw_plot.max()


# ----------------------------------------------------------------------
# HISTORICAL DISPERSION RELATION
#
# Exactly as used in the original thesis notebook:
#
#   k1 = 0.1 ... 8
#   three cold-plasma branches
#   solveDispersionCold(...)
# ----------------------------------------------------------------------
print("Computing analytical cold-plasma branches...")

k1 = np.linspace(0.1, 8.0, 40)

w1_1 = np.zeros(40)
w1_2 = np.zeros(40)
w1_3 = np.zeros(40)

w1_1[0] = 0.0001
w1_2[0] = 1.5001
w1_3[0] = 2.5001

for i in range(40):

    if i == 0:

        w1_1[i] = utils.solveDispersionCold(
            k1[i], +1, c, wce, wpe,
            w1_1[i], 1e-6, 100
        )[0]

        w1_2[i] = utils.solveDispersionCold(
            k1[i], -1, c, wce, wpe,
            w1_2[i], 1e-6, 100
        )[0]

        w1_3[i] = utils.solveDispersionCold(
            k1[i], +1, c, wce, wpe,
            w1_3[i], 1e-6, 100
        )[0]

    else:

        w1_1[i] = utils.solveDispersionCold(
            k1[i], +1, c, wce, wpe,
            w1_1[i - 1], 1e-6, 100
        )[0]

        w1_2[i] = utils.solveDispersionCold(
            k1[i], -1, c, wce, wpe,
            w1_2[i - 1], 1e-6, 100
        )[0]

        w1_3[i] = utils.solveDispersionCold(
            k1[i], +1, c, wce, wpe,
            w1_3[i - 1], 1e-6, 100
        )[0]


# ----------------------------------------------------------------------
# HISTORICAL FIGURE STYLE
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# PROFESSIONAL THESIS TYPOGRAPHY
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 17,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

ticks = np.power(10.0, np.linspace(-8, 0, 5))
lvls = np.logspace(-8, 0, 60)


# ----------------------------------------------------------------------
# FIGURE 4.7
# ----------------------------------------------------------------------
f8 = plt.figure()
f8.set_figheight(5)
f8.set_figwidth(12)

# =========================
# LEFT PANEL
# =========================
ax1 = plt.subplot(121)

cf1 = plt.contourf(
    K,
    W,
    Bkw_norm,
    cmap="jet",
    norm=LogNorm(),
    levels=lvls,
)

# Historical black zoom box
plt.plot(np.linspace(0, 8, 10), np.zeros(10), "k")
plt.plot(np.linspace(0, 8, 10), np.ones(10) * 6, "k")
plt.plot(np.zeros(10), np.linspace(0, 6, 10), "k")
plt.plot(np.ones(10) * 8, np.linspace(0, 6, 10), "k")

plt.ylim((-20, 20))

plt.xlabel(r"$kc/ |\Omega_\mathrm{ce}|$")
plt.ylabel(r"$\omega_\mathrm{r}/ |\Omega_\mathrm{ce}|$")

plt.title(
    r"$|\hat{B}_x(k,\omega_\mathrm{r})|/"
    r"|\hat{B}_x(k,\omega_\mathrm{r})|_\mathrm{max}$",
    fontsize=18,
)


# =========================
# RIGHT PANEL
# =========================
ax2 = plt.subplot(122)

cf2 = plt.contourf(
    K,
    W,
    Bkw_norm,
    cmap="jet",
    norm=LogNorm(),
    levels=lvls,
)

plt.xlim((0, 8))
plt.ylim((0, 6))

plt.colorbar(cf2, ticks=ticks)

# Historical thesis intentionally suppresses tick labels
plt.xticks([], [])
plt.yticks([], [])

# Three analytical cold-plasma branches
plt.plot(k1, w1_1, "k--", linewidth=1.5)
plt.plot(k1, w1_2, "k--", linewidth=1.5)
plt.plot(k1, w1_3, "k--", linewidth=1.5)

plt.subplots_adjust(hspace=0.3)
plt.tight_layout()


# ----------------------------------------------------------------------
# HISTORICAL CONNECTION PATCHES
# ----------------------------------------------------------------------
con1 = ConnectionPatch(
    xyA=(0, 6),
    xyB=(0, 6),
    coordsA="data",
    coordsB="data",
    axesA=ax2,
    axesB=ax1,
    color="black",
    linewidth=1,
)

con2 = ConnectionPatch(
    xyA=(0, 0),
    xyB=(0, 0),
    coordsA="data",
    coordsB="data",
    axesA=ax2,
    axesB=ax1,
    color="black",
    linewidth=1,
)

con3 = ConnectionPatch(
    xyA=(0, 4.3),
    xyB=(8, 6),
    coordsA="data",
    coordsB="data",
    axesA=ax2,
    axesB=ax1,
    color="black",
    linewidth=1,
)

con4 = ConnectionPatch(
    xyA=(0, 2.4),
    xyB=(8, 0),
    coordsA="data",
    coordsB="data",
    axesA=ax2,
    axesB=ax1,
    color="black",
    linewidth=1,
)

ax2.add_artist(con1)
ax2.add_artist(con2)
ax2.add_artist(con3)
ax2.add_artist(con4)


# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
outfile_png = OUT / "fig4_7.png"
outfile_pdf = OUT / "fig4_7.pdf"

f8.savefig(outfile_png, dpi=400, bbox_inches="tight")
f8.savefig(outfile_pdf, bbox_inches="tight")

plt.close(f8)


# Also preserve the evaluated field used to make the FFT.
np.savez_compressed(
    OUT / "fig4_7_evaluated_Bx.npz",
    time=time,
    z=z,
    Bx=Bx,
    omega=w,
    k=ks,
)


print("================================================")
print("FIGURE 4.7 COMPLETE")
print("================================================")
print("PNG :", outfile_png)
print("PDF :", outfile_pdf)
print("Field:", OUT / "fig4_7_evaluated_Bx.npz")
print("h =", Lz / Nz)
print("hk limit = 2.5")
print("k resolution limit =", 2.5 / (Lz / Nz))
print("================================================")
