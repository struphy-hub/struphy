from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

from struphy.bsplines.evaluation_kernels_3d import evaluate_3d
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid
from feectools.ddm.mpi import mpi as MPI


# ============================================================
# Current-method Fig. 4.2 post-processing
#
# Reconstruct B_x from the saved FEEC H(div) coefficients,
# form the magnetic-field energy signal B_x^2, and determine
# its dominant frequency.
#
# No simulation is run by this script.
# ============================================================

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "thesis_run1_fig4_9"
H5 = RUN / "data" / "data_proc0.hdf5"
OUT = RUN / "post_processing"
OUT.mkdir(parents=True, exist_ok=True)

# Analytical frequency from the dispersion relation.
OMEGA_R = 0.4742
OMEGA_E = 2.0 * OMEGA_R
F_EXPECTED = OMEGA_E / (2.0 * np.pi)

# Logical FEEC evaluation point.
# Physical domain: [0,1] x [0,1] x [0,pi]
# Hence eta_3 = z/pi.
ETA1 = 0.5
ETA2 = 0.5
ETA3 = 0.25

Z_PHYSICAL = np.pi * ETA3
EXPECTED_BX0 = 1.0e-4 * np.sin(2.0 * Z_PHYSICAL)

print("=" * 72)
print("CURRENT FIG. 4.2 — FEEC POST-PROCESSING")
print("=" * 72)
print("Run:", RUN)
print("No simulation will be run.")

# ------------------------------------------------------------
# Load saved FEEC magnetic-field coefficients
# ------------------------------------------------------------
with h5py.File(H5, "r") as f:
    t = np.asarray(f["time/value"][:], dtype=float)
    bx_coeff = np.asarray(
        f["feec/em_fields/b_field/1"][:],
        dtype=float,
    )

# ------------------------------------------------------------
# Reconstruct the same FEEC spaces used by the simulation
# ------------------------------------------------------------
comm = MPI.COMM_WORLD

grid = TensorProductGrid(
    num_elements=(1, 1, 32),
)

derham = Derham(
    grid,
    DerhamOptions(
        degree=(1, 1, 1),
        bcs=(None, None, None),
    ),
    comm=comm,
)

# evaluate_3d requires the complete knot vectors and original
# index arrays.
tn1, tn2, tn3 = derham.V0fem.knots
indN = derham.indN
indD = derham.indD

p1 = derham.degree[0]
p2 = derham.degree[1] - 1
p3 = derham.degree[2] - 1

# B_x is the first H(div) component: N x D x D.
kind1 = 1
kind2 = 2
kind3 = 2

ind1 = indN[0]
ind2 = indD[1]
ind3 = indD[2]

# ------------------------------------------------------------
# Initial-field validation
# ------------------------------------------------------------
bx0_logical = evaluate_3d(
    kind1,
    kind2,
    kind3,
    tn1,
    tn2,
    tn3,
    p1,
    p2,
    p3,
    ind1,
    ind2,
    ind3,
    bx_coeff[0],
    ETA1,
    ETA2,
    ETA3,
)

# For the affine map [0,1]x[0,1]x[0,pi], the H(div) push
# transformation gives physical B_x = logical B_x / pi.
bx0_physical = bx0_logical / np.pi

relative_error_bx0 = (
    abs(bx0_physical - EXPECTED_BX0) / abs(EXPECTED_BX0)
)

print()
print("INITIAL-FIELD CHECK")
print(f"logical point = ({ETA1}, {ETA2}, {ETA3})")
print(f"physical z    = {Z_PHYSICAL:.12e}")
print(f"Bx(0) logical = {bx0_logical:.12e}")
print(f"Bx(0) physical = {bx0_physical:.12e}")
print(f"Bx(0) expected = {EXPECTED_BX0:.12e}")
print(f"relative error = {relative_error_bx0:.6e}")

# The p=1 FEEC representation does not reproduce the sine
# pointwise to machine precision. Check sign and reasonable
# amplitude instead of imposing a false exact-equality test.
if np.sign(bx0_physical) != np.sign(EXPECTED_BX0):
    raise RuntimeError("FEEC reconstruction has the wrong initial-field sign.")

if not (0.95 <= abs(bx0_physical / EXPECTED_BX0) <= 1.05):
    raise RuntimeError(
        "FEEC reconstruction has an unexpectedly large initial-field "
        "amplitude error."
    )

print("INITIAL FIELD: PASS — discrete FEEC representation")

# ------------------------------------------------------------
# Reconstruct B_x(t)
# ------------------------------------------------------------
bx = np.empty(len(t), dtype=float)

for i in range(len(t)):
    bx_logical = evaluate_3d(
        kind1,
        kind2,
        kind3,
        tn1,
        tn2,
        tn3,
        p1,
        p2,
        p3,
        ind1,
        ind2,
        ind3,
        bx_coeff[i],
        ETA1,
        ETA2,
        ETA3,
    )

    bx[i] = bx_logical / np.pi

# ------------------------------------------------------------
# Magnetic-field energy signal
#
# B_x oscillates at omega_r, so B_x^2 oscillates at 2 omega_r.
# ------------------------------------------------------------
bx_energy = bx**2

# ------------------------------------------------------------
# Frequency extraction
# ------------------------------------------------------------
T0 = 20.0
T1 = 100.0

mask = (t >= T0) & (t <= T1)

tw = t[mask]
yw = bx_energy[mask]

# Remove the slowly varying instability envelope before FFT.
yw = yw - np.polyval(
    np.polyfit(tw, yw, 1),
    tw,
)

dt = float(np.mean(np.diff(tw)))

freq = np.fft.rfftfreq(
    len(yw),
    d=dt,
)

spec = np.abs(np.fft.rfft(yw))

# Exclude the low-frequency envelope.
valid = freq >= 0.05

peak_index = np.argmax(spec[valid])
peak_freq = freq[valid][peak_index]
omega_peak = 2.0 * np.pi * peak_freq

print()
print("FREQUENCY CHECK")
print(f"FFT interval = {T0:.3f} to {T1:.3f}")
print(f"dominant Bx^2 frequency = {peak_freq:.8f} cycles/time")
print(f"dominant angular frequency = {omega_peak:.8f}")
print(f"expected 2*omega_r = {OMEGA_E:.8f}")
print(f"expected frequency = {F_EXPECTED:.8f} cycles/time")
print(f"frequency-bin spacing = {freq[1] - freq[0]:.8f}")

# ------------------------------------------------------------
# Final thesis-style current-method Fig. 4.2
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(3.6, 2.5))

ax.plot(
    freq[valid],
    spec[valid],
    color="purple",
    lw=0.9,
    label=r"$B_x^2$ spectrum",
)

ax.axvline(
    F_EXPECTED,
    color="black",
    ls="--",
    lw=0.8,
    label=r"$2\omega_r/(2\pi)$",
)

ax.set_xlabel(r"frequency $f$")
ax.set_ylabel(r"$|\widehat{B_x^2}|$")
ax.set_title("Magnetic field energy frequency")

ax.set_xlim(0.05, 0.30)

ax.legend(
    loc="upper right",
    frameon=False,
)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.18,
    top=0.88,
)

png = OUT / "Figure_4_2_current_FEEC.png"
pdf = OUT / "Figure_4_2_current_FEEC.pdf"

fig.savefig(png, dpi=400)
fig.savefig(pdf)

plt.close(fig)

print()
print("FIGURE SAVED")
print(png)
print(pdf)
print("=" * 72)
