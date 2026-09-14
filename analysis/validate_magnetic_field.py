import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

BASE = "thesis_postprocess_validation/post_processing"
FIELDS = os.path.join(BASE, "fields_data")
OUT = "validation_plots"

os.makedirs(OUT, exist_ok=True)

with open(os.path.join(FIELDS, "grids_phy.bin"), "rb") as f:
    grids = pickle.load(f)

with open(
    os.path.join(FIELDS, "em_fields/b_field_phy.bin"), "rb"
) as f:
    B = pickle.load(f)

z = np.asarray(grids[2])[0, 0, :]

times = sorted(B.keys())
t0 = times[0]

ix = np.asarray(grids[0]).shape[0] // 2
iy = np.asarray(grids[1]).shape[1] // 2

Bx = np.asarray(B[t0])[0, ix, iy, :]
By = np.asarray(B[t0])[1, ix, iy, :]
Bz = np.asarray(B[t0])[2, ix, iy, :]

target = 1e-4 * np.sin(2 * z)

max_abs_error = np.max(np.abs(Bx - target))
relative_l2 = np.linalg.norm(Bx - target) / np.linalg.norm(target)

basis = np.sin(2 * z)
A_fit = np.dot(Bx, basis) / np.dot(basis, basis)

fit = A_fit * basis

relative_residual = (
    np.linalg.norm(Bx - fit) /
    np.linalg.norm(Bx)
)

print("============================================================")
print("INITIAL MAGNETIC FIELD VALIDATION")
print("============================================================")

print("time =", t0)

print("\nz range:")
print(np.min(z), np.max(z))

print("\nBx:")
print("min =", np.min(Bx))
print("max =", np.max(Bx))
print("max abs =", np.max(np.abs(Bx)))

print("\nBy max abs =", np.max(np.abs(By)))
print("Bz max abs =", np.max(np.abs(Bz)))

print("\nComparison with intended 1e-4 sin(2z):")
print("max absolute error =", max_abs_error)
print("relative L2 error =", relative_l2)

print("\nBest fit:")
print("A_fit =", A_fit)
print("target amplitude =", 1e-4)
print(
    "relative amplitude error =",
    abs(A_fit - 1e-4) / 1e-4,
)
print("relative residual =", relative_residual)

plt.figure(figsize=(10, 6))

plt.plot(z, Bx, label=r"Struphy: $B_x(z)$")
plt.plot(
    z,
    target,
    "--",
    label=r"Expected: $10^{-4}\sin(2z)$",
)

plt.xlabel("z")
plt.ylabel(r"$B_x$")
plt.title("Initial Magnetic Perturbation Validation")
plt.legend()
plt.grid()

plt.tight_layout()

path = os.path.join(
    OUT,
    "magnetic_perturbation_validation.png",
)

plt.savefig(path, dpi=200)
plt.close()

print("\nSaved:", path)
