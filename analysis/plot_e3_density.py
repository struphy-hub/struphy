import os
import numpy as np
import matplotlib.pyplot as plt

BASE = (
    "thesis_postprocess_validation/post_processing/"
    "kinetic_data/hot_elec/distribution_function/e3_density"
)

TIME_PATH = (
    "thesis_postprocess_validation/post_processing/t_grid.npy"
)

OUT = "validation_plots"
os.makedirs(OUT, exist_ok=True)

t = np.load(TIME_PATH)

e3 = np.load(os.path.join(BASE, "grid_e3.npy"))
F = np.load(os.path.join(BASE, "f_binned.npy"))
DF = np.load(os.path.join(BASE, "delta_f_binned.npy"))

de3 = np.mean(np.diff(e3))

integrals = np.sum(F, axis=1) * de3

print("============================================================")
print("E3 SPATIAL DENSITY VALIDATION")
print("============================================================")

print("grid shape =", e3.shape)
print("distribution shape =", F.shape)

print("\ne3 range:")
print(np.min(e3), np.max(e3))

print("\nIntegral:")
print("initial =", integrals[0])
print("final =", integrals[-1])
print("minimum =", np.min(integrals))
print("maximum =", np.max(integrals))

relative_variation = np.max(
    np.abs(integrals - integrals[0])
) / abs(integrals[0])

print("maximum relative variation =", relative_variation)

print(
    "\nDelta-f identically zero =",
    np.all(DF == 0),
)

indices = [0, len(t) // 2, len(t) - 1]

plt.figure(figsize=(10, 6))

for i in indices:
    plt.plot(
        e3,
        F[i],
        label=f"t = {t[i]:.3f}",
    )

plt.xlabel(r"$e_3$")
plt.ylabel("f")
plt.title("Hot-Electron Spatial Density Distribution")
plt.legend()
plt.grid()

plt.tight_layout()

path = os.path.join(
    OUT,
    "e3_density_selected_times.png",
)

plt.savefig(path, dpi=200)
plt.close()

print("\nSaved:", path)
