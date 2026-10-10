import os
import numpy as np
import matplotlib.pyplot as plt

BASE = (
    "thesis_postprocess_validation/post_processing/"
    "kinetic_data/hot_elec/distribution_function/v3_density"
)

TIME_PATH = (
    "thesis_postprocess_validation/post_processing/t_grid.npy"
)

OUT = "validation_plots"
os.makedirs(OUT, exist_ok=True)

t = np.load(TIME_PATH)

v3 = np.load(os.path.join(BASE, "grid_v3.npy"))
F = np.load(os.path.join(BASE, "f_binned.npy"))
DF = np.load(os.path.join(BASE, "delta_f_binned.npy"))

dv = np.mean(np.diff(v3))

density = 0.06
sigma_target = 0.2

analytic = (
    density
    / (np.sqrt(2 * np.pi) * sigma_target)
    * np.exp(
        -(v3 ** 2)
        / (2 * sigma_target ** 2)
    )
)

f0 = F[0]

integral = np.sum(f0) * dv

mean = np.sum(v3 * f0) * dv / integral

variance = (
    np.sum((v3 - mean) ** 2 * f0)
    * dv
    / integral
)

sigma = np.sqrt(variance)

relative_l2 = (
    np.linalg.norm(f0 - analytic)
    / np.linalg.norm(analytic)
)

integrals = np.sum(F, axis=1) * dv

print("============================================================")
print("V3 MAXWELLIAN VALIDATION")
print("============================================================")

print("grid shape =", v3.shape)
print("distribution shape =", F.shape)

print("\nInitial integral =", integral)
print("mean =", mean)
print("sigma =", sigma)
print("target sigma =", sigma_target)

print("\nrelative L2 error =", relative_l2)

print(
    "maximum relative integral variation =",
    np.max(np.abs(integrals - integrals[0]))
    / abs(integrals[0]),
)

print(
    "Delta-f identically zero =",
    np.all(DF == 0),
)

plt.figure(figsize=(10, 6))

plt.step(
    v3,
    f0 / density,
    where="mid",
    label=r"Loaded $v_3$ distribution",
)

plt.plot(
    v3,
    analytic / density,
    "--",
    label=r"Prescribed Maxwellian, $\sigma=0.2$",
)

plt.xlabel(r"$v_3$")
plt.ylabel("Probability density")
plt.title("Validation of Hot-Electron v3 Distribution")
plt.legend()
plt.grid()

plt.tight_layout()

path = os.path.join(
    OUT,
    "v3_distribution_validation.png",
)

plt.savefig(path, dpi=200)
plt.close()

print("\nSaved:", path)
