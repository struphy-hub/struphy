import os
import numpy as np
import matplotlib.pyplot as plt

BASE = (
    "thesis_postprocess_validation/post_processing/"
    "kinetic_data/hot_elec/distribution_function/v1_v3_density"
)

TIME_PATH = (
    "thesis_postprocess_validation/post_processing/t_grid.npy"
)

OUT = "validation_plots"
os.makedirs(OUT, exist_ok=True)

t = np.load(TIME_PATH)

v1 = np.load(os.path.join(BASE, "grid_v1.npy"))
v3 = np.load(os.path.join(BASE, "grid_v3.npy"))
F = np.load(os.path.join(BASE, "f_binned.npy"))

dv1 = np.mean(np.diff(v1))
dv3 = np.mean(np.diff(v3))


def moments(Fi):

    norm = np.sum(Fi) * dv1 * dv3

    mean1 = (
        np.sum(v1[:, None] * Fi)
        * dv1 * dv3 / norm
    )

    mean3 = (
        np.sum(v3[None, :] * Fi)
        * dv1 * dv3 / norm
    )

    var1 = (
        np.sum(
            (v1[:, None] - mean1) ** 2 * Fi
        )
        * dv1 * dv3 / norm
    )

    var3 = (
        np.sum(
            (v3[None, :] - mean3) ** 2 * Fi
        )
        * dv1 * dv3 / norm
    )

    covariance = (
        np.sum(
            (v1[:, None] - mean1)
            * (v3[None, :] - mean3)
            * Fi
        )
        * dv1 * dv3 / norm
    )

    return (
        norm,
        mean1,
        mean3,
        np.sqrt(var1),
        np.sqrt(var3),
        covariance,
    )


print("============================================================")
print("V1-V3 ANISOTROPY VALIDATION")
print("============================================================")

for index in [0, len(t) // 2, len(t) - 1]:

    (
        norm,
        mean1,
        mean3,
        sigma1,
        sigma3,
        covariance,
    ) = moments(F[index])

    print(f"\nt = {t[index]:.6f}")

    print("integral =", norm)
    print("mean_v1 =", mean1)
    print("mean_v3 =", mean3)
    print("sigma_v1 =", sigma1)
    print("sigma_v3 =", sigma3)
    print("variance ratio =", sigma1**2 / sigma3**2)
    print("covariance =", covariance)


def save_distribution(index, filename, title):

    plt.figure(figsize=(9, 7))

    plt.pcolormesh(
        v3,
        v1,
        F[index],
        shading="auto",
    )

    plt.xlabel(r"$v_3$")
    plt.ylabel(r"$v_1$")
    plt.title(title)

    plt.colorbar(label="Distribution density")

    plt.tight_layout()

    path = os.path.join(OUT, filename)

    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


save_distribution(
    0,
    "v1_v3_distribution_t0.png",
    "Hot-Electron v1-v3 Distribution at t = 0",
)

save_distribution(
    -1,
    "v1_v3_distribution_t1.png",
    "Hot-Electron v1-v3 Distribution at Final Time",
)

# ------------------------------------------------------------
# COMPUTE MARGINAL DISTRIBUTIONS
# ------------------------------------------------------------

# Initial distribution
F0 = F[0]

# Final distribution
F1 = F[-1]

# Integrate over the opposite coordinate
#
# f(v1) = ∫ f(v1,v3) dv3
# f(v3) = ∫ f(v1,v3) dv1

f0_v1 = np.sum(F0, axis=1) * dv3
f0_v3 = np.sum(F0, axis=0) * dv1

f1_v1 = np.sum(F1, axis=1) * dv3
f1_v3 = np.sum(F1, axis=0) * dv1

# Normalize only for visualization
f0_v1 /= np.max(f0_v1)
f0_v3 /= np.max(f0_v3)

f1_v1 /= np.max(f1_v1)
f1_v3 /= np.max(f1_v3)

# ------------------------------------------------------------
# MOMENTS
# ------------------------------------------------------------

sigma_v1_0 = np.sqrt(
    np.sum((v1**2) * f0_v1) * dv1 /
    np.sum(f0_v1 * dv1)
)

sigma_v3_0 = np.sqrt(
    np.sum((v3**2) * f0_v3) * dv3 /
    np.sum(f0_v3 * dv3)
)

sigma_v1_1 = np.sqrt(
    np.sum((v1**2) * f1_v1) * dv1 /
    np.sum(f1_v1 * dv1)
)

sigma_v3_1 = np.sqrt(
    np.sum((v3**2) * f1_v3) * dv3 /
    np.sum(f1_v3 * dv3)
)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14,5),
    sharey=True
)

# Initial

ax[0].plot(
    v1,
    f0_v1,
    lw=3,
    label=rf"$v_1$ ($\sigma$={sigma_v1_0:.3f})"
)

ax[0].plot(
    v3,
    f0_v3,
    lw=3,
    label=rf"$v_3$ ($\sigma$={sigma_v3_0:.3f})"
)

ax[0].set_title("Initial Distribution ($t=0$)")
ax[0].set_xlabel("Velocity")
ax[0].set_ylabel("Normalized distribution")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# Final

ax[1].plot(
    v1,
    f1_v1,
    lw=3,
    label=rf"$v_1$ ($\sigma$={sigma_v1_1:.3f})"
)

ax[1].plot(
    v3,
    f1_v3,
    lw=3,
    label=rf"$v_3$ ($\sigma$={sigma_v3_1:.3f})"
)

ax[1].set_title("Final Distribution ($t=1$)")
ax[1].set_xlabel("Velocity")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

plt.tight_layout()

plt.savefig(
    "validation_plots/v1_v3_distribution_t0_t1.png",
    dpi=300,
    bbox_inches="tight",
)

print()
print("Saved:")
print("validation_plots/v1_v3_distribution_t0_t1.png")
