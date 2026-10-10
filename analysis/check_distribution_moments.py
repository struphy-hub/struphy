import os
import numpy as np

BASE = (
    "thesis_postprocess_validation/post_processing/"
    "kinetic_data/hot_elec/distribution_function/v1_v3_density"
)

TIME_PATH = (
    "thesis_postprocess_validation/post_processing/t_grid.npy"
)

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


results = np.array([moments(Fi) for Fi in F])

norm = results[:, 0]
mean1 = results[:, 1]
mean3 = results[:, 2]
sigma1 = results[:, 3]
sigma3 = results[:, 4]
covariance = results[:, 5]

anisotropy = sigma1**2 / sigma3**2

relative_change = np.array(
    [
        np.linalg.norm(Fi - F[0])
        / np.linalg.norm(F[0])
        for Fi in F
    ]
)

marginal_v1_initial = np.sum(F[0], axis=1) * dv3
marginal_v1_final = np.sum(F[-1], axis=1) * dv3

marginal_v3_initial = np.sum(F[0], axis=0) * dv1
marginal_v3_final = np.sum(F[-1], axis=0) * dv1

v1_change = (
    np.linalg.norm(
        marginal_v1_final - marginal_v1_initial
    )
    / np.linalg.norm(marginal_v1_initial)
)

v3_change = (
    np.linalg.norm(
        marginal_v3_final - marginal_v3_initial
    )
    / np.linalg.norm(marginal_v3_initial)
)

print("============================================================")
print("DISTRIBUTION MOMENT ANALYSIS")
print("============================================================")

print("\nInitial values:")
print("integral =", norm[0])
print("sigma_v1 =", sigma1[0])
print("sigma_v3 =", sigma3[0])
print("anisotropy =", anisotropy[0])
print("covariance =", covariance[0])

print("\nFinal values:")
print("integral =", norm[-1])
print("sigma_v1 =", sigma1[-1])
print("sigma_v3 =", sigma3[-1])
print("anisotropy =", anisotropy[-1])
print("covariance =", covariance[-1])

print("\nMarginal changes:")
print("v1 relative L2 change =", v1_change)
print("v3 relative L2 change =", v3_change)

print("\nFull 2D distribution change:")
print("initial -> final =", relative_change[-1])
print("maximum =", np.max(relative_change))
print(
    "time of maximum change =",
    t[np.argmax(relative_change)],
)

print("\nRanges over complete run:")

print(
    "sigma_v1:",
    np.min(sigma1),
    np.max(sigma1),
)

print(
    "sigma_v3:",
    np.min(sigma3),
    np.max(sigma3),
)

print(
    "anisotropy:",
    np.min(anisotropy),
    np.max(anisotropy),
)

print(
    "covariance:",
    np.min(covariance),
    np.max(covariance),
)

print("\nAnalysis complete.")
