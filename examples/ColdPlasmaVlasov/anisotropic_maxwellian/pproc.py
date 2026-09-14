import os
import numpy as np
import matplotlib.pyplot as plt

from struphy import PostProcessor


SIM_PATH = os.path.abspath("thesis_fig4_9_validation")


def main():

    print("=" * 60)
    print("POST PROCESSING: ANISOTROPIC MAXWELLIAN")
    print("=" * 60)

    # --------------------------------------------------
    # Process raw Struphy output
    # --------------------------------------------------

    pp = PostProcessor(path_out=SIM_PATH)
    pp.process()

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    pproc = os.path.join(SIM_PATH, "post_processing")

    out = os.path.join(SIM_PATH, "validation_plots")
    os.makedirs(out, exist_ok=True)

    time_path = os.path.join(pproc, "t_grid.npy")

    base = os.path.join(
        pproc,
        "kinetic_data",
        "hot_elec",
        "distribution_function",
    )

    # --------------------------------------------------
    # Load time
    # --------------------------------------------------

    t = np.load(time_path)

    print(f"\nLoaded {len(t)} time points")
    print(f"Time range: {t[0]} -> {t[-1]}")

    # ==================================================
    # E3 DISTRIBUTION
    # ==================================================

    print("\nProcessing e3 distribution...")

    e3_path = os.path.join(base, "e3_density")

    e3 = np.load(os.path.join(e3_path, "grid_e3.npy"))
    f_e3 = np.load(os.path.join(e3_path, "f_binned.npy"))

    indices = [0, len(t) // 2, len(t) - 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx in indices:
        ax.plot(
            e3,
            f_e3[idx],
            label=f"t = {t[idx]:.2f}"
        )

    ax.set_xlabel("e3")
    ax.set_ylabel("Distribution")
    ax.set_title("Hot-electron e3 distribution")
    ax.legend()

    fig.tight_layout()

    filename = os.path.join(out, "e3_density_selected_times.png")
    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"Saved: {filename}")

    # ==================================================
    # V3 DISTRIBUTION
    # ==================================================

    print("\nProcessing v3 distribution...")

    v3_path = os.path.join(base, "v3_density")

    v3 = np.load(os.path.join(v3_path, "grid_v3.npy"))
    f_v3 = np.load(os.path.join(v3_path, "f_binned.npy"))

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx in indices:
        ax.plot(
            v3,
            f_v3[idx],
            label=f"t = {t[idx]:.2f}"
        )

    ax.set_xlabel(r"$v_3$")
    ax.set_ylabel("Distribution")
    ax.set_title(r"Hot-electron $v_3$ distribution")
    ax.legend()

    fig.tight_layout()

    filename = os.path.join(out, "v3_distribution_validation.png")
    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"Saved: {filename}")

    # ==================================================
    # V1-V3 ANISOTROPY
    # ==================================================

    print("\nProcessing v1-v3 anisotropy...")

    anis_path = os.path.join(base, "v1_v3_density")

    v1 = np.load(os.path.join(anis_path, "grid_v1.npy"))
    v3 = np.load(os.path.join(anis_path, "grid_v3.npy"))

    F = np.load(os.path.join(anis_path, "f_binned.npy"))

    dv1 = np.mean(np.diff(v1))
    dv3 = np.mean(np.diff(v3))

    def moments(Fi):

        norm = np.sum(Fi) * dv1 * dv3

        mean_v1 = (
            np.sum(v1[:, None] * Fi)
            * dv1 * dv3 / norm
        )

        mean_v3 = (
            np.sum(v3[None, :] * Fi)
            * dv1 * dv3 / norm
        )

        variance_v1 = (
            np.sum(
                (v1[:, None] - mean_v1) ** 2 * Fi
            )
            * dv1 * dv3 / norm
        )

        variance_v3 = (
            np.sum(
                (v3[None, :] - mean_v3) ** 2 * Fi
            )
            * dv1 * dv3 / norm
        )

        covariance = (
            np.sum(
                (v1[:, None] - mean_v1)
                * (v3[None, :] - mean_v3)
                * Fi
            )
            * dv1 * dv3 / norm
        )

        return (
            norm,
            mean_v1,
            mean_v3,
            np.sqrt(variance_v1),
            np.sqrt(variance_v3),
            variance_v1 / variance_v3,
            covariance,
        )

    # Times to analyse
    indices = [0, len(t) // 2, len(t) - 1]

    print("\n" + "=" * 60)
    print("V1-V3 ANISOTROPY VALIDATION")
    print("=" * 60)

    for idx in indices:

        result = moments(F[idx])

        print(f"\nt = {t[idx]:.6f}")
        print(f"integral = {result[0]}")
        print(f"mean_v1 = {result[1]}")
        print(f"mean_v3 = {result[2]}")
        print(f"sigma_v1 = {result[3]}")
        print(f"sigma_v3 = {result[4]}")
        print(f"variance ratio = {result[5]}")
        print(f"covariance = {result[6]}")

    # --------------------------------------------------
    # Distinctive 2D distribution plots
    # --------------------------------------------------

    selected = [0, len(t) - 1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
    )

    for ax, idx in zip(axes, selected):

        pcm = ax.pcolormesh(
            v1,
            v3,
            F[idx].T,
            shading="auto",
        )

        ax.set_xlabel(r"$v_1$")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
            rf"$f(v_1,v_3)$ at $t={t[idx]:.2f}$"
        )

        fig.colorbar(
            pcm,
            ax=ax,
            label="Distribution"
        )

    fig.suptitle(
        r"Hot-electron velocity-space anisotropy: $v_1$-$v_3$"
    )

    fig.tight_layout()

    filename = os.path.join(
        out,
        "v1_v3_distribution_t0_t1.png"
    )

    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {filename}")

    print("\n" + "=" * 60)
    print("POST PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
