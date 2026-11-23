"""Generate the figures of the SPH documentation page (doc/sections/subsections/numerics-sph.rst).

Adapted from the lecture "Meshless Structure-Preserving Algorithms" (SPEGEO school, Strasbourg 2026,
https://github.com/spossann/2026_SPEGEO_Strasbourg_school), with the notation of the Struphy documentation.
The figure ``sph_convergence.png`` is taken from the lecture as is (it needs a long parameter scan).

The density figures evaluate Struphy's SPH kernels, so Struphy must be installed and compiled
(``struphy compile``). Run from anywhere::

    python doc/pics/sph/make_sph_figures.py

The PNG files are written next to this script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150

KERNEL_COLORS = {"gaussian": "#d95f02", "trigonometric": "#1b9e77", "linear": "#7570b3"}
EXACT_COLOR = "#0f4c81"


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def lagrange_euler_figure():
    """Lagrangian (follow a parcel) vs. Eulerian (observe at a fixed point) description of a flow."""
    fig, (ax_l, ax_e) = plt.subplots(1, 2, figsize=(9, 2.8))

    t = np.linspace(0.0, 1.1, 320)
    tau = t / t[-1]
    x0_values = np.linspace(0.14, 0.50, 5)

    def trajectory(tau, x0):
        target = 0.66 + 0.05 * np.sin(2.6 * (x0 - 0.14)) + 0.03 * (x0 - 0.32)
        relaxation = (1.0 - np.exp(-2.9 * tau)) / (1.0 - np.exp(-2.9))
        drift = relaxation * (target - x0)
        waviness = 0.032 * np.sin(1.8 * np.pi * tau + 2.2 * x0) * (1.0 - 0.30 * tau)
        bend = 0.050 * tau * (1.0 - tau) * (0.56 - x0)
        spread = 0.020 * tau * (1.0 - 0.65 * tau) * np.cos(1.7 * np.pi * x0)
        return x0 + drift + waviness + bend + spread

    trajectories = [trajectory(tau, x0) for x0 in x0_values]

    # Lagrangian: follow one parcel
    for xtraj in trajectories:
        ax_l.plot(t, xtraj, color="#7aa6c2", lw=1.8, alpha=0.95)
        k = 150
        ax_l.annotate(
            "",
            xy=(t[k + 8], xtraj[k + 8]),
            xytext=(t[k], xtraj[k]),
            arrowprops=dict(arrowstyle="->", color="#7aa6c2", lw=1.1),
        )
        ax_l.scatter([0.0], [xtraj[0]], s=28, color="#7aa6c2", zorder=5)

    xr = trajectories[2]
    ax_l.plot(t, xr, color=EXACT_COLOR, lw=2.8, zorder=6)
    ax_l.scatter([0.0], [xr[0]], s=105, facecolor="#ffd166", edgecolor=EXACT_COLOR, linewidth=1.4, zorder=7)
    ax_l.annotate(
        r"$\mathbf{x}_0$",
        xy=(0.0, xr[0]),
        xytext=(0.08, xr[0] - 0.10),
        color=EXACT_COLOR,
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color=EXACT_COLOR, lw=1.3),
    )
    for idx, lbl, dx, dy in [
        (55, r"$\Phi_{t_1}(\mathbf{x}_0)$", 0.03, 0.04),
        (150, r"$\Phi_{t_2}(\mathbf{x}_0)$", 0.03, -0.08),
        (255, r"$\Phi_{t_3}(\mathbf{x}_0)$", 0.03, 0.04),
    ]:
        ax_l.plot(t[idx], xr[idx], "o", ms=5.8, color=EXACT_COLOR, zorder=7)
        ax_l.text(t[idx] + dx, xr[idx] + dy, lbl, fontsize=10, color=EXACT_COLOR)
    ax_l.set_title("Lagrangian view", color="#1f9d8a", fontsize=13)

    # Eulerian: observe at a fixed point x_*
    for xtraj in trajectories:
        ax_e.plot(t, xtraj, color="#d98c8c", lw=1.8, alpha=0.9)
    x_star = 0.58
    ax_e.axhline(x_star, color="#8b1e3f", lw=2.2, zorder=4)
    ax_e.text(0.70, x_star - 0.055, r"$\mathbf{x} = \mathbf{x}_*$", color="#8b1e3f", fontsize=12)
    for xtraj in trajectories:
        hit = np.argmin(np.abs(xtraj - x_star))
        ax_e.plot(t[hit], xtraj[hit], "o", ms=5.6, color="#8b1e3f", zorder=6)
    ax_e.set_title("Eulerian view", color="#1f9d8a", fontsize=13)

    for ax in (ax_l, ax_e):
        ax.set_xlim(0.0, 1.12)
        ax.set_ylim(0.08, 0.92)
        ax.set_xlabel(r"time $t$")
        ax.set_ylabel(r"position $\mathbf{x}$")
        ax.grid(alpha=0.22)

    plt.tight_layout(pad=0.2)
    _save(fig, "sph_lagrange_euler.png")


def kernels_figure():
    """The 1D smoothing kernels of Struphy with h = 1."""
    from struphy.pic.sph_smoothing_kernels import gaussian_uni, linear_uni, trigonometric_uni

    x = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for name, fun in [("trigonometric", trigonometric_uni), ("gaussian", gaussian_uni), ("linear", linear_uni)]:
        ax.plot(x, [fun(xi, 1.0) for xi in x], color=KERNEL_COLORS[name], lw=2.5, label=name)
    ax.set_xlabel(r"$x/h$", fontsize=13)
    ax.set_ylabel(r"$h\, W_h(x)$", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, "sph_kernels.png")


def _particles(domain, background, pert, ppb, boxes_per_dim):
    """ParticlesSPH on a periodic domain, loaded by tesselation, with weights from the density."""
    from struphy import BoundaryParameters, LoadingParameters, SortingParameters
    from struphy.pic.particles import ParticlesSPH

    particles = ParticlesSPH(
        comm_world=None,
        loading_params=LoadingParameters(ppb=ppb, loading="tesselation"),
        boundary_params=BoundaryParameters(bc_sph=("periodic", "periodic", "periodic")),
        sorting_params=SortingParameters(boxes_per_dim=boxes_per_dim),
        bufsize=1.0,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
    )
    particles.draw_markers(sort=False)
    particles.initialize_weights()
    return particles


def density_1d_figure():
    """SPH density estimate in 1D with the three kernel types (tesselation, 4 markers per box, h = 1/24)."""
    from struphy import domains, equils, perturbations

    domain = domains.Cuboid(l1=1.0, r1=2.0, l2=10.0, r2=20.0, l3=100.0, r3=200.0)
    background = equils.ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain
    pert = {"n": perturbations.ModesCos(ls=(1,), amps=(1.0,))}
    particles = _particles(domain, background, pert, ppb=4, boxes_per_dim=(24, 1, 1))

    eta1 = np.linspace(0, 1.0, 100)
    ee1, ee2, ee3 = np.meshgrid(eta1, np.array([0.0]), np.array([0.0]), indexing="ij")
    rho_exact = 1.5 + np.cos(2 * np.pi * eta1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
    for ax, name in zip(axes, ["gaussian", "trigonometric", "linear"]):
        kernel = name + "_1d"
        rho = np.array(
            particles.eval_density(ee1, ee2, ee3, h1=1 / 24, h2=1.0, h3=1.0, kernel_type=kernel, derivative=0)
        ).squeeze()
        err = np.max(np.abs(rho - rho_exact)) / np.max(np.abs(rho_exact))
        ax.plot(eta1, rho_exact, color=EXACT_COLOR, lw=2.0, label="exact")
        ax.plot(eta1, rho, "--", color=KERNEL_COLORS[name], lw=2.0, label=kernel)
        ax.set_xlabel(r"$\eta_1$", fontsize=11)
        ax.set_title(f"{kernel}, rel. error = {err:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\rho$", fontsize=13)
    plt.tight_layout()
    _save(fig, "sph_density_1d.png")


def density_2d_figure():
    """SPH density estimate in 2D (tesselation on 12 x 12 boxes with 16 markers each, gaussian_2d, h = 1/12)."""
    from struphy import domains, equils, perturbations

    domain = domains.Cuboid(l1=1.0, r1=2.0, l2=0.0, r2=2.0, l3=100.0, r3=200.0)
    background = equils.ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain
    pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(1.0,))}
    particles = _particles(domain, background, pert, ppb=16, boxes_per_dim=(12, 12, 1))

    eta = np.linspace(0, 1.0, 50)
    ee1, ee2, ee3 = np.meshgrid(eta, eta, np.array([0.0]), indexing="ij")
    rho = np.array(
        particles.eval_density(ee1, ee2, ee3, h1=1 / 12, h2=1 / 12, h3=1.0, kernel_type="gaussian_2d", derivative=0)
    ).squeeze()
    rho_exact = (1.5 + np.cos(2 * np.pi * ee1) * np.cos(2 * np.pi * ee2)).squeeze()
    err = np.max(np.abs(rho - rho_exact)) / np.max(np.abs(rho_exact))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, data, title in [
        (axes[0], rho_exact, "exact"),
        (axes[1], rho, f"SPH (gaussian_2d), rel. error = {err:.4f}"),
    ]:
        im = ax.pcolormesh(eta, eta, data.T, cmap="RdBu_r", vmin=0.5, vmax=2.5, shading="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$\eta_1$", fontsize=11)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\eta_2$", fontsize=11)
    plt.tight_layout()
    _save(fig, "sph_density_2d.png")


if __name__ == "__main__":
    lagrange_euler_figure()
    kernels_figure()
    density_1d_figure()
    density_2d_figure()
