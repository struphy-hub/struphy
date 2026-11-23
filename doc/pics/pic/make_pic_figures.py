"""Generate the figures of the PIC documentation page (doc/sections/subsections/numerics-pic.rst).

Adapted from the lecture "Meshless Structure-Preserving Algorithms" (SPEGEO school, Strasbourg 2026,
https://github.com/spossann/2026_SPEGEO_Strasbourg_school), with the notation of the Struphy documentation.

Run from anywhere::

    python doc/pics/pic/make_pic_figures.py

The PNG files are written next to this script.
"""

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150

BLUE_FACE, BLUE_EDGE = "#cfe8f3", "#0f4c81"
ORANGE_FACE, ORANGE_EDGE = "#f8d8a8", "#8a4f00"
ARROW = "#2f2f2f"


def _blob():
    """Boundary of the initial region Omega_0."""
    theta = np.linspace(0, 2 * np.pi, 220)
    r0 = 0.95 + 0.10 * np.sin(3 * theta + 0.2)
    x0 = -1.9 + 1.05 * r0 * np.cos(theta)
    y0 = 0.00 + 0.72 * r0 * np.sin(theta)
    return x0, y0


def _region(ax, x, y, face, edge):
    """Add a closed region with boundary (x, y) to ax and return the patch."""
    verts = np.c_[x, y]
    codes = np.full(len(verts), Path.LINETO)
    codes[0] = Path.MOVETO
    patch = patches.PathPatch(Path(verts, codes), facecolor=face, edgecolor=edge, lw=2.2, alpha=0.95)
    ax.add_patch(patch)
    return patch


def _arrows(ax, x0, y0, x1, y1):
    """Curved arrows from the initial to the transported marker positions."""
    for i, (xa, ya, xb, yb) in enumerate(zip(x0, y0, x1, y1)):
        rad = [0.16, 0.08, -0.08, -0.16][i]
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.8,
                color=ARROW,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=5,
        )


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def flow_map_figure():
    """Markers transported by the flow map Phi_t, together with the phase-space density."""
    fig, ax = plt.subplots(figsize=(8.976, 3.344))

    x0, y0 = _blob()
    _region(ax, x0, y0, BLUE_FACE, BLUE_EDGE)

    def flow_map(x, y):
        return (
            2.30 + 0.92 * (x + 1.9) + 0.72 * y + 0.28 * np.sin(2.6 * y) + 0.08 * np.sin(2.1 * x),
            0.03 + 0.26 * (x + 1.9) + 1.20 * y + 0.20 * np.sin(2.8 * (x + 1.9)),
        )

    x1, y1 = flow_map(x0, y0)
    _region(ax, x1, y1, ORANGE_FACE, ORANGE_EDGE)

    px0 = np.array([-2.28, -1.72, -2.28, -1.72])
    py0 = np.array([0.24, 0.24, -0.24, -0.24])
    px1, py1 = flow_map(px0, py0)

    # reorder by vertical position in each region to avoid visual crossings
    left, right = np.argsort(py0), np.argsort(py1)
    px0, py0 = px0[left], py0[left]
    px1, py1 = px1[right], py1[right]

    ax.scatter(px0, py0, s=42, color=BLUE_EDGE, edgecolors="white", linewidths=0.9, zorder=6)
    ax.scatter(px1, py1, s=42, color=ORANGE_EDGE, edgecolors="white", linewidths=0.9, zorder=6)
    _arrows(ax, px0, py0, px1, py1)

    ax.text(0.0, 0.54, r"$\Phi_t$", fontsize=15, color=ARROW, ha="center")
    ax.text(-2.02, 0.05, r"$\Omega_0$", fontsize=14.5, color=BLUE_EDGE, ha="center")
    ax.text(
        -1.9,
        -0.95,
        r"$f^{\mathrm{vol}}(0,\mathbf{z}_0)$, markers $\mathbf{z}_{p0}$",
        fontsize=13,
        color=BLUE_EDGE,
        ha="center",
    )
    ax.text(3.7, -0.45, r"$\Phi_t(\Omega_0)$", fontsize=14.5, color=ORANGE_EDGE, ha="center")
    ax.text(
        2.9,
        -1.2,
        r"$f^{\mathrm{vol}}(t,\mathbf{z})$, markers $\mathbf{z}_p(t)$",
        fontsize=13,
        color=ORANGE_EDGE,
        ha="center",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 4.6)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    _save(fig, "pic_flow_map.png")


def partition_figure():
    """Partition of Omega_0 into subregions, one marker per subregion, transported by Phi_t."""
    fig, ax = plt.subplots(figsize=(10.2, 2.8))

    x0, y0 = _blob()
    patch0 = _region(ax, x0, y0, BLUE_FACE, BLUE_EDGE)

    # flow map, one-to-one over the blob (Jacobian determinant >= 0.845)
    def flow_map(x, y):
        u = x + 1.9
        return (
            2.30 + 0.88 * u + 0.35 * y + 0.15 * np.sin(0.9 * y),
            0.03 + 0.20 * u + 1.15 * y + 0.18 * np.sin(0.8 * u),
        )

    x1, y1 = flow_map(x0, y0)
    patch1 = _region(ax, x1, y1, ORANGE_FACE, ORANGE_EDGE)

    px0 = np.array([-2.42, -1.62, -2.18, -1.42])
    py0 = np.array([0.38, 0.32, -0.32, -0.28])
    order = np.lexsort((px0, py0))
    px0, py0 = px0[order], py0[order]
    px1, py1 = flow_map(px0, py0)

    # Voronoi-like interfaces between the subregions, clipped to the region
    clip_path = patch0.get_path()
    xmin, xmax = clip_path.vertices[:, 0].min() - 0.18, clip_path.vertices[:, 0].max() + 0.18
    ymin, ymax = clip_path.vertices[:, 1].min() - 0.18, clip_path.vertices[:, 1].max() + 0.18
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 900), np.linspace(ymin, ymax, 700))
    inside = clip_path.contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    d2 = (X[..., None] - px0[None, None, :]) ** 2 + (Y[..., None] - py0[None, None, :]) ** 2
    labels = np.argmin(d2, axis=-1).astype(float)
    labels[~inside] = np.nan
    cs = ax.contour(X, Y, labels, levels=np.arange(0.5, len(px0) - 0.5, 1.0))
    segments = [seg for lev in cs.allsegs for seg in lev if len(seg) > 1]
    cs.remove()

    for seg in segments:
        (line0,) = ax.plot(seg[:, 0], seg[:, 1], color="black", lw=1.1, zorder=4)
        line0.set_clip_path(patch0)
        xr, yr = flow_map(seg[:, 0], seg[:, 1])
        (line1,) = ax.plot(xr, yr, color="black", lw=1.1, zorder=4)
        line1.set_clip_path(patch1)

    ax.scatter(px0, py0, s=42, color=BLUE_EDGE, edgecolors="white", linewidths=0.9, zorder=6)
    ax.scatter(px1, py1, s=42, color=ORANGE_EDGE, edgecolors="white", linewidths=0.9, zorder=6)
    _arrows(ax, px0, py0, px1, py1)

    ax.text(0.0, 0.0, r"$\Phi_t$", fontsize=15, color=ARROW, ha="center")
    ax.text(-3.1, 0.15, r"$\Omega_0 = \bigcup_p \Omega_0^p$", fontsize=14, color=BLUE_EDGE, ha="right")
    ax.text(-3.1, -0.3, r"$f^{\mathrm{vol}}(0,\mathbf{z}_0)$", fontsize=13.5, color=BLUE_EDGE, ha="right")
    ax.text(3.3, -0.2, r"$\Phi_t(\Omega_0)$", fontsize=14.5, color=ORANGE_EDGE, ha="center")
    ax.text(3.2, -0.5, r"$f^{\mathrm{vol}}(t,\mathbf{z})$", fontsize=13.5, color=ORANGE_EDGE, ha="center")

    ax.set_aspect("equal")
    ax.set_xlim(-4.3, 4.2)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    _save(fig, "pic_partition.png")


def kernel_density_figure(with_boxes):
    """Kernel density estimate at z*: only markers within the support of W_h contribute.
    With boxes, the box of z* and its neighbours are highlighted (box size >= h)."""
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(11.28, 3.36))

    z_star = np.array([0.0, 0.0])
    h = 0.5
    xmin, xmax = -2.6, 2.6
    ymin, ymax = -1.75, 1.55
    points = np.column_stack([rng.uniform(xmin, xmax, 150), rng.uniform(ymin, ymax, 150)])

    dist = np.linalg.norm(points - z_star, axis=1)
    inside = points[dist < h]
    outside = points[dist >= h]

    for radius, alpha in [(0.50, 0.06), (0.40, 0.08), (0.30, 0.10), (0.20, 0.12)]:
        ax.add_patch(Circle(z_star, radius=radius, facecolor="#5dade2", edgecolor="none", alpha=alpha, zorder=0))
    ax.add_patch(Circle(z_star, radius=h, facecolor="#5dade2", edgecolor="#1f5f8b", lw=2.6, alpha=0.16, zorder=1))
    ax.add_patch(Circle(z_star, radius=h, facecolor="none", edgecolor="#1f5f8b", lw=2.6, zorder=3))

    size = 60
    ax.scatter(outside[:, 0], outside[:, 1], s=size, color="#9aa3ad", edgecolor="white", linewidth=1.0, zorder=2)
    ax.scatter(inside[:, 0], inside[:, 1], s=size, color="#d95f02", edgecolor="white", linewidth=1.1, zorder=4)
    for px, py in inside:
        ax.plot([px, z_star[0]], [py, z_star[1]], color="#d95f02", lw=1.1, alpha=0.32, zorder=2)

    ax.scatter([z_star[0]], [z_star[1]], s=size, color="#1f5f8b", edgecolor="white", linewidth=1.1, zorder=5)
    ax.text(0.1, 0.0, r"$\mathbf{z}^*$", color="#1f5f8b", fontsize=15, weight="bold")
    ax.annotate(
        "",
        xy=(h / np.sqrt(2), -h / np.sqrt(2)),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="<->", color="#1f5f8b", lw=1.8),
    )
    ax.text(0.43 * h, -0.28 * h, r"$h$", fontsize=13, color="#1f5f8b", ha="center")
    ax.text(0.16, 0.58, r"$w_p\,W_h$", color="#d95f02", fontsize=12)
    ax.text(-1.63, 0.96, "ignored", color="#6b7280", fontsize=11)

    if with_boxes:
        x_edges = np.arange(xmin, xmax + h, h)
        y_edges = np.arange(ymin, ymax + h, h)
        ix = int(np.searchsorted(x_edges, z_star[0], side="right")) - 1
        iy = int(np.searchsorted(y_edges, z_star[1], side="right")) - 1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ci, cj = ix + di, iy + dj
                if 0 <= ci < len(x_edges) - 1 and 0 <= cj < len(y_edges) - 1:
                    color = "#f39c12" if (di == 0 and dj == 0) else "#f7dc6f"
                    ax.add_patch(
                        Rectangle(
                            (x_edges[ci], y_edges[cj]), h, h, facecolor=color, edgecolor="none", alpha=0.45, zorder=1
                        )
                    )
        for x in x_edges:
            ax.plot([x, x], [ymin, ymax], color="#6b7280", lw=0.7, alpha=0.45, zorder=6)
        for y in y_edges:
            ax.plot([xmin, xmax], [y, y], color="#6b7280", lw=0.7, alpha=0.45, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    _save(fig, "pic_kernel_boxes.png" if with_boxes else "pic_kernel_density.png")


if __name__ == "__main__":
    flow_map_figure()
    partition_figure()
    kernel_density_figure(with_boxes=False)
    kernel_density_figure(with_boxes=True)
