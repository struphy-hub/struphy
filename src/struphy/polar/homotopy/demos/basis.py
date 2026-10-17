"""Basis-function figures of the talk (Sections 1, 2 and 5).

* the disk-like mapping ``R = R1 x (1 + eps x cos(2 pi k y))`` (``eps = 0.2``, ``k = 3``),
* tensor-product 0-forms: cubic clamped B-splines on ``N^x = 5`` cells and cubic periodic B-splines on ``N^y = 12``
  cells (or trigonometric functions in ``y``),
* the polar 0-forms for ``r = 1, 2, 3``: the first ``r + 1`` rings are replaced by ``(r + 1)(r + 2)/2`` projected
  polynomials,
* the central polar 0-, 1- and 2-forms for ``r = 3``.

The spaces are those of a 2D Struphy Derham complex on the domain ``PerturbedDiskCylinder`` (see :mod:`.common`).
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos import common
from struphy.polar.homotopy.demos.plotting import DIVERGING, clean_axis, draw_mesh

RS = np.linspace(0, 1, 121)
TS = np.linspace(0, 1, 241)


#: cells of the talk's basis figures: cubic splines on N^x = 5 radial cells and N^y = 12 angular cells (B-splines) or
#: 13 trigonometric functions
CELLS = {"bspline": (5, 12), "trig": (5, 13)}


def _complex(regularity, angular="bspline", domain=None):
    return common.polar_complex(regularity, 3, CELLS[angular], domain=domain, angular=angular)


def _panel(ax, mapping, values, title=None, mesh=None):
    X, Y = mapping(*np.meshgrid(RS, TS, indexing="ij"))
    vmax = np.max(np.abs(values)) or 1.0
    ax.pcolormesh(X, Y, values / vmax, cmap=DIVERGING, vmin=-1, vmax=1, shading="gouraud")
    if mesh is not None:
        draw_mesh(ax, mapping, *mesh, alpha=0.25)
    ax.set_aspect("equal")
    clean_axis(ax)
    if title:
        ax.set_title(title, fontsize=10)


def figure_mapping(mapping=None, cells=(5, 12)):
    """The disk-like coordinate transformation: logical mesh and its image."""
    mapping = common.cross_section() if mapping is None else mapping
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    rb, tb = np.linspace(0, 1, cells[0] + 1), np.linspace(0, 1, cells[1] + 1)
    for r in rb:
        axes[0].axvline(r, color="k", lw=0.6, alpha=0.4)
    for t in tb:
        axes[0].axhline(t, color="k", lw=0.6, alpha=0.4)
    axes[0].axvline(0, color="C3", lw=2.5)
    axes[0].set_xlabel(r"$\hat x$")
    axes[0].set_ylabel(r"$\hat y$")
    axes[0].set_title("logical domain")
    axes[0].set_aspect("equal")
    draw_mesh(axes[1], mapping, rb, tb, alpha=0.4)
    axes[1].plot([0], [0], "o", color="C3")
    axes[1].set_aspect("equal")
    axes[1].set_title(r"$\Phi(\hat x, \hat y) = (R C, R S)$, $\hat x = 0 \mapsto 0$")
    fig.tight_layout()
    return fig


def figure_tensor_basis(angular="bspline", mapping=None, rings=(0, 1, 2, 3), angles=(0, 1, 2, 3)):
    """Physical tensor-product 0-form basis functions ``B_i(x) U_j(y)`` (rows: rings, columns: angular index)."""
    mapping = common.cross_section() if mapping is None else mapping
    space = common.tensor_space(0, 3, CELLS[angular], angular=angular)
    radial, ang = space.logical.x, space.logical.y
    fig, axes = plt.subplots(len(rings), len(angles), figsize=(2.4 * len(angles), 2.4 * len(rings)))
    mesh = (np.linspace(0, 1, radial.cells + 1), np.linspace(0, 1, 13))
    for a, i in enumerate(rings):
        for b, j in enumerate(angles):
            coefficients = np.zeros(space.n)
            coefficients[space.layout.index(i, j, radial.n, ang.n)] = 1.0
            _panel(axes[a, b], mapping, space.evaluate(coefficients, RS, TS), rf"$B_{i} U_{j}$", mesh)
    kind = "cubic periodic B-splines" if angular == "bspline" else "trigonometric functions"
    fig.suptitle(f"tensor-product 0-forms: cubic clamped B-splines ($N^x = 5$) x {kind}")
    fig.tight_layout()
    return fig


def figure_polar_zero_forms(regularity, angular="bspline", mapping=None):
    """The central polar 0-forms of regularity ``r`` (they replace the first ``r + 1`` rings)."""
    mapping = common.cross_section() if mapping is None else mapping
    c = _complex(regularity, angular)
    radial, ang = c.radial, c.angular
    central = c[0].central_elements()
    n = central.shape[1]
    columns = min(n, 5)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.6 * rows), squeeze=False)
    mesh = (np.linspace(0, 1, radial.cells + 1), np.linspace(0, 1, ang.n + 1 if angular == "bspline" else 13))
    for l in range(rows * columns):
        ax = axes.ravel()[l]
        if l >= n:
            ax.axis("off")
            continue
        _panel(
            ax,
            mapping,
            c.spaces[0].evaluate(central[:, l], RS, TS),
            rf"$\bar\Lambda^{{0,{regularity}}}_{{{l + 1}}}$",
            mesh,
        )
    fig.suptitle(f"polar 0-forms, $r = {regularity}$: the first {regularity + 1} rings replaced by {n} functions")
    fig.tight_layout()
    return fig


def figure_all_forms(regularity=3, angular="bspline", mapping=None):
    """Central polar 0-forms, 1-forms (magnitude and arrows) and 2-forms of regularity ``r``."""
    mapping = common.cross_section() if mapping is None else mapping
    c = _complex(regularity, angular)
    counts = c.central_counts()
    columns = max(counts)
    fig, axes = plt.subplots(3, columns, figsize=(1.8 * columns, 6), squeeze=False)
    rs = np.linspace(1e-3, 1, 121)
    X, Y = mapping(*np.meshgrid(rs, TS, indexing="ij"))
    for k in range(3):
        central = c[k].central_elements()
        for l in range(columns):
            ax = axes[k, l]
            if l >= counts[k]:
                ax.axis("off")
                continue
            values = c.spaces[k].evaluate(central[:, l], rs, TS)
            if k == 1:
                magnitude = np.hypot(values[0], values[1])
                ax.pcolormesh(X, Y, magnitude / magnitude.max(), cmap="viridis", shading="gouraud")
                qr, qt = np.linspace(0.08, 0.95, 7), np.linspace(0, 1, 25)[:-1]
                Qr, Qt = np.meshgrid(qr, qt, indexing="ij")
                vx, vy = c.spaces[1].evaluate_points(central[:, l], Qr, Qt)
                ax.quiver(*mapping(Qr, Qt), vx, vy, color="w", scale=None, width=0.006)
            else:
                vmax = np.max(np.abs(values)) or 1.0
                ax.pcolormesh(X, Y, values / vmax, cmap=DIVERGING, vmin=-1, vmax=1, shading="gouraud")
            ax.set_aspect("equal")
            clean_axis(ax)
            ax.set_title(rf"$\bar\Lambda^{{{k},{regularity}}}_{{{l + 1}}}$", fontsize=9)
    fig.suptitle(f"central polar basis functions for 0-, 1- and 2-forms, $r = {regularity}$")
    fig.tight_layout()
    return fig


def figure_discontinuity(mapping=None):
    """A tensor-product 0-form is discontinuous at the origin: ``P(0) = sum_j p_1j U_j(y)`` depends on the angle."""
    mapping = common.cross_section() if mapping is None else mapping
    space = common.tensor_space(0, 3, CELLS["bspline"])
    rng = np.random.default_rng(1)
    coefficients = 1 + 0.3 * rng.standard_normal(space.n)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    values = space.evaluate(coefficients, RS, TS)
    X, Y = mapping(*np.meshgrid(RS, TS, indexing="ij"))
    m = axes[0].pcolormesh(X, Y, values, cmap="viridis", shading="gouraud")
    plt.colorbar(m, ax=axes[0], shrink=0.8)
    axes[0].set_aspect("equal")
    clean_axis(axes[0])
    axes[0].set_title("random tensor-product 0-form")
    axes[1].plot(TS, values[0], label=r"$\lim_{\hat x \to 0} P$")
    axes[1].set_xlabel(r"$\hat y$")
    axes[1].set_title(r"the value at the origin depends on the angle")
    axes[1].legend()
    fig.tight_layout()
    return fig
