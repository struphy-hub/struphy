"""Smooth polar splines at the centre of a disk (port of PolarFEEC.jl ``examples/04_polar_regularity.jl``).

A disk-like mapping collapses the edge ``x = 0`` of the logical square onto the centre, so tensor-product splines are
in general not smooth there. A polar complex of regularity ``r`` replaces the ``r + 1`` innermost rings by projected
polynomials of degree ``<= r``, so its 0-forms are ``C^r`` at the centre. A function is ``C^k`` at the centre exactly
when, for each ``j <= k``, its ``j``-th derivative along the ray at angle ``alpha`` contains only the angular harmonics
``j, j - 2, ...``. The size ``delta_j`` of the remaining harmonics is measured by polynomial fits along 48 rays.

The defaults are those of the Julia example: the star-shaped disk ``R = x (1 + 0.2 cos(6 pi y))``, B-splines of
degree 5 on 8 cells in ``x`` and 24 Fourier modes in ``y``, regularities ``r = 0, ..., 3``.
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy import ClampedBSplineSpace, DiskMapping, PeriodicBSplineSpace, PolarComplex2D
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, clean_axis
from struphy.polar.homotopy.forms2d import PhysicalFormSpace2D
from struphy.polar.homotopy.spaces1d import TrigonometricSpace


def star_disk():
    """The star-shaped disk of the Julia example (radius linear in ``x``)."""
    return DiskMapping(radius=lambda r, t: r * (1 + 0.2 * np.cos(6 * np.pi * t)))


def f(x, y):
    return np.sin(3 * x + 1) * np.cos(2 * y - 0.5) + 0.5 * np.cos(4 * x * y)


def grad_f(x, y):
    return (
        3 * np.cos(3 * x + 1) * np.cos(2 * y - 0.5) - 2 * y * np.sin(4 * x * y),
        -2 * np.sin(3 * x + 1) * np.sin(2 * y - 0.5) - 2 * x * np.sin(4 * x * y),
    )


def rays(n=48):
    return 2 * np.pi * (np.arange(1, n + 1) - 0.5) / n


def ray_derivatives(u, alpha, a=0.06, degree=10, samples=40):
    """One-sided derivatives ``d^k u(0+)``, ``k = 0..3``, along the ray at angle ``alpha`` (least-squares fit)."""
    s = a * (1 - np.cos(np.pi * np.arange(1, samples + 1) / (2 * samples)))
    V = (s[:, None] / a) ** np.arange(degree + 1)[None, :]
    c = np.linalg.lstsq(V, u(s * np.cos(alpha), s * np.sin(alpha)), rcond=None)[0]
    return np.array([_factorial(k) * c[k] / a**k for k in range(4)])


def _factorial(k):
    out = 1
    for i in range(2, k + 1):
        out *= i
    return out


def nonpolynomial_part(d, alpha, k):
    """Max norm of the part of ``d(alpha)`` outside the harmonics ``k, k - 2, ...`` (least squares)."""
    columns = []
    for m in range(k, -1, -2):
        columns.append(np.cos(m * alpha))
        if m > 0:
            columns.append(np.sin(m * alpha))
    H = np.column_stack(columns)
    residual = d - H @ np.linalg.lstsq(H, d, rcond=None)[0]
    return np.max(np.abs(residual))


def smoothness_defects(u):
    """Relative defects ``delta_k`` of ``d^k u(0+; alpha)`` over 48 rays, ``k = 1, 2, 3``."""
    alpha = rays()
    D = np.column_stack([ray_derivatives(u, a) for a in alpha])
    return np.array([nonpolynomial_part(D[k], alpha, k) / np.max(np.abs(D[k])) for k in range(1, 4)])


def build(radial=None, angular=None, regularities=(0, 1, 2, 3), mapping=None):
    """Tensor-product 0-form space and the polar complexes of the study."""
    mapping = star_disk() if mapping is None else mapping
    radial = ClampedBSplineSpace(5, 8) if radial is None else radial
    angular = TrigonometricSpace(24) if angular is None else angular
    tensor = PhysicalFormSpace2D.from_factors(0, radial, angular, mapping)
    polar = {r: PolarComplex2D(radial, angular, mapping, regularity=r) for r in regularities}
    return mapping, tensor, polar


def study(radial=None, angular=None, regularities=(0, 1, 2, 3), verbose=True):
    """Dimensions, cohomology and the defects of the commuting projections of ``f``; returns a dict of results."""
    mapping, tensor, polar = build(radial, angular, regularities)
    lines = ["  r   dim V0  dim V1  dim V2   central 0/1/2   cohomology"]
    for r, c in polar.items():
        dims = tuple(c.dimension(k) for k in range(3))
        lines.append(
            f"  {r}   {dims[0]:6d}  {dims[1]:6d}  {dims[2]:6d}     {c.central_counts()}   {c.cohomology_dimensions()}"
        )
    spaces = [("tensor product", tensor, tensor.commuting_projection(f))] + [
        (f"r = {r}", c.spaces[0], c[0](f)) for r, c in polar.items()
    ]

    def physical(space, coefficients):
        return lambda X, Y: space.evaluate_physical(coefficients, X, Y)

    defects = {name: smoothness_defects(physical(space, u)) for name, space, u in spaces}
    errors = {name: space.l2_error(u, f) for name, space, u in spaces}
    lines.append("\nrelative defects delta_k of d^k u(0+; alpha) (C^k at the centre <=> delta_1 = ... = delta_k = 0)")
    lines.append(f"  {'exact f':15s} " + " ".join(f"{d:9.1e}" for d in smoothness_defects(f)))
    for name in defects:
        lines.append(f"  {name:15s} " + " ".join(f"{d:9.1e}" for d in defects[name]) + f"  {errors[name]:9.1e}")
    if verbose:
        print("\n".join(lines))
    return dict(mapping=mapping, tensor=tensor, polar=polar, spaces=spaces, defects=defects, errors=errors)


STENCILS = (([0], [1.0]), ([-1, 1], [-0.5, 0.5]), ([-1, 0, 1], [1.0, -2.0, 1.0]))


def ring_defects(space1, gradient_coefficients, rho):
    """Non-polynomial part of ``d_s^k (u_h - f)`` on the circle ``|x| = rho`` (relative to ``d_s^k f``), ``k = 1..3``."""
    alpha = rays()
    h = rho / 5
    e = np.stack([np.cos(alpha), np.sin(alpha)])

    def radial_derivative(gradient, k):
        total = 0.0
        for j, w in zip(*STENCILS[k - 1]):
            X, Y = (rho + j * h) * e[0], (rho + j * h) * e[1]
            gx, gy = gradient(X, Y)
            total = total + w * (gx * e[0] + gy * e[1])
        return total / h ** (k - 1)

    def grad_uh(X, Y):
        return space1.evaluate_physical(gradient_coefficients, X, Y)

    out = []
    for k in (1, 2, 3):
        dh, df = radial_derivative(grad_uh, k), radial_derivative(grad_f, k)
        out.append(nonpolynomial_part(dh - df, alpha, k) / np.max(np.abs(df)))
    return np.array(out)


def figure_central_basis(regularity=3, results=None):
    """The central 0-forms of the regularity-3 complex (ordered like the Zernike polynomials)."""
    mapping = star_disk() if results is None else results["mapping"]
    c = results["polar"][regularity] if results is not None else build(regularities=(regularity,))[2][regularity]
    space = c.spaces[0]
    central = c[0].central_elements()
    ts = np.linspace(0, 1, 241)
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(regularity + 1, 2 * (regularity + 1) + 2, width_ratios=[4] * 2 + [1] * (2 * regularity + 2))

    def panel(ax, column, rs):
        values = space.evaluate(central[:, column], rs, ts)
        values = values / np.max(np.abs(values))
        X, Y = np.meshgrid(rs, ts, indexing="ij")
        ax.pcolormesh(*mapping(X, Y), values, cmap=DIVERGING, vmin=-1, vmax=1, shading="gouraud")
        ax.plot(*mapping(np.full_like(ts, 0.5), ts), "k--", linewidth=0.8, alpha=0.6)
        ax.set_aspect("equal")
        clean_axis(ax)

    ax = fig.add_subplot(grid[:, :2])
    panel(ax, 0, np.linspace(0, 1, 161))
    for r in np.linspace(0, 1, c.radial.cells + 1)[1:]:
        ax.plot(*mapping(np.full_like(ts, r), ts), "k", linewidth=0.6 if r < 1 else 1.2, alpha=0.25 if r < 1 else 1)
    ax.set_title("$n = 0$: the central function of $T_1 = 1$")
    index = 1
    for n in range(1, regularity + 1):
        for j in range(n + 1):
            start = 2 + (regularity - n) + 2 * j
            axn = fig.add_subplot(grid[n - 1 if n > 0 else 0, start : start + 2])
            panel(axn, index, np.linspace(0, 0.5, 81))
            axn.set_title(f"$n={n}$, $m={2 * j - n}$", fontsize=10)
            index += 1
    fig.suptitle(rf"central 0-forms of $\bar V^{{0,{regularity}}}$ (zoom to $\hat x < 1/2$)")
    return fig


def figure_regularity(results, radii=None):
    """``d_s^k u_h`` is continuous at the centre (defect -> 0) exactly when ``k <= r``."""
    radii = np.logspace(-4, -1.1, 30) if radii is None else radii
    curves = {}
    for name, space, u in results["spaces"]:
        polar = results["polar"]
        space1 = (
            polar[int(name.split("=")[1])].spaces[1]
            if name.startswith("r")
            else PhysicalFormSpace2D(space.logical.with_degree(1), space.mapping)
        )
        gradient = space.logical.derivative_matrix @ u
        curves[name] = np.column_stack([ring_defects(space1, gradient, rho) for rho in radii])
    styles = [
        (CATEGORICAL[0], "-", 4),
        ("k", "--", 1.5),
        (CATEGORICAL[1], "-", 2),
        (CATEGORICAL[2], "-", 2),
        (CATEGORICAL[5], "-", 2),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.7]))
    for k in range(3):
        ax = axes[k]
        for (name, curve), (color, linestyle, width) in zip(curves.items(), styles):
            ax.loglog(radii, curve[k], color=color, linestyle=linestyle, linewidth=width, label=name)
        ax.set_ylim(1e-9, 1e-2)
        ax.set_xlabel(r"$\rho$")
        ax.set_title(f"$k = {k + 1}$")
        if k == 0:
            ax.set_ylabel(r"defect of $\partial_s^k (u_h - f)$ on $|x| = \rho$")
        if k == 2:
            ax.legend(loc="lower right", fontsize=9)
    names = list(results["defects"])
    delta = np.array([results["defects"][n] for n in names])
    ax = axes[3]
    ax.imshow(delta < 1e-6, cmap="Reds_r", alpha=0.25, aspect="auto")
    for i in range(delta.shape[0]):
        for k in range(3):
            ax.text(k, i, f"{delta[i, k]:.0e}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), [f"{k}" for k in (1, 2, 3)])
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("$k$")
    ax.set_title(r"$\delta_k$ at $\rho = 0$")
    fig.suptitle(r"$\partial_s^k u_h$ is continuous at the centre (defect $\to 0$) exactly when $k \leq \mathfrak{r}$")
    fig.tight_layout()
    return fig


def bspline_angular_defects(regularity=3):
    """The defects with a B-spline angular factor: regularity up to the spline approximation of the polynomials."""
    mapping = star_disk()
    c = PolarComplex2D(ClampedBSplineSpace(5, 8), PeriodicBSplineSpace(5, 24), mapping, regularity=regularity)
    u = c[0](f)
    return smoothness_defects(lambda X, Y: c.spaces[0].evaluate_physical(u, X, Y))
