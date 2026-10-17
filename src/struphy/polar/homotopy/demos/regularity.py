"""Smooth polar splines at the centre of a disk (port of PolarFEEC.jl ``examples/04_polar_regularity.jl``).

A disk-like mapping collapses the edge ``x = 0`` of the logical square onto the centre, so tensor-product splines are
in general not smooth there. A polar complex of regularity ``r`` replaces the ``r + 1`` innermost rings by projected
polynomials of degree ``<= r``, so its 0-forms are ``C^r`` at the centre. A function is ``C^k`` at the centre exactly
when, for each ``j <= k``, its ``j``-th derivative along the ray at angle ``alpha`` contains only the angular harmonics
``j, j - 2, ...``. The size ``delta_j`` of the remaining harmonics is measured by polynomial fits along 48 rays.

The defaults are those of the Julia example: the star-shaped disk ``R = x (1 + 0.2 cos(6 pi y))`` (the Struphy
domain ``PerturbedDiskCylinder(power=0)``), B-splines of degree 5 on 8 cells in ``x`` and 24 Fourier modes in ``y``
(``angular="trig"``; ``angular="bspline"``: 24 periodic B-spline cells), regularities ``r = 0, ..., 3``. The spaces are
those of a 2D Struphy Derham complex (see :mod:`.common`).
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos import common
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, clean_axis
from struphy.polar.homotopy.forms2d import PhysicalFormSpace2D

#: the default angular space; the talk uses 24 Fourier modes (``"trig"``)
ANGULAR = "trig"


def angular_cells(angular):
    """24 periodic B-spline cells, or 24 Fourier modes (49 trigonometric functions)."""
    return 24 if angular == "bspline" else 49


def star_disk():
    """The cross-section of the star-shaped disk of the Julia example (radius linear in ``x``)."""
    return common.cross_section(common.star_disk())


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


def _inverse_series(x, order):
    """Taylor coefficients of ``eta(s)`` for the polynomial ``s = x(eta)`` with ``x[0] = 0``, up to ``s^order``."""
    from numpy.polynomial import polynomial as P

    identity = np.zeros(order + 1)
    identity[1] = 1.0
    eta = identity / x[1]
    for _ in range(order + 1):
        eta = eta + (identity - _compose(x, eta, order)) / x[1]
    return eta


def _compose(u, eta, order):
    """Taylor coefficients of ``u(eta(s))`` up to ``s^order`` (``u``, ``eta`` coefficient arrays, ``eta[0] = 0``)."""
    from numpy.polynomial import polynomial as P

    out, power = np.zeros(order + 1), np.zeros(order + 1)
    power[0] = 1.0
    for c in u:
        out[: power.size] += c * power[: order + 1]
        power = P.polymul(power, eta)[: order + 1]
    return out


def line_derivative_jumps(field, domain, num_cells, degree, eta3=0.0, orders=5):
    """Jumps ``|d^k u/dx^k(0+) - d^k u/dx^k(0-)|``, ``k = 0, ..., orders - 1``, of a Struphy 0-form across the pole
    along the line through the rays ``eta2 = 0`` and ``eta2 = 1/2``.

    On the first radial cell the field and the mapping are polynomials of degree ``degree`` in ``eta1`` along each
    ray, so the one-sided derivatives in the distance ``x`` to the pole are exact (polynomial fit in ``eta1`` and
    series inversion of ``x(eta1)``)."""
    import math

    import cunumpy as xp
    from numpy.polynomial import polynomial as P

    e1 = np.linspace(0, 1 / num_cells, 3 * degree + 3)
    pole = np.array([xp.to_numpy(c) for c in domain(xp.zeros(1), xp.zeros(1), xp.asarray([eta3]))]).reshape(3)
    derivatives = []
    for e2 in (0.0, 0.5):
        etas = (xp.asarray(e1), xp.asarray([e2]), xp.asarray([eta3]))
        points = np.array([xp.to_numpy(c).ravel() for c in domain(*etas)])
        distance = np.hypot(points[0] - pole[0], points[1] - pole[1]) * (1 if e2 == 0.0 else -1)
        u = P.polyfit(e1, xp.to_numpy(field(*etas)).ravel(), degree)
        x = P.polyfit(e1, distance, degree)
        x[0] = 0.0
        series = _compose(u, _inverse_series(x, degree), degree)
        derivatives.append([math.factorial(k) * series[k] for k in range(orders)])
    return np.abs(np.subtract(*derivatives))


def field_smoothness_defects(field, domain, eta3=0.0, a=0.06, degree=10, samples=40):
    """The defects ``delta_k`` of a Struphy 0-form ``field`` (a :class:`~struphy.feec.psydac_derham.SplineFunction`)
    on a domain whose cross-section is parametrized by the polar angle ``2 pi eta2`` (e.g.
    ``PerturbedDiskCylinder``, ``HollowCylinder(a1=0)``): the field is sampled along each ray ``eta2 = alpha / 2 pi``
    and fitted by a polynomial in the distance ``s`` to the pole."""
    import cunumpy as xp

    alpha = rays()
    t = 1 - np.cos(np.pi * np.arange(1, 4 * samples + 1) / (8 * samples))
    pole = np.array([xp.to_numpy(c) for c in domain(xp.zeros(1), xp.zeros(1), xp.asarray([eta3]))]).reshape(3)
    D = []
    for angle in alpha:
        e1, e2, e3 = xp.asarray(t), xp.asarray([angle / (2 * np.pi)]), xp.asarray([eta3])
        points = np.array([xp.to_numpy(c).ravel() for c in domain(e1, e2, e3)])
        s = np.hypot(points[0] - pole[0], points[1] - pole[1])
        assert np.allclose(np.arctan2(points[1] - pole[1], points[0] - pole[0]) % (2 * np.pi), angle % (2 * np.pi)), (
            "the domain is not parametrized by the polar angle 2 pi eta2"
        )
        values = xp.to_numpy(field(e1, e2, e3)).ravel()
        inside = s <= a
        V = (s[inside, None] / a) ** np.arange(degree + 1)[None, :]
        c = np.linalg.lstsq(V, values[inside], rcond=None)[0]
        D.append([_factorial(k) * c[k] / a**k for k in range(4)])
    D = np.array(D).T
    return np.array([nonpolynomial_part(D[k], alpha, k) / np.max(np.abs(D[k])) for k in range(1, 4)])


def build(angular=None, regularities=(0, 1, 2, 3), degree=5, radial_cells=8):
    """Tensor-product 0-form space and the polar complexes of the study (Struphy Derham spaces on the star disk)."""
    angular = ANGULAR if angular is None else angular
    domain = common.star_disk()
    cells = (radial_cells, angular_cells(angular))
    tensor = common.tensor_space(0, degree, cells, domain=domain, angular=angular)
    polar = {
        r: common.polar_complex(r, degree, cells, domain=domain, angular=angular, check_assumptions=False)
        for r in regularities
    }
    return tensor.mapping, tensor, polar


def study(angular=None, regularities=(0, 1, 2, 3), verbose=True, **kwargs):
    """Dimensions, cohomology and the defects of the commuting projections of ``f``; returns a dict of results."""
    mapping, tensor, polar = build(angular, regularities, **kwargs)
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
    c = common.polar_complex(regularity, 5, (8, 24), domain=common.star_disk(), check_assumptions=False)
    u = c[0](f)
    return smoothness_defects(lambda X, Y: c.spaces[0].evaluate_physical(u, X, Y))


# ---------------------------------------------------------------------------------------------------- line figures
def line_derivatives(complex_, form, coefficients, k_max, cells=2, samples=24, fit_degree=10):
    """``d^k u / dx^k`` (``k <= k_max``) of the 0-form (``form=0``) or the 2-form density (``form=2``) along the line
    ``y = 0`` through the pole (the rays ``eta2 = 0`` and ``1/2``), on the first ``cells`` radial cells each side.

    On each radial cell the field is fitted by a polynomial in ``x`` (exact for 0-forms on the star disk, where ``x``
    is linear in ``eta1`` along these rays). Returns ``x``, the derivative curves ``(k_max + 1, len(x))`` and the
    one-sided limits ``(k_max + 1, 2)`` at the pole."""
    space = complex_.spaces[form]
    mapping = space.mapping
    breaks = np.linspace(0, 1, complex_.radial.cells + 1)[: cells + 1]
    xs, curves, limits = [], [], np.zeros((k_max + 1, 2))
    for side, eta2 in ((0, 0.5), (1, 0.0)):
        pieces_x, pieces_d = [], []
        for c in range(cells):
            t = breaks[c] + (breaks[c + 1] - breaks[c]) * (1 - np.cos(np.pi * (np.arange(samples) + 0.5) / samples)) / 2
            X, Y = mapping(t, np.full_like(t, eta2))
            x = X if side else X  # signed distance along y = 0 (X < 0 on the ray eta2 = 1/2)
            u = np.asarray(space.evaluate_points(coefficients, t, np.full_like(t, eta2)))
            fit = np.polynomial.Polynomial.fit(x, u, min(fit_degree, samples - 1))
            grid = np.linspace(x.min(), x.max(), 60)
            pieces_x.append(grid)
            pieces_d.append(np.array([fit.deriv(k)(grid) if k else fit(grid) for k in range(k_max + 1)]))
            if c == 0:
                x0 = 0.0
                limits[:, side] = [fit.deriv(k)(x0) if k else fit(x0) for k in range(k_max + 1)]
        order = np.argsort(np.concatenate(pieces_x))
        xs.append(np.concatenate(pieces_x)[order])
        curves.append(np.concatenate(pieces_d, axis=1)[:, order])
    x = np.concatenate([xs[0], [np.nan], xs[1]])
    curve = np.concatenate([curves[0], np.full((k_max + 1, 1), np.nan), curves[1]], axis=1)
    return x, curve, limits


def random_polar_element(complex_, form, seed=3):
    """Full coefficients of a random element of the polar space of degree ``form``."""
    rng = np.random.default_rng(seed)
    E = complex_.extraction(form)
    return E @ rng.standard_normal(E.shape[1])


def figure_regularity_lines(form=0, regularities=None, orders=None, angular=None, degree=5, radial_cells=8, seed=3):
    """The talk's regularity figures: ``d^k`` of a random polar ``form``-form (0-forms: ``r = 0..3``, ``k = 1..4``;
    2-form densities: ``r = 1..4``, ``k = 0..3``) along a line through the pole; grey panels jump at the pole."""
    angular = ANGULAR if angular is None else angular
    regularities = (0, 1, 2, 3) if form == 0 else (1, 2, 3, 4) if regularities is None else regularities
    regularities = regularities if regularities is not None else (0, 1, 2, 3)
    orders = (1, 2, 3, 4) if form == 0 else (0, 1, 2, 3) if orders is None else orders
    cells = (radial_cells, angular_cells(angular))
    smooth_up_to = (lambda r: r) if form == 0 else (lambda r: r - 2)
    fig, axes = plt.subplots(len(orders), len(regularities), figsize=(2.3 * len(regularities), 1.6 * len(orders)))
    for col, r in enumerate(regularities):
        c = common.polar_complex(r, degree, cells, domain=common.star_disk(), angular=angular, check_assumptions=False)
        x, curves, limits = line_derivatives(c, form, random_polar_element(c, form, seed), max(orders))
        color = CATEGORICAL[col % len(CATEGORICAL)]
        for row, k in enumerate(orders):
            ax = axes[row, col]
            jump = k > smooth_up_to(r)
            ax.set_facecolor("0.94" if jump else "white")
            ax.plot(x, curves[k], color=color, linewidth=1.6)
            ax.plot([0, 0], limits[k], "o", color=color, markersize=4)
            ax.axvline(0, color="0.7", linewidth=0.8)
            for b in np.linspace(0, 1, radial_cells + 1)[1:3]:
                for sign_x in (np.nanmax(x) / (2 / radial_cells), np.nanmin(x) / (2 / radial_cells)):
                    ax.axvline(b * sign_x, color="0.7", linestyle=":", linewidth=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$r = {r}$", color=color, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"$k = {k}$", rotation=0, labelpad=22, fontsize=12)
            if row == len(orders) - 1:
                ax.set_xlabel("$x = 0$")
    name = "0-form $u$" if form == 0 else "2-form density"
    fig.suptitle(rf"$d^k/dx^k$ of a random polar {name} along $y = 0$ (grey: discontinuous at the pole)", fontsize=11)
    fig.tight_layout()
    return fig
