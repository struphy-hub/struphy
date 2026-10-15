"""Poisson and curl-curl problems on a rounded square with the polar complex (port of PolarFEEC.jl
``examples/05_polar_pde.jl``).

A disk-like mapping maps the logical rectangle onto the rounded square ``{g > 0}``,
``g = 1 - x^2 - y^2 - eps Re((x + i y)^4)``. The solvers work in the homotopy gauge of the complex::

    stiffness_solver(complex, 0)(f)   (grad v, grad u) = (v, f)         for all v in V0
    stiffness_solver(complex, 1)(J)   (curl Phi, curl A) = (Phi, J)     for all Phi in V1

For the free complex the kernel of the gradient holds the constants, so ``f`` needs zero mean and the gauge fixes
``u(centre) = 0``. For curl-curl it holds the gradients; with ``J = curl* B = (d_y B, -d_x B)`` the free problem needs
``B = 0`` on the boundary, the Dirichlet problem (``A . t = 0``) needs ``B`` to have zero mean. The curl-curl solve is
the "free two-dimensional lunch" of the paper: one polar 2-form mass solve and the homotopy ``h_2``.

The manufactured solutions are built from ``g``: ``u = g s`` vanishes on the boundary, ``u = g^2 s`` has
``d_n u = 0``. ``B = sin 2x cos y`` is odd in ``x`` (zero mean) and ``g B`` vanishes on the boundary. The expected
rates are ``p + 1`` for ``u`` and ``p`` for ``curl A`` (regularity ``r = p``).
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy import ClampedBSplineSpace, DiskMapping, PeriodicBSplineSpace, PolarComplex2D
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, draw_mesh, field_on_disk
from struphy.polar.homotopy.solvers import stiffness_solver

EPS = 0.15
QUADRATURE = 8


def boundary_radius(t):
    """On the ray at angle ``2 pi t`` the boundary radius solves ``rho^2 + eps cos(8 pi t) rho^4 = 1``."""
    return np.sqrt(2 / (1 + np.sqrt(1 + 4 * EPS * np.cos(8 * np.pi * t))))


def rounded_square():
    return DiskMapping(radius=lambda r, t: r * boundary_radius(t))


# ---------------------------------------------------------------- manufactured solutions as (value, gradient, laplacian)
class Jet:
    def __init__(self, value, gradient, laplacian):
        self.value, self.gradient, self.laplacian = value, gradient, laplacian

    def __mul__(self, other):
        return Jet(
            self.value * other.value,
            (
                self.value * other.gradient[0] + other.value * self.gradient[0],
                self.value * other.gradient[1] + other.value * self.gradient[1],
            ),
            self.value * other.laplacian
            + other.value * self.laplacian
            + 2 * (self.gradient[0] * other.gradient[0] + self.gradient[1] * other.gradient[1]),
        )


def level(x, y):
    return Jet(
        1 - x**2 - y**2 - EPS * (x**4 - 6 * x**2 * y**2 + y**4),
        (-2 * x - 4 * EPS * (x**3 - 3 * x * y**2), -2 * y - 4 * EPS * (y**3 - 3 * x**2 * y)),
        -4.0 * np.ones_like(x),
    )


def shape(x, y):
    return Jet(1 + np.sin(2 * x + y), (2 * np.cos(2 * x + y), np.cos(2 * x + y)), -5 * np.sin(2 * x + y))


def u_dirichlet(x, y):
    return level(x, y) * shape(x, y)


def u_free(x, y):
    return level(x, y) * level(x, y) * shape(x, y)


def field_curl(x, y):
    return Jet(
        np.sin(2 * x) * np.cos(y),
        (2 * np.cos(2 * x) * np.cos(y), -np.sin(2 * x) * np.sin(y)),
        -5 * np.sin(2 * x) * np.cos(y),
    )


def curl_dirichlet(x, y):
    return field_curl(x, y)


def curl_free(x, y):
    return level(x, y) * field_curl(x, y)


def source(u):
    return lambda x, y: -u(x, y).laplacian


def current(B):
    return lambda x, y: (B(x, y).gradient[1], -B(x, y).gradient[0])


def exact(u, shift=0.0):
    return lambda x, y: u(x, y).value - shift


EXACT = dict(
    u_free=exact(u_free, float(u_free(np.array(0.0), np.array(0.0)).value)),
    u_dirichlet=exact(u_dirichlet),
    B_free=exact(curl_free),
    B_dirichlet=exact(curl_dirichlet),
)


def solve_all(p, n, mapping=None):
    """Free and Dirichlet complexes of degree p with n radial and 6n angular cells, and the four solutions."""
    mapping = rounded_square() if mapping is None else mapping
    radial, angular = ClampedBSplineSpace(p, n), PeriodicBSplineSpace(p, 6 * n)
    free = PolarComplex2D(radial, angular, mapping, regularity=p)
    dirichlet = PolarComplex2D(radial, angular, mapping, regularity=p, boundary="dirichlet")
    q = QUADRATURE
    return dict(
        free=free,
        dirichlet=dirichlet,
        u_free=stiffness_solver(free, 0)(source(u_free), additional_order=q),
        u_dirichlet=stiffness_solver(dirichlet, 0)(source(u_dirichlet), additional_order=q),
        A_free=stiffness_solver(free, 1)(current(curl_free), additional_order=q),
        A_dirichlet=stiffness_solver(dirichlet, 1)(current(curl_dirichlet), additional_order=q),
    )


def errors(s):
    """L2 errors of u (free, Dirichlet) and of curl A (free, Dirichlet)."""
    free, q = s["free"], QUADRATURE
    V0, V1, V2 = free.spaces
    C = V1.derivative_matrix
    return np.array(
        [
            V0.l2_error(s["u_free"], EXACT["u_free"], q),
            V0.l2_error(s["u_dirichlet"], EXACT["u_dirichlet"], q),
            V2.l2_error(C @ s["A_free"], EXACT["B_free"], q),
            V2.l2_error(C @ s["A_dirichlet"], EXACT["B_dirichlet"], q),
        ]
    )


def checks(s):
    """Compatibility, gauge and boundary checks of the example (returns a dict)."""
    free, dirichlet = s["free"], s["dirichlet"]
    load = free.spaces[0].inner_product(source(u_free), QUADRATURE)
    ts = np.linspace(0, 1, 721)
    V0, V1 = free.spaces[0], free.spaces[1]
    centre = V0.evaluate(s["u_free"], [0.0], [0.0])[0, 0]
    boundary_u = np.max(np.abs(V0.evaluate(s["u_dirichlet"], [1.0], ts)))
    A_polar = dirichlet.duals(1).T @ s["A_dirichlet"]
    A_free_polar = free.duals(1).T @ s["A_free"]
    logical = V1.logical.evaluate(s["A_dirichlet"], [1.0], ts)
    return dict(
        mean_of_source=float(np.sum(load)),
        u_centre=float(centre),
        max_u_on_boundary=float(boundary_u),
        gauge_free=float(np.max(np.abs(free.homotopy(1) @ A_free_polar))),
        gauge_dirichlet=float(np.max(np.abs(dirichlet.homotopy(1) @ A_polar))),
        tangential_trace=float(np.max(np.abs(logical[1]))),
    )


CELLS = (4, 6, 8, 12, 16, 24)
DEGREES = (2, 3, 4, 5)


def convergence(cells=CELLS, degrees=DEGREES, verbose=True):
    table = {p: np.array([errors(solve_all(p, n)) for n in cells]) for p in degrees}
    if verbose:
        rates = lambda e: np.log(e[:-1] / e[1:]) / np.log(np.array(cells[1:]) / np.array(cells[:-1]))  # noqa: E731
        print("Observed L2 rates over the last three refinements (expected p + 1 for u, p for curl A)")
        print("   p   u free           u dirichlet      curl A free      curl A dirichlet")
        for p in degrees:
            parts = ["  ".join(f"{r:4.1f}" for r in rates(table[p][:, j])[-3:]) for j in range(4)]
            print(f"   {p}   " + "   ".join(parts))
    return table


def _sample(space, coefficients, rs, ts):
    return space.evaluate(coefficients, rs, ts)


def figure_poisson(s, n=6):
    mapping = s["free"].mapping
    rs, ts = np.linspace(1e-3, 1, 121), np.linspace(0, 1, 361)
    V0 = s["free"].spaces[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, key, title, symmetric in (
        (axes[0], "u_free", r"$-\Delta u_h = f,\ \partial_n u_h = 0$ (gauge $u_h(0) = 0$)", True),
        (axes[1], "u_dirichlet", r"$-\Delta u_h = f,\ u_h = 0$ on $\partial\Omega$", False),
    ):
        field_on_disk(
            ax,
            mapping,
            rs,
            ts,
            _sample(V0, s[key], rs, ts),
            symmetric=symmetric,
            contours=12,
            colorbar=True,
            label=r"$u_h$",
        )
        draw_mesh(ax, mapping, np.linspace(0, 1, n + 1), np.linspace(0, 1, 6 * n + 1))
        ax.set_title(title)
    fig.tight_layout()
    return fig


def figure_curl_curl(s, n=6):
    mapping = s["free"].mapping
    V1, V2 = s["free"].spaces[1], s["free"].spaces[2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    ax = axes[0]
    draw_mesh(ax, mapping, np.linspace(0, 1, n + 1), np.linspace(0, 1, 6 * n + 1))
    for r in np.linspace(0.1, 0.95, 9):
        ts = np.linspace(0, 1, int(round(44 * r)) + 1)[:-1]
        X, Y = mapping(np.full_like(ts, r), ts)
        A = V1.evaluate_points(s["A_free"], np.full_like(ts, r), ts)
        ax.quiver(X, Y, A[0], A[1], np.hypot(A[0], A[1]), cmap="viridis", scale=12)
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{curl}^\star\mathrm{curl}\,A_h = J$, homotopy gauge $A_h \cdot x = 0$")
    rs, ts = np.linspace(1e-3, 1, 121), np.linspace(0, 1, 361)
    C = V1.derivative_matrix
    field_on_disk(
        axes[1],
        mapping,
        rs,
        ts,
        _sample(V2, C @ s["A_free"], rs, ts),
        contours=12,
        colorbar=True,
        label=r"$\mathrm{curl}\,A_h$",
    )
    draw_mesh(axes[1], mapping, np.linspace(0, 1, n + 1), np.linspace(0, 1, 6 * n + 1))
    axes[1].set_title(r"$\mathrm{curl}\,A_h \approx g(x,y)\sin(2x)\cos(y)$")
    fig.tight_layout()
    return fig


def figure_convergence(table, cells=CELLS, degrees=DEGREES):
    hs = 1 / np.array(cells)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = (
        (r"$\Vert u - u_h \Vert_{L^2}$", "Poisson: free (o), Dirichlet (s)", (0, 1), 1),
        (r"$\Vert \mathrm{curl}\,A - \mathrm{curl}\,A_h\Vert_{L^2}$", "curl-curl: free (o), Dirichlet (s)", (2, 3), 0),
    )
    for ax, (ylabel, title, columns, shift) in zip(axes, panels):
        for i, p in enumerate(degrees):
            for k, j in enumerate(columns):
                ax.loglog(
                    hs,
                    table[p][:, j],
                    marker="os"[k],
                    linestyle=("-", "--")[k],
                    color=CATEGORICAL[i],
                    label=f"$p = {p}$" if k == 0 else None,
                    markersize=5,
                )
            segment = hs[-3:]
            anchor = 0.3 * np.min(table[p][-1, list(columns)])
            ax.loglog(segment, anchor * (segment / hs[-1]) ** (p + shift), ":", color=CATEGORICAL[i])
        ax.set_xticks(hs, [f"1/{n}" for n in cells])
        ax.set_xlabel("$h = 1/n$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
