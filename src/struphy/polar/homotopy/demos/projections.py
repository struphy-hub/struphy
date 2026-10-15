"""Convergence and commutativity of the polar projection operators (talk, Section 6).

The polar projections are ``pi_bar = Pi_bar o pi`` with the commuting tensor-product projection ``pi`` (geometric
degrees of freedom), and the polar L2 projections for comparison. With ``r_x = r_y = r`` (B-spline degree equal to
the regularity) the 0-form projections converge like ``h^{r+1}`` and the 1- and 2-form projections like ``h^r``. The
commuting projections commute with ``grad``, ``curl`` and with the homotopies ``grad^dagger``, ``curl^dagger`` up to
round-off (and quadrature), the L2 projections only up to the approximation error.

The test functions, the mesh (``N`` radial and ``4 N`` angular cells) and the perturbed disk are choices of this port;
the talk does not state them.
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy import ClampedBSplineSpace, PeriodicBSplineSpace, PolarComplex2D, perturbed_disk
from struphy.polar.homotopy.demos.plotting import CATEGORICAL


def f0(x, y):
    return np.sin(3 * x + 1) * np.cos(2 * y - 0.5) + 0.5 * np.cos(4 * x * y)


def grad_f0(x, y):
    return (
        3 * np.cos(3 * x + 1) * np.cos(2 * y - 0.5) - 2 * y * np.sin(4 * x * y),
        -2 * np.sin(3 * x + 1) * np.sin(2 * y - 0.5) - 2 * x * np.sin(4 * x * y),
    )


def A1(x, y):
    return (np.cos(x - y), x * np.exp(y))


def curl_A1(x, y):
    return np.exp(y) - np.sin(x - y)


def B2(x, y):
    return np.sin(x + 2 * y) + x**2


def _gauss01(n=20):
    t, w = np.polynomial.legendre.leggauss(n)
    return (t + 1) / 2, w / 2


def poincare_gradient(A):
    """``(grad^dagger A)(x) = int_0^1 A(t x) . x dt``."""
    t, w = _gauss01()

    def h(x, y):
        total = 0.0
        for tq, wq in zip(t, w):
            ax, ay = A(tq * x, tq * y)
            total = total + wq * (ax * x + ay * y)
        return total

    return h


def poincare_curl(B):
    """``(curl^dagger B)(x) = (-y, x) int_0^1 t B(t x) dt``."""
    t, w = _gauss01()

    def h(x, y):
        s = sum(wq * tq * B(tq * x, tq * y) for tq, wq in zip(t, w))
        return (-y * s, x * s)

    return h


def complexes(regularity, cells=(4, 8, 16, 32), angular_factor=4, mapping=None):
    mapping = perturbed_disk() if mapping is None else mapping
    return [
        PolarComplex2D(
            ClampedBSplineSpace(regularity, n),
            PeriodicBSplineSpace(regularity, angular_factor * n),
            mapping,
            regularity=regularity,
        )
        for n in cells
    ]


def _norm(c, k, coefficients):
    M = c.full_mass_matrix(k)
    return float(np.sqrt(max(coefficients @ (M @ coefficients), 0.0)))


def project(c, k, f, method):
    return c[k](f, additional_order=6) if method == "commuting" else c.l2_projection(k, f, additional_order=0)


def convergence(regularities=(1, 2, 3, 4, 5, 6), cells=(4, 8, 16, 32), method="commuting"):
    """Relative L2 errors of the 0-, 1-, 2-form projections, ``errors[r]`` of shape ``(len(cells), 3)``."""
    targets = (f0, A1, B2)
    errors = {}
    for r in regularities:
        rows = []
        for c in complexes(r, cells):
            row = []
            for k, f in enumerate(targets):
                u = project(c, k, f, method)
                zero = np.zeros_like(u)
                row.append(c.spaces[k].l2_error(u, f, 4) / c.spaces[k].l2_error(zero, f, 4))
            rows.append(row)
        errors[r] = np.array(rows)
    return errors


def commutation(regularities=(1, 2, 3, 4, 5, 6), cells=(4, 8, 16, 32), method="commuting"):
    """Relative defects of ``grad``, ``curl``, ``grad^dagger`` and ``curl^dagger`` with the projections."""
    defects = {}
    for r in regularities:
        rows = []
        for c in complexes(r, cells):
            G, C = c.spaces[0].derivative_matrix, c.spaces[1].derivative_matrix
            HG, HC = c.gradient_homotopy_matrix, c.curl_homotopy_matrix
            P = lambda k, f: project(c, k, f, method)  # noqa: E731
            pairs = (
                (1, P(1, grad_f0), G @ P(0, f0)),
                (2, P(2, curl_A1), C @ P(1, A1)),
                (0, P(0, poincare_gradient(A1)), HG @ P(1, A1)),
                (1, P(1, poincare_curl(B2)), HC @ P(2, B2)),
            )
            rows.append([_norm(c, k, a - b) / _norm(c, k, a) for k, a, b in pairs])
        defects[r] = np.array(rows)
    return defects


def figure_convergence(errors, cells=(4, 8, 16, 32), title=None):
    hs = 1 / np.array(cells)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for k, ax in enumerate(axes):
        for i, (r, e) in enumerate(errors.items()):
            ax.loglog(hs, e[:, k], "o-", color=CATEGORICAL[i % len(CATEGORICAL)], markersize=4, label=f"$r = {r}$")
        r_max = max(errors)
        rate = r_max + 1 if k == 0 else r_max
        e = errors[r_max][:, k]
        ax.loglog(hs[-3:], 0.3 * e[-1] * (hs[-3:] / hs[-1]) ** rate, "k:", label=f"$O(h^{rate})$")
        ax.set_xlabel("$h$")
        ax.set_title(f"{k}-forms")
        if k == 0:
            ax.set_ylabel("relative $L^2$ error")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_commutation(defects, cells=(4, 8, 16, 32), title=None):
    hs = 1 / np.array(cells)
    names = (r"grad", r"curl", r"grad$^\dagger$", r"curl$^\dagger$")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for j, ax in enumerate(axes):
        for i, (r, d) in enumerate(defects.items()):
            ax.loglog(
                hs,
                np.maximum(d[:, j], 1e-17),
                "o-",
                color=CATEGORICAL[i % len(CATEGORICAL)],
                markersize=4,
                label=f"$r = {r}$",
            )
        ax.set_ylim(1e-17, 2)
        ax.set_xlabel("$h$")
        ax.set_title(names[j])
        if j == 0:
            ax.set_ylabel("defect")
    axes[-1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
