"""Spectrally correct Maxwell eigenvalues on the unit disk (port of PolarFEEC.jl ``examples/06_maxwell_spectrum.jl``).

The polar 1-form space with Dirichlet boundary conditions (zero tangential trace) discretizes the Maxwell eigenproblem
``(curl A, curl B) = lambda (A, B)``, i.e. ``C^T M_2 C a = lambda M_1 a`` in polar coefficients. The nonzero eigenvalues
are ``(j'_{mk})^2``, the squared zeros of the Bessel derivatives ``J_m'`` (double for ``m >= 1``), and
``curl A = J_m(j'_{mk} r) cos(m (theta - theta_0))``. The relative complex has cohomology ``(0, 0, 1)``, so the kernel is
exactly ``grad V0`` with dimension ``dim V0`` and there are ``dim V2 - 1`` nonzero eigenvalues; no spurious modes appear
and the eigenvalues converge like ``h^{2p}``.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from scipy.special import jv, jvp

from struphy.polar.homotopy import ClampedBSplineSpace, DiskMapping, PeriodicBSplineSpace, PolarComplex2D, zernike_basis
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, clean_axis

DEGREE = 4
REGULARITY = 3
MESHES = ((12, 24), (24, 48))
COMPARED = 60


def dbessel_zeros(m, xmax, step=0.05):
    """Zeros of ``J_m'`` in ``(0, xmax)`` by bracketing and bisection."""
    zeros, a = [], max(m, 0.5)
    while a < xmax:
        b = a + step
        fa, fb = jvp(m, a), jvp(m, b)
        if np.signbit(fa) != np.signbit(fb):
            lo, hi = a, b
            for _ in range(200):
                c = (lo + hi) / 2
                if not lo < c < hi:
                    break
                if np.signbit(jvp(m, c)) == np.signbit(jvp(m, lo)):
                    lo = c
                else:
                    hi = c
            zeros.append((lo + hi) / 2)
        a = b
    return zeros


def exact_spectrum(lambda_max=2000.0):
    """Sorted exact eigenvalues below ``lambda_max`` with multiplicity, as ``(lambda, m, k)``."""
    xmax = np.sqrt(lambda_max)
    modes = []
    for m in range(int(np.ceil(xmax)) + 1):
        for k, j in enumerate(dbessel_zeros(m, xmax), start=1):
            modes += [(j**2, m, k)] * (1 if m == 0 else 2)
    return sorted(modes)


def eigenproblem(radial_cells, angular_cells, degree=DEGREE, regularity=REGULARITY):
    """Dense generalized eigenproblem ``C^T M_2 C a = lambda M_1 a`` in the Dirichlet polar 1-form space."""
    polar = PolarComplex2D(
        ClampedBSplineSpace(degree, radial_cells),
        PeriodicBSplineSpace(degree, angular_cells),
        DiskMapping(),
        regularity=regularity,
        boundary="dirichlet",
        basis=zernike_basis(regularity, radius=degree / radial_cells),
    )
    C = polar.derivative(1)
    K = (C.T @ polar.mass_matrix(2) @ C).toarray()
    lam, a = sla.eigh((K + K.T) / 2, polar.mass_matrix(1).toarray())
    kernel = int(np.sum(lam < 1e-8 * lam.max()))
    return dict(polar=polar, lam=lam[kernel:], a=a[:, kernel:], kernel=kernel)


def study(meshes=MESHES, compared=COMPARED, verbose=True):
    exact = exact_spectrum()
    lam_exact = np.array([e[0] for e in exact[:compared]])
    results = []
    for nr, nt in meshes:
        r = eigenproblem(nr, nt)
        dims = [r["polar"].dimension(k) for k in range(3)]
        error = (r["lam"][:compared] - lam_exact) / lam_exact
        r["error"] = np.abs(error)
        if verbose:
            print(
                f"mesh {nr:2d}x{nt:2d}: dim V = {dims}, cohomology {r['polar'].cohomology_dimensions()}, kernel "
                f"{r['kernel']} (= dim V0), nonzero {len(r['lam'])} (= dim V2 - 1 = {dims[2] - 1}); "
                f"upper bounds: {bool(np.all(error > -1e-12))}"
            )
        results.append(r)
    if verbose and len(results) == 2:
        coarse, fine = results
        for i in (1, 5, 10, 20):
            rate = np.log2(coarse["error"][i - 1] / fine["error"][i - 1])
            print(
                f"lambda_{i:<3d} exact {lam_exact[i - 1]:10.4f}  rel. error {coarse['error'][i - 1]:.1e} -> "
                f"{fine['error'][i - 1]:.1e}  (rate h^{rate:.1f})"
            )
    return dict(exact=exact, lam_exact=lam_exact, results=results, meshes=meshes)


def figure_spectrum(s):
    compared = len(s["lam_exact"])
    idx = np.arange(1, compared + 1)
    fine = s["results"][-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    axes[0].plot(idx, s["lam_exact"], "o", mfc="none", mec="k", label=r"exact $(j'_{m,k})^2$")
    axes[0].plot(idx, fine["lam"][:compared], ".", color=CATEGORICAL[0], label="computed, finest mesh")
    axes[0].text(
        0.97,
        0.04,
        rf"plus {fine['kernel']} $= \dim\bar V^0$ zero eigenvalues (gradients)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
    )
    axes[0].set_xlabel("index $i$")
    axes[0].set_ylabel(r"eigenvalue $\lambda_i$")
    axes[0].set_title("Nonzero Maxwell eigenvalues on the unit disk")
    axes[0].legend(loc="upper left")
    for r, (nr, nt), color in zip(s["results"], s["meshes"], (CATEGORICAL[1], CATEGORICAL[0])):
        axes[1].semilogy(idx, np.maximum(r["error"], 1e-16), ".", color=color, label=f"{nr}x{nt} mesh")
    axes[1].set_xlabel("index $i$")
    axes[1].set_ylabel(r"$|\lambda_{h,i} - \lambda_i| / \lambda_i$")
    axes[1].set_title(f"Relative error, degree {DEGREE} splines")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def curl_mode(r, i, m):
    """Polar coefficients of mode ``i`` (degenerate pairs rotated to ``cos(m theta)``) and its curl (full coefficients)."""
    polar = r["polar"]
    E1, C = polar.extraction(1), polar.spaces[1].derivative_matrix
    V2 = polar.spaces[2]

    def curl(a):
        return C @ (E1 @ a)

    if m == 0:
        return curl(r["a"][:, i])
    c1, c2 = curl(r["a"][:, i]), curl(r["a"][:, i + 1])
    v1 = V2.evaluate(c1, [0.4], [1 / (4 * m)])[0, 0]
    v2 = V2.evaluate(c2, [0.4], [1 / (4 * m)])[0, 0]
    return curl(v2 * r["a"][:, i] - v1 * r["a"][:, i + 1])


def figure_modes(s):
    fine = s["results"][-1]
    V2 = fine["polar"].spaces[2]
    exact = s["exact"]
    distinct, seen = [], set()
    for i, (lam, m, k) in enumerate(exact):
        if (m, k) not in seen:
            seen.add((m, k))
            distinct.append(i)
        if len(distinct) == 8:
            break
    xs = np.linspace(1e-7, 1, 97)
    ys = np.linspace(0, 1, 6 * s["meshes"][-1][1] + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Px, Py = X * np.cos(2 * np.pi * Y), X * np.sin(2 * np.pi * Y)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for ax, i in zip(axes.ravel(), distinct):
        lam, m, k = exact[i]
        values = V2.evaluate(curl_mode(fine, i, m), xs, ys)
        reference = jv(m, np.sqrt(lam) * X) * np.cos(2 * np.pi * m * Y)
        values = values * np.sign(np.sum(values * reference)) / np.max(np.abs(values))
        ax.pcolormesh(Px, Py, values, cmap=DIVERGING, vmin=-1, vmax=1, shading="gouraud")
        ax.set_aspect("equal")
        clean_axis(ax)
        ax.set_title(rf"$(m,k) = ({m},{k}),\ \lambda_h = {fine['lam'][i]:.3f}$", fontsize=11)
    fig.suptitle(
        f"Maxwell eigenmodes on the unit disk ({s['meshes'][-1][0]}x{s['meshes'][-1][1]} mesh, degree {DEGREE})"
    )
    fig.tight_layout()
    return fig
