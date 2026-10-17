"""1D spaces, tensor-product forms and mapped forms (ports of PolarFEEC.jl ``examples/01_spaces_1d.jl``,
``02_commuting_projections.jl`` and ``03_mapped_forms.jl``).

* 01: clamped and periodic B-splines and trigonometric functions with their derivative-scaled companions; the exact
  coefficient derivative ``d/dx B_i = C_{i-1} - C_i`` and the commuting 1D diagram ``d pi^0 f = pi^1 f'``.
* 02: 0-, 1- and 2-forms on the unit square (clamped x periodic), ``curl o grad = 0`` exactly, projections that
  commute with the derivatives to round-off; commuting and L2 projections converge at order ``p + 1`` for 0-forms and
  ``p`` for 1- and 2-forms.
* 03: a quarter annulus (``HollowCylinder(a1=1, a2=2, poc=4)``; the Julia example warps it); physical 0-, 1- and
  2-forms (scalar, covariant Piola and density pushforwards) with commuting projections
  ``grad pi^0 phi = pi^1 grad phi`` and ``curl pi^1 u = pi^2 curl u``.

The 1D spaces are the spline spaces of Struphy's Derham complex (feectools ``SplineSpace``) and the trigonometric
spaces of :mod:`struphy.feec.trig_space`; the square and the annulus are Struphy domains.
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos import common
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING
from struphy.polar.homotopy.forms2d import PhysicalFormSpace2D
from struphy.polar.homotopy.layout import STRUPHY


def complex_step(f, x, *args, argnum=0, h=1e-30):
    """Derivative of the (analytic) function ``f`` in its argument ``argnum`` by the complex step."""
    values = [np.asarray(x, dtype=complex), *[np.asarray(a, dtype=complex) for a in args]]
    values[argnum] = values[argnum] + 1j * h
    return np.imag(f(*values)) / h


# ---------------------------------------------------------------------------------------------------- 01
def spaces_1d(verbose=True):
    """Commuting-diagram residuals ``|d pi^0 f - pi^1 f'|`` of the three 1D families."""
    families = {
        "clamped B-splines (3, 6)": (common.spline_space_1d(3, 6), lambda x: np.exp(-x) * np.sin(4 * np.pi * x)),
        "periodic B-splines (3, 6)": (common.spline_space_1d(3, 6, True), lambda x: np.exp(np.sin(2 * np.pi * x))),
        "trigonometric (2 modes)": (common.trigonometric_space_1d(5), lambda x: np.exp(np.sin(2 * np.pi * x))),
    }
    residuals = {}
    for name, (space, f) in families.items():
        u = space.commuting_projection(f)
        derivative = space.scaled.commuting_projection(lambda x, f=f: complex_step(f, x), additional_order=8)
        residuals[name] = float(np.max(np.abs(space.derivative_matrix() @ u - derivative)))
        if verbose:
            print(f"{name:28s} |d pi0 f - pi1 f'| = {residuals[name]:.1e}")
    return families, residuals


def figure_spaces_1d():
    families, _ = spaces_1d(verbose=False)
    x = np.linspace(0, 1, 801)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, (name, (space, f)) in enumerate(families.items()):
        for k, s in enumerate((space, space.scaled)):
            values = s.basis_matrix(x).toarray()
            for i in range(values.shape[1]):
                axes[k, j].plot(x, values[:, i], color=CATEGORICAL[i % len(CATEGORICAL)], lw=1.2)
            for p in s.greville_points():
                axes[k, j].axvline(p, color="gray", lw=0.5, ls=":")
            axes[k, j].set_title(("$V^0$: " if k == 0 else "$V^1$ (scaled): ") + name, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("$x$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------- 02
def identity():
    """The unit square (the Struphy domain ``Cuboid``)."""
    from struphy.geometry import domains
    from struphy.polar.homotopy.derham_blocks import DomainCrossSection

    return DomainCrossSection(domains.Cuboid(), centered=False)


def _g(x):
    return np.exp(-(((x - 0.5) / 0.3) ** 2))


def psi(x, y):
    return _g(x) * np.sin(2 * np.pi * (y - x / 2))


def A_field(x, y):
    """``A = (-d_y psi, d_x psi)``."""
    return (-complex_step(psi, x, y, argnum=1), complex_step(psi, x, y, argnum=0))


def curl_A(x, y):
    """``curl A = Laplace psi`` (analytic)."""
    g = _g(x)
    dg = -2 * (x - 0.5) / 0.09 * g
    ddg = (4 * (x - 0.5) ** 2 / 0.09**2 - 2 / 0.09) * g
    phase = 2 * np.pi * (y - x / 2)
    S, C = np.sin(phase), np.cos(phase)
    return ddg * S + 2 * dg * (-np.pi * C) + g * (-(np.pi**2) * S) + g * (-4 * np.pi**2 * S)


def tensor_spaces(p, n):
    return [
        PhysicalFormSpace2D.from_factors(
            k, common.spline_space_1d(p, n), common.spline_space_1d(p, n, True), identity(), STRUPHY
        )
        for k in range(3)
    ]


def identities(p=3, n=8, verbose=True):
    """``curl o grad = 0`` and the commutation residuals of the projections on the square."""
    V0, V1, V2 = tensor_spaces(p, n)
    G, C = V0.derivative_matrix, V1.derivative_matrix
    curl_grad = float(abs(C @ G).max())
    grad_residual = float(
        np.max(
            np.abs(
                G @ V0.commuting_projection(psi)
                - V1.commuting_projection(
                    lambda x, y: (complex_step(psi, x, y, argnum=0), complex_step(psi, x, y, argnum=1)),
                    additional_order=4,
                )
            )
        )
    )
    curl_residual = float(
        np.max(
            np.abs(
                C @ V1.commuting_projection(A_field, additional_order=4)
                - V2.commuting_projection(curl_A, additional_order=4)
            )
        )
    )
    if verbose:
        print(
            f"|curl grad| = {curl_grad:.1e}, |grad pi0 psi - pi1 grad psi| = {grad_residual:.1e}, "
            f"|curl pi1 A - pi2 curl A| = {curl_residual:.1e}"
        )
    return curl_grad, grad_residual, curl_residual


def convergence(degrees=(1, 2, 3, 4, 5), cells=(8, 16, 32, 64, 128), verbose=True):
    """L2 errors of the commuting and L2 projections of psi, A and curl A, ``errors[method][p]`` (cells x 3)."""
    targets = (psi, A_field, curl_A)
    errors = {"commuting": {}, "l2": {}}
    for p in degrees:
        rows = {"commuting": [], "l2": []}
        for n in cells:
            spaces = tensor_spaces(p, n)
            for method in rows:
                row = []
                for k, (space, f) in enumerate(zip(spaces, targets)):
                    u = space.commuting_projection(f, 4) if method == "commuting" else space.l2_projection(f)
                    row.append(space.l2_error(u, f, 4))
                rows[method].append(row)
        for method in rows:
            errors[method][p] = np.array(rows[method])
        if verbose:
            e = errors["commuting"][p]
            print(f"p = {p}: observed rates (commuting) {np.log2(e[-2] / e[-1]).round(2)}")
    return errors


def figure_convergence(errors, cells=(8, 16, 32, 64, 128)):
    hs = 1 / np.array(cells)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True)
    for k, ax in enumerate(axes):
        for i, p in enumerate(errors["commuting"]):
            color = CATEGORICAL[i % len(CATEGORICAL)]
            ax.loglog(hs, errors["commuting"][p][:, k], "o-", color=color, label=f"$p = {p}$", ms=4)
            ax.loglog(hs, errors["l2"][p][:, k], "s", mfc="none", color=color, ms=5)
            rate = p + 1 if k == 0 else p
            e = errors["commuting"][p][:, k]
            ax.loglog(hs[-3:], 0.3 * e[-1] * (hs[-3:] / hs[-1]) ** rate, ":", color=color)
        ax.set_ylim(1e-13, 10)
        ax.set_xlabel("$h$")
        ax.set_title([r"0-form $\psi$", "1-form $A$", r"2-form curl $A$"][k])
    axes[0].set_ylabel(r"$L^2$ error (circles: commuting, squares: $L^2$ projection)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------- 03
def quarter_annulus():
    """The quarter annulus ``1 < r < 2`` (the Struphy domain ``HollowCylinder(a1=1, a2=2, poc=4)``)."""
    from struphy.geometry import domains
    from struphy.polar.homotopy.derham_blocks import DomainCrossSection

    return DomainCrossSection(domains.HollowCylinder(a1=1.0, a2=2.0, Lz=1.0, poc=4), centered=False)


def vortices(x, y):
    """Two Gaussian vortices (strengths 1 and -0.8, width 0.4)."""
    out = 0.0
    for (cx, cy), s in (((1.45, 0.55), 1.0), ((0.6, 1.45), -0.8)):
        out = out + s * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.4**2)
    return out


def mapped_spaces(n=8):
    mapping = quarter_annulus()
    return [
        PhysicalFormSpace2D.from_factors(
            k, common.spline_space_1d(3, n), common.spline_space_1d(3, 2 * n), mapping, STRUPHY
        )
        for k in range(3)
    ]


def mapped_identities(n=8, verbose=True):
    V0, V1, V2 = mapped_spaces(n)

    def grad_phi(x, y):
        return (complex_step(vortices, x, y, argnum=0), complex_step(vortices, x, y, argnum=1))

    def u(x, y):
        gx, gy = grad_phi(x, y)
        return (-gy + 0.3 * y, gx + 0.2 * x**2)

    def curl_u(x, y):
        h = 1e-5
        dvx = (u(x + h, y)[1] - u(x - h, y)[1]) / (2 * h)
        duy = (u(x, y + h)[0] - u(x, y - h)[0]) / (2 * h)
        return dvx - duy

    o = 8
    grad_residual = float(
        np.max(
            np.abs(V0.derivative_matrix @ V0.commuting_projection(vortices, o) - V1.commuting_projection(grad_phi, o))
        )
    )
    curl_residual = float(
        np.max(np.abs(V1.derivative_matrix @ V1.commuting_projection(u, o) - V2.commuting_projection(curl_u, o)))
    )
    if verbose:
        print(
            f"mapped: |grad pi0 phi - pi1 grad phi| = {grad_residual:.1e}, |curl pi1 u - pi2 curl u| = "
            f"{curl_residual:.1e} (finite differences in curl u)"
        )
    return dict(spaces=(V0, V1, V2), phi=V0.commuting_projection(vortices, o), u=V1.commuting_projection(u, o))


def figure_mapped_forms(n=8):
    result = mapped_identities(n, verbose=False)
    V0, V1, V2 = result["spaces"]
    mapping = V0.mapping
    xs, ys = np.linspace(0, 1, 81), np.linspace(0, 1, 161)
    X, Y = mapping(*np.meshgrid(xs, ys, indexing="ij"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    phi = V0.evaluate(result["phi"], xs, ys)
    axes[0].pcolormesh(X, Y, phi, cmap=DIVERGING, vmin=-np.abs(phi).max(), vmax=np.abs(phi).max(), shading="gouraud")
    axes[0].contour(X, Y, phi, 12, colors="k", linewidths=0.5)
    axes[0].set_title(r"$\phi_h$ (0-form)")
    qx, qy = np.linspace(0.05, 0.95, 9), np.linspace(0.02, 0.98, 17)
    Qx, Qy = np.meshgrid(qx, qy, indexing="ij")
    ux, uy = V1.evaluate_points(result["u"], Qx, Qy)
    axes[1].quiver(*mapping(Qx, Qy), ux, uy, np.hypot(ux, uy), cmap="viridis")
    axes[1].set_title(r"$u_h$ (1-form, covariant Piola)")
    curl = V2.evaluate(V1.derivative_matrix @ result["u"], xs, ys)
    axes[2].pcolormesh(X, Y, curl, cmap=DIVERGING, vmin=-np.abs(curl).max(), vmax=np.abs(curl).max(), shading="gouraud")
    axes[2].set_title(r"curl $u_h$ (2-form)")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig
