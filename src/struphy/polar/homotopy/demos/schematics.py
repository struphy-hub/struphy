"""The schematic diagrams of the talk: de Rham diagrams (drawn with ``tikzfigure``, needs LaTeX), the 3D coordinate
transformation of the helical tube and the steps of the boundary-compatible homotopy (computed with the Struphy
domains and polar complexes, drawn with matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos import common
from struphy.polar.homotopy.demos.plotting import DIVERGING, clean_axis, draw_mesh


# ---------------------------------------------------------------------------------------------------------- diagrams
def _diagram(rows, horizontal, vertical, column_sep="1.7cm", row_sep="1.25cm", origin=(0, 0), name="m"):
    """TikZ code of a commutative diagram: ``rows`` (math entries), ``horizontal[(i, j)] = (label, reversed)``
    arrows from column ``j`` to ``j + 1`` in row ``i`` and ``vertical[j] = label`` arrows from the last to the first
    row in column ``j``."""
    entries = " \\\\\n".join(" & ".join(row) for row in rows)
    code = [
        f"\\matrix ({name}) [matrix of math nodes, row sep={row_sep}, column sep={column_sep}, "
        f"shift={{({origin[0]}cm, {origin[1]}cm)}}] {{\n{entries} \\\\\n}};"
    ]
    for (i, j), (label, reverse) in horizontal.items():
        a, b = f"{name}-{i + 1}-{j + 1}", f"{name}-{i + 1}-{j + 2}"
        if reverse:
            a, b = b, a
        code.append(f"\\draw[-{{Latex}}] ({a}) -- node[above]{{\\scriptsize {label}}} ({b});")
    last = len(rows)
    for j, label in vertical.items():
        code.append(
            f"\\draw[-{{Latex}}, color=struphyblue] ({name}-{last}-{j + 1}) -- "
            f"node[left]{{\\scriptsize ${label}$}} ({name}-1-{j + 1});"
        )
    return "\n".join(code)


def _figure(code):
    from tikzfigure import TikzFigure

    fig = TikzFigure()
    fig.add_package("amssymb")
    fig.usetikzlibrary("matrix", "arrows.meta")
    fig.add_raw("\\definecolor{struphyblue}{rgb}{0.0, 0.447, 0.741}\n" + code)
    return fig


def derham_logical_and_physical():
    """The commuting projections of the logical and of the physical 2D de Rham complex."""
    spaces = ["H^1", r"H(\mathrm{curl})", "L^2"]
    logical = _diagram(
        [
            [r"\hat V^0_h", r"\hat V^1_h", r"\hat V^2_h"],
            [r"\hat " + s if s != r"H(\mathrm{curl})" else r"\hat H(\mathrm{curl})" for s in spaces],
        ],
        {(0, 0): ("grad", False), (0, 1): ("curl", False), (1, 0): ("grad", False), (1, 1): ("curl", False)},
        {0: r"\hat\pi^0", 1: r"\hat\pi^1", 2: r"\hat\pi^2"},
        name="l",
    )
    physical = _diagram(
        [["V^0_h", "V^1_h", "V^2_h"], spaces],
        {(0, 0): ("grad", False), (0, 1): ("curl", False), (1, 0): ("grad", False), (1, 1): ("curl", False)},
        {0: r"\pi^0", 1: r"\pi^1", 2: r"\pi^2"},
        origin=(9, 0),
        name="p",
    )
    link = (
        "\\draw[-{Latex}, dashed] (l-1-3.east) ++(0.6, -1.1) -- node[above]{\\scriptsize pushforward $F_*$} ++(2.2, 0);"
    )
    return _figure(logical + "\n" + physical + "\n" + link)


def derham_polar_cochain_chain():
    """The polar subspaces form a cochain complex (grad, curl) and a chain complex (the homotopies) with commuting
    projections ``pi_bar^{k, r}``."""
    top = [r"\bar V^{0,\mathfrak r}", r"\bar V^{1,\mathfrak r}", r"\bar V^{2,\mathfrak r}"]
    bottom = ["H^1", r"H(\mathrm{curl})", "L^2"]
    vertical = {0: r"\bar\pi^{0,\mathfrak r}", 1: r"\bar\pi^{1,\mathfrak r}", 2: r"\bar\pi^{2,\mathfrak r}"}
    cochain = _diagram(
        [top, bottom],
        {(0, 0): ("grad", False), (0, 1): ("curl", False), (1, 0): ("grad", False), (1, 1): ("curl", False)},
        vertical,
        name="c",
    )
    chain = _diagram(
        [top, bottom],
        {
            (0, 0): (r"$\mathrm{grad}^\dagger$", True),
            (0, 1): (r"$\mathrm{curl}^\dagger$", True),
            (1, 0): (r"$\mathrm{grad}^\dagger$", True),
            (1, 1): (r"$\mathrm{curl}^\dagger$", True),
        },
        vertical,
        origin=(9, 0),
        name="h",
    )
    return _figure(cochain + "\n" + chain)


# ---------------------------------------------------------------------------------------------------------- geometry
def coordinate_transformation_3d(n=(8, 16, 48)):
    """The logical cube (the face ``eta1 = 0`` collapses to the magnetic axis) and the helical tube with two
    cross-sections, the mesh lines and the axis."""
    from struphy.geometry import domains

    domain = domains.HelicalTube("stellarator", num_elements=(8, 24, 64), degree=(4, 4, 4))
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for t in np.linspace(0, 1, 5):
        for s in np.linspace(0, 1, 5):
            ax.plot([t, t], [s, s], [0, 1], color="0.6", linewidth=0.5)
            ax.plot([t, t], [0, 1], [s, s], color="0.6", linewidth=0.5)
            ax.plot([0, 1], [t, t], [s, s], color="0.6", linewidth=0.5)
    face = np.linspace(0, 1, 2)
    Y, Z = np.meshgrid(face, face)
    ax.plot_surface(np.zeros_like(Y), Y, Z, color="#E69F00", alpha=0.5)
    ax.set_xlabel(r"$\eta_1$")
    ax.set_ylabel(r"$\eta_2$")
    ax.set_zlabel(r"$\eta_3$")
    ax.set_title(r"logical cube ($\eta_1 = 0$: the axis)")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    e1, e2, e3 = np.array([1.0]), np.linspace(0, 1, 49), np.linspace(0, 0.8, 120)
    X, Y, Z = (np.asarray(c)[0] for c in domain(e1, e2, e3))
    ax.plot_surface(X, Y, Z, color="0.85", alpha=0.35, linewidth=0)
    for z in np.linspace(0, 0.8, 9):
        ring = np.asarray(domain(e1, e2, np.array([z])))[:, 0, :, 0]
        ax.plot(*ring, color="0.4", linewidth=0.5)
    for z in (0.0, 0.8):
        r, t = np.linspace(0, 1, 9), np.linspace(0, 1, 49)
        S = np.asarray(domain(r, t, np.array([z])))[:, :, :, 0]
        ax.plot_surface(*S, color="0.95", edgecolor="0.4", linewidth=0.3)
    axis = np.asarray(domain(np.array([0.0]), np.array([0.0]), e3))[:, 0, 0, :]
    ax.plot(*axis, color="#E69F00", linewidth=2)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_axis_off()
    ax.set_title("HelicalTube (magnetic axis in orange)")
    fig.tight_layout()
    return fig


def boundary_homotopy(regularity=3):
    """The boundary-compatible homotopy ``curl^dagger_partial`` of a zero-mean 2-form ``B``: the radial homotopy
    ``curl^dagger B`` has a tangential trace on the boundary; its periodic primitive, extended into the domain, is
    subtracted as a gradient, which leaves a potential ``A`` with ``curl A = B`` and ``A x n = 0``."""
    c = common.polar_complex(regularity, 4, (6, 18), boundary="dirichlet")
    mapping = c.spaces[0].mapping

    def B(x, y):
        return np.sin(3 * x + 1) * np.cos(2 * y) + x * y

    b = c[2](B)
    # zero mean in polar coefficients, then back to full coefficients
    polar = c.duals(2).T @ b
    polar = polar - np.asarray(c.cohomology_projection(2) @ polar).ravel()
    b0 = c.extraction(2) @ polar
    radial = c.curl_homotopy_matrix @ b0
    bounded = c.extraction(1) @ np.asarray(c.homotopy(2) @ polar).ravel()
    rs, ts = np.linspace(1e-3, 1, 81), np.linspace(0, 1, 181)
    X, Y = mapping(*np.meshgrid(rs, ts, indexing="ij"))
    panels = [
        (c.spaces[2].evaluate(b0, rs, ts), r"$B$ (zero mean)"),
        (
            np.hypot(*c.spaces[1].evaluate(radial, rs, ts)),
            r"$|\mathrm{curl}^\dagger B|$: tangential trace on $\partial\Omega$",
        ),
        (
            np.hypot(*c.spaces[1].evaluate(radial - bounded, rs, ts)),
            r"$|\mathrm{grad}\,\phi|$: boundary primitive (outer ring)",
        ),
        (np.hypot(*c.spaces[1].evaluate(bounded, rs, ts)), r"$|\mathrm{curl}^\dagger_\partial B|$: $A \times n = 0$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    # one colour range for the potentials (the extension of the boundary primitive lives in the outer ring and
    # saturates there)
    common_limit = np.max(panels[1][0]) or 1.0
    for index, (ax, (values, title)) in enumerate(zip(axes, panels)):
        signed = np.min(values) < 0
        limit = (np.max(np.abs(values)) or 1.0) if index == 0 else common_limit
        ax.pcolormesh(
            X,
            Y,
            values,
            cmap=DIVERGING if signed else "viridis",
            vmin=-limit if signed else 0,
            vmax=limit,
            shading="gouraud",
        )
        draw_mesh(ax, mapping, np.linspace(0, 1, 7), np.linspace(0, 1, 19), alpha=0.15)
        ax.set_aspect("equal")
        clean_axis(ax)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def write_all(folder):
    """Write all schematics into ``folder`` (``.pdf`` for the TikZ diagrams and ``.png``)."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, make in (
        ("derham_logical_and_physical_2d", derham_logical_and_physical),
        ("derham_polar_cochain_chain_2d", derham_polar_cochain_chain),
    ):
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(folder / f"{name}.{ext}")
        print("wrote", folder / f"{name}.png")
    for name, make in (
        ("coordinate_transformation_3d", coordinate_transformation_3d),
        ("bdy_homotopy", boundary_homotopy),
    ):
        fig = make()
        fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", folder / f"{name}.png")
