"""Plotting helpers shared by the demos (matplotlib versions of the conventions of PolarFEEC.jl's ``examples/common.jl``).

Signed fields use the colour scheme of the paper's figures: MATLAB blue, white, MATLAB orange, with eased transitions
(use a symmetric colour range).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_BLUE, _WHITE, _ORANGE = (0.0, 0.4470, 0.7410), (1.0, 1.0, 1.0), (0.8500, 0.3250, 0.0980)


def _diverging():
    t = np.linspace(0, 1, 256)
    eased = 3 * t**2 - 2 * t**3
    colors = []
    for s in eased:
        if s < 0.5:
            a = s / 0.5
            colors.append(tuple((1 - a) * b + a * w for b, w in zip(_BLUE, _WHITE)))
        else:
            a = (s - 0.5) / 0.5
            colors.append(tuple((1 - a) * w + a * o for w, o in zip(_WHITE, _ORANGE)))
    return LinearSegmentedColormap.from_list("polarfeec_diverging", colors)


DIVERGING = _diverging()
SEQUENTIAL = "viridis"
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def mapped_grid(mapping, xs, ys):
    """Physical coordinates of the logical tensor grid ``xs (x) ys``."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return mapping(X, Y)


def draw_mesh(ax, mapping, radial_breaks, angular_breaks, color="k", alpha=0.13, linewidth=0.6, boundary=True):
    """Mesh lines of a disk-like mapping (rings ``x = const`` and spokes ``y = const``)."""
    t = np.linspace(0, 1, 400)
    for r in radial_breaks:
        if 0 < r < 1:
            ax.plot(*mapping(np.full_like(t, r), t), color=color, alpha=alpha, linewidth=linewidth)
    for th in angular_breaks:
        ax.plot(*mapping(t, np.full_like(t, th)), color=color, alpha=alpha, linewidth=linewidth)
    if boundary:
        ax.plot(*mapping(np.ones_like(t), t), color=color, linewidth=1.2)


def field_on_disk(ax, mapping, xs, ys, values, symmetric=True, cmap=None, contours=0, colorbar=False, label=None):
    """``pcolormesh`` of values on the mapped logical grid ``xs (x) ys`` (values of shape ``(len(xs), len(ys))``)."""
    X, Y = mapped_grid(mapping, xs, ys)
    vmax = np.max(np.abs(values)) or 1.0
    vmin = -vmax if symmetric else np.min(values)
    cmap = (DIVERGING if symmetric else SEQUENTIAL) if cmap is None else cmap
    mesh = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    if contours:
        ax.contour(
            X, Y, values, levels=np.linspace(vmin, vmax, contours + 2)[1:-1], colors="k", linewidths=0.6, alpha=0.5
        )
    ax.set_aspect("equal")
    if colorbar:
        plt.colorbar(mesh, ax=ax, label=label, shrink=0.8)
    return mesh


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def loglog_convergence(ax, hs, errors, rate, color, label=None, marker="o", linestyle="-", reference=True):
    """Error curve against ``h`` with a dotted ``h^rate`` reference below the finest points."""
    ax.loglog(hs, errors, marker=marker, linestyle=linestyle, color=color, label=label, markersize=5)
    if reference:
        segment = np.asarray(hs[-3:])
        anchor = 0.3 * errors[-1]
        ax.loglog(segment, anchor * (segment / segment[-1]) ** rate, ":", color=color, linewidth=1.3)
