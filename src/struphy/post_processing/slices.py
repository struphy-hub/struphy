"""Standard slice plots of FEEC fields in physical planes (used e.g. by the pproc scripts of the FEEC examples).

For each field, :func:`plot_physical_slices` makes one figure per physical plane, each with the Struphy result on top
of its error (with respect to an exact solution) below:

* ``xy``, ``xz``, ``yz``: the planes z = const, y = 0 and x = 0 through the domain, as functions of the Cartesian
  coordinates of the plane,
* ``rz`` (cylinders and tori only): the poloidal plane at the angle ``phi`` (atan2(y, x), default pi), as a function
  of the distance to the axis (r for a cylinder, the major radius R for a torus) and z.

The fields are evaluated from the saved spline coefficients on the logical surfaces that are mapped onto these
planes (see :func:`physical_planes`), at ``points_per_cell`` points per cell of the simulation grid.
"""

from dataclasses import dataclass

import cunumpy as xp
import numpy as np
from matplotlib import pyplot as plt

from struphy.post_processing.output import Output


@dataclass
class Plane:
    """A physical plane, made of the logical surfaces at ``cut = values`` (one or two, e.g. two half planes)."""

    hlabel: str  # coordinate on the horizontal axis: "x", "y", "z", "r" or "R"
    vlabel: str  # coordinate on the vertical axis
    where: str  # description of the plane, for the title
    cut: str  # "eta1", "eta2" or "eta3"
    values: tuple[float, ...]


def physical_planes(domain, phi=np.pi) -> dict[str, Plane]:
    """The physical planes through ``domain`` as logical surfaces (Cuboid, HollowCylinder, IGAPolarCylinder,
    HollowTorus, IGAPolarTorus).

    For the spline mappings (IGAPolar...) the poloidal angle is only approximately 2 pi eta2, so the planes cut at
    eta2 are approximate. Domains covering only part of an angle (``poc``, ``pol_period`` or ``tor_period`` other
    than 1) raise a ValueError."""
    kind = type(domain).__name__
    # the cuts below assume full angles: theta = 2 pi eta2 and phi = -2 pi eta3
    periods = {k: domain.params.get(k, 1) for k in ("poc", "pol_period", "tor_period")}
    if any(v != 1 for v in periods.values()):
        raise ValueError(f"Physical planes of {kind} only for full angles (periods 1), got {periods}.")
    if kind == "Cuboid":
        return {
            "xy": Plane("x", "y", "z at the center", "eta3", (0.5,)),
            "xz": Plane("x", "z", "y at the center", "eta2", (0.5,)),
            "yz": Plane("y", "z", "x at the center", "eta1", (0.5,)),
        }
    if kind in ("HollowCylinder", "IGAPolarCylinder"):
        # theta = 2 pi eta2, z = Lz eta3
        return {
            "xy": Plane("x", "y", "z at the center", "eta3", (0.5,)),
            "xz": Plane("x", "z", "y = 0", "eta2", (0.0, 0.5)),
            "yz": Plane("y", "z", "x = 0", "eta2", (0.25, 0.75)),
            "rz": Plane("r", "z", f"phi = {phi:.3g}", "eta2", ((phi / (2 * np.pi)) % 1.0,)),
        }
    if kind in ("HollowTorus", "IGAPolarTorus"):
        # theta = 2 pi eta2, toroidal angle phi = -2 pi eta3 (y = R sin(-2 pi eta3))
        def eta3(angle):
            return (-angle / (2 * np.pi)) % 1.0

        return {
            "xy": Plane("x", "y", "z = 0", "eta2", (0.0, 0.5)),
            "xz": Plane("x", "z", "y = 0", "eta3", (eta3(0.0), eta3(np.pi))),
            "yz": Plane("y", "z", "x = 0", "eta3", (eta3(np.pi / 2), eta3(3 * np.pi / 2))),
            "rz": Plane("R", "z", f"phi = {phi:.3g}", "eta3", (eta3(phi),)),
        }
    raise NotImplementedError(f"No physical planes defined for the domain {kind}.")


def evaluate_plane(out: Output, domain, name, t, exact, plane: Plane, n_pts, kind=None):
    """Numerical and exact values of the field ``name`` on the logical surfaces of ``plane``, and the physical
    coordinates of the evaluation points (including r = R = sqrt(x^2 + y^2)).

    A scalar field (``kind`` None) is plotted as it is. A vector field is given by the ``kind`` of its
    push-forward ("1", "2" or "v"): it is pushed forward to its Cartesian components and its magnitude is plotted.
    ``exact(eta1, eta2, eta3, x, y, z)`` returns the exact values of what is plotted. All arrays have the axis ``cut``
    of length ``len(plane.values)``."""
    etas = {e: np.linspace(0.0, 1.0, n_pts) for e in ("eta1", "eta2", "eta3")}
    etas[plane.cut] = list(plane.values)
    data = out.evaluate(name, t=t, representation=kind, **etas)
    grids = [data.coords[e].values for e in ("eta1", "eta2", "eta3")]
    if kind is None:
        values = data.values
    else:
        cartesian = domain.push([data.isel(component=k).values for k in range(3)], *grids, kind=kind)
        values = np.linalg.norm(np.stack([xp.to_numpy(c) for c in cartesian]), axis=0)
    E1, E2, E3 = np.meshgrid(*grids, indexing="ij")
    coords = {c.lower(): data.coords[c].values for c in ("X", "Y", "Z")}
    coords["r"] = coords["R"] = np.sqrt(coords["x"] ** 2 + coords["y"] ** 2)
    return values, exact(E1, E2, E3, coords["x"], coords["y"], coords["z"]), coords


def plot_plane(num, exact, coords, plane: Plane, label):
    """Struphy result (top) and its error (bottom) in one physical plane, as a function of the physical coordinates
    of that plane. The logical surfaces making up the plane share one color scale."""
    axis = ("eta1", "eta2", "eta3").index(plane.cut)
    h, v = coords[plane.hlabel], coords[plane.vlabel]
    # subplots of about 8 x 8 (inches) at most, shaped like the plane (aspect = height / width)
    aspect = (np.ptp(v) or 1.0) / (np.ptp(h) or 1.0)
    width, height = (8.0, max(8.0 * aspect, 2.0)) if aspect <= 1.0 else (max(8.0 / aspect, 2.5), 8.0)
    fig, axes = plt.subplots(2, 1, figsize=(width + 2.0, 2 * height + 1.5))
    for ax, (data, what) in zip(axes, [(num, "from struphy"), (np.abs(num - exact), "error")]):
        for i in range(len(plane.values)):
            pcm = ax.pcolormesh(
                np.take(h, i, axis=axis),
                np.take(v, i, axis=axis),
                np.take(data, i, axis=axis),
                shading="gouraud",
                vmin=np.nanmin(data),
                vmax=np.nanmax(data),
            )
        fig.colorbar(pcm, ax=ax)
        ax.set_aspect("equal")
        ax.set_xlabel(plane.hlabel)
        ax.set_ylabel(plane.vlabel)
        ax.set_title(f"{label} {what}, {plane.hlabel}-{plane.vlabel} plane ({plane.where})")
    fig.tight_layout()
    return fig


def plot_physical_slices(out: Output, domain, num_elements, fields, phi=np.pi, points_per_cell=2):
    """Figures of the fields in the physical planes of ``domain`` (see :func:`physical_planes`).

    Parameters
    ----------
    out : Output
        Output of the run.
    domain : Domain
        Domain of the run (``sim.domain``).
    num_elements : tuple[int]
        Number of cells of the simulation grid; the planes are sampled at ``points_per_cell`` points per cell.
    fields : list[tuple]
        ``(key, label, name, t, exact, kind)`` per field: ``key`` names the figures (``<key>_slices_<plane>``),
        ``label`` is used in the titles, ``name`` and ``t`` select the raw FEEC field and snapshot (as in
        :meth:`Output.evaluate`), ``kind`` is None for a scalar field and the push-forward ("1", "2" or "v") of a
        vector field, whose magnitude is plotted. ``exact(eta1, eta2, eta3, x, y, z)`` returns the exact values of
        what is plotted.
    phi : float
        Angle of the poloidal plane ``rz``.

    Returns
    -------
    dict[str, Figure]
        The figures by name, e.g. ``"phi_slices_xy"``.
    """
    n_pts = points_per_cell * max(num_elements) + 1
    figs = {}
    for plane_name, plane in physical_planes(domain, phi).items():
        for key, label, name, t, exact, kind in fields:
            num, exact_values, coords = evaluate_plane(out, domain, name, t, exact, plane, n_pts, kind)
            figs[f"{key}_slices_{plane_name}"] = plot_plane(num, exact_values, coords, plane, label)
    return figs
