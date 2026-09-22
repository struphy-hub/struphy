"""Axisymmetric (r, z) ion-optics domains as thin wedges of revolution.

A rotationally symmetric electrode system is represented by its meridional half-plane
``(r, z)``, revolved about the beam axis ``z`` through a small angle ``2 pi / tor_period``.
This is a :class:`~struphy.geometry.base.PoloidalSplineTorus` whose poloidal plane is
the meridional plane:

.. math::

    x = r \\cos\\theta, \\quad y = -r \\sin\\theta, \\quad z = z, \\qquad
    z = L \\eta_1, \\quad r = r_0 + \\eta_2 [R(z) - r_0], \\quad \\theta = 2\\pi \\eta_3 / N.

Consequences for ion optics:

* The FEEC volume element is ``r dr dz dtheta``, so the weak Poisson problem on the wedge
  *is* the axisymmetric one; one element in ``eta3`` represents the (theta-independent)
  potential exactly.
* The natural boundary condition on the wedge faces (``eta3 = 0, 1``) and on the thin
  inner cylinder ``r = r_0`` around the axis (``eta2 = 0``) gives the symmetry conditions
  ``d phi / d theta = 0`` and ``d phi / d r = 0``. Use free FEEC boundaries in ``eta3``.
* Particles are reflected on these surfaces (``BoundaryParameters(bc=(..., "reflect", "reflect"))``).
  For an axisymmetric ensemble this is exact, because reflection is a symmetry of the field.
* Electrodes are :class:`~struphy.geometry.domains.ElectrodeSegment` on the outer wall
  (``side="upper"``, i.e. ``eta2 = 1``), with axial coordinates ``x0, x1``.
* A ray at radius ``r`` stands for a ring: in the wedge it carries the fraction
  ``1 / tor_period`` of the ring current.
"""

import copy
from dataclasses import asdict

import cunumpy as xp

from struphy.geometry.base import PoloidalSplineTorus, interp_mapping
from struphy.geometry.domains import ElectrodeSegment


class AxisymmetricElectrodeChannel(PoloidalSplineTorus):
    """Wedge of revolution bounded by the axis cylinder ``r = r_0`` and an electrode wall ``r = R(z)``.

    Parameters
    ----------
    length : float
        Axial length L of the domain.

    wall_profile : tuple
        ``(z_nodes, r_nodes)`` of the outer (electrode) wall; linearly interpolated.

    segments : tuple[ElectrodeSegment]
        Electrodes on the outer wall (``side="upper"``), axial extent ``[x0, x1]``.

    axis_radius : float
        Radius r_0 of the thin cylinder that replaces the axis (natural boundary).

    tor_period : int
        The wedge angle is ``2 pi / tor_period``.

    num_elements, degree : tuple[int, int]
        Spline data of the meridional mapping (axial, radial).
    """

    def __init__(
        self,
        length: float,
        wall_profile: tuple,
        segments: tuple = (),
        axis_radius: float = 1e-3,
        tor_period: int = 360,
        num_elements: tuple = (16, 8),
        degree: tuple = (3, 3),
    ):
        if length <= 0.0 or axis_radius <= 0.0:
            raise ValueError("length and axis_radius must be positive.")
        z_wall, r_wall = (xp.asarray(values, dtype=float) for values in wall_profile)
        if z_wall.ndim != 1 or z_wall.shape != r_wall.shape or z_wall.size < 2:
            raise ValueError("wall_profile must be a pair of equally sized 1D node arrays.")
        if xp.any(xp.diff(z_wall) <= 0.0) or not xp.isclose(z_wall[0], 0.0) or not xp.isclose(z_wall[-1], length):
            raise ValueError("wall_profile z nodes must be strictly increasing and span [0, length].")
        if xp.any(r_wall <= axis_radius):
            raise ValueError("The wall must stay outside the axis cylinder.")
        segments = tuple(ElectrodeSegment(**s) if isinstance(s, dict) else s for s in segments)
        for segment in segments:
            if segment.side != "upper":
                raise ValueError("Axisymmetric electrodes lie on the outer wall (side='upper').")
            if segment.x0 < 0.0 or segment.x1 > length:
                raise ValueError("Electrode segments must lie inside [0, length].")

        self.length = float(length)
        self.wall_profile = (z_wall, r_wall)
        self.segments = segments
        self.axis_radius = float(axis_radius)
        self.tor_period = int(tor_period)
        self.params = copy.deepcopy(locals())

        def R(eta1, eta2):
            wall = xp.interp(length * eta1, z_wall, r_wall)
            return axis_radius + eta2 * (wall - axis_radius)

        def Z(eta1, eta2):
            return length * eta1 + 0.0 * eta2

        cr, cz = interp_mapping(num_elements, degree, (False, False), R, Z)
        super().__init__(
            num_elements=num_elements,
            degree=degree,
            spl_kind=(False, False),
            cx=cr,
            cy=cz,
            tor_period=tor_period,
        )
        # a wedge, not a full revolution: the wedge faces are symmetry planes, not periodic
        self.periodic_eta3 = False
        self.pole = False

    def wall_radius(self, z):
        """Outer wall radius at axial position ``z``."""
        return xp.interp(z, *self.wall_profile)

    @property
    def wedge_fraction(self) -> float:
        """Fraction of the full revolution represented by the wedge."""
        return 1.0 / self.tor_period

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": {
                "length": self.length,
                "wall_profile": [self.wall_profile[0].tolist(), self.wall_profile[1].tolist()],
                "segments": [asdict(segment) for segment in self.segments],
                "axis_radius": self.axis_radius,
                "tor_period": self.tor_period,
            },
        }


def meridional(domain, eta):
    """Physical ``(z, r)`` of logical points ``eta`` (..., 3) in an axisymmetric wedge."""
    eta = xp.asarray(eta, dtype=float)
    flat = eta.reshape(-1, 3)
    xyz = xp.asarray(domain(xp.clip(flat, 0.0, 1.0), change_out_order=True, remove_outside=False)).reshape(-1, 3)
    r = xp.hypot(xyz[:, 0], xyz[:, 1])
    return xyz[:, 2].reshape(eta.shape[:-1]), r.reshape(eta.shape[:-1])
