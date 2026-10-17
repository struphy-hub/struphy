"""Struphy objects of the 2D studies of the talk: the domains, a 2D Derham complex of the cross-section and the polar
complexes built on it.

All demos use these helpers, so that the figures of the talk are computed with Struphy's spline spaces
(:class:`~struphy.polar.homotopy.derham_spaces.DerhamSpace1D`), Struphy domains
(:class:`~struphy.polar.homotopy.derham_blocks.DomainCrossSection`) and Struphy's coefficient ordering.
"""

import numpy as np

from struphy.polar.homotopy.layout import STRUPHY


# ---------------------------------------------------------------------------------------------------------- domains
def perturbed_disk(eps=0.2, k=3):
    """The perturbed disk of the talk, ``R = x (1 + 0.2 x cos(6 pi y))``."""
    from struphy.geometry import domains

    return domains.PerturbedDiskCylinder(R1=1.0, eps=eps, k=k, Lz=1.0)


def star_disk(eps=0.2, k=3):
    """The star-shaped disk of the regularity study, ``R = x (1 + 0.2 cos(6 pi y))``."""
    from struphy.geometry import domains

    return domains.PerturbedDiskCylinder(R1=1.0, eps=eps, k=k, Lz=1.0, power=0.0)


def unit_disk():
    from struphy.geometry import domains

    return domains.HollowCylinder(a1=0.0, a2=1.0, Lz=1.0)


def rounded_square(eps=0.3):
    from struphy.geometry import domains

    return domains.RoundedSquareCylinder(eps=eps, Lz=1.0)


# ---------------------------------------------------------------------------------------------------------- 1D
def spline_space_1d(degree, cells, periodic=False):
    """The 1D B-spline space (0-forms, with its M-spline 1-forms) of a Struphy Derham complex (feectools spaces)."""
    from feectools.fem.splines import SplineSpace

    from struphy.polar.homotopy.derham_spaces import DerhamSpace1D

    grid = np.linspace(0, 1, cells + 1)
    B = SplineSpace(degree, grid=grid, periodic=periodic)
    M = SplineSpace(degree - 1, grid=grid, periodic=periodic, basis="M", pads=degree, parent_multiplicity=1)
    return DerhamSpace1D(B, M)


def trigonometric_space_1d(n):
    """The ``n`` (odd) nodal trigonometric functions of :mod:`struphy.feec.trig_space` (with their 1-forms)."""
    from struphy.feec.trig_space import TrigonometricSpace
    from struphy.polar.homotopy.derham_spaces import DerhamSpace1D

    return DerhamSpace1D(TrigonometricSpace(n, "B"), TrigonometricSpace(n, "M"))


# ---------------------------------------------------------------------------------------------------------- Derham
def _pair(value):
    return (value, value) if np.isscalar(value) else tuple(value)


def derham_2d(degree=3, cells=(5, 12), domain=None, angular="bspline", nquads=None):
    """2D Derham complex of the cross-section (one cell in ``eta3``); ``angular="trig"`` uses ``cells[1]``
    trigonometric functions in ``eta2`` (odd) instead of periodic B-splines."""
    from struphy.polar.homotopy.derham_blocks import poloidal_derham

    if angular not in ("bspline", "trig"):
        raise ValueError("angular must be 'bspline' or 'trig'")
    if angular == "trig":
        return poloidal_derham(_pair(degree), _pair(cells), domain=domain, nquads=nquads, angular="trig")
    return poloidal_derham(_pair(degree), _pair(cells), domain=domain, nquads=nquads)


def factors(degree=3, cells=(5, 12), angular="bspline"):
    """The radial and angular 0-form spaces of the 2D Derham complex."""
    from struphy.polar.homotopy.derham_spaces import derham_factors

    radial, ang, _ = derham_factors(derham_2d(degree, cells, angular=angular))
    return radial, ang


def polar_complex(
    regularity,
    degree=3,
    cells=(5, 12),
    domain=None,
    angular="bspline",
    boundary="free",
    enrichment="cochain_chain",
    cross_section="domain",
    polynomial_projection="commuting",
    check_assumptions=True,
):
    """The polar complex of regularity ``r`` on the 2D Derham complex of ``domain`` (default: the perturbed disk)."""
    from struphy.polar.homotopy.derham_blocks import derham_polar_complex

    domain = perturbed_disk() if domain is None else domain
    derham = derham_2d(degree, cells, domain=domain, angular=angular)
    return derham_polar_complex(
        derham,
        domain,
        regularity,
        boundary=boundary,
        enrichment=enrichment,
        polynomial_projection=polynomial_projection,
        cross_section=cross_section,
        check_assumptions=check_assumptions,
    )


def tensor_space(k, degree=3, cells=(5, 12), domain=None, angular="bspline"):
    """The tensor-product ``k``-forms of the 2D Derham complex pushed forward to the cross-section of ``domain``."""
    from struphy.polar.homotopy.derham_blocks import DomainCrossSection
    from struphy.polar.homotopy.forms2d import PhysicalFormSpace2D

    domain = perturbed_disk() if domain is None else domain
    radial, ang = factors(degree, cells, angular)
    return PhysicalFormSpace2D.from_factors(k, radial, ang, DomainCrossSection(domain), STRUPHY)


def cross_section(domain=None):
    """The 2D map of the cross-section ``eta3 = 0`` of a Struphy domain (for plots)."""
    from struphy.polar.homotopy.derham_blocks import DomainCrossSection

    return DomainCrossSection(perturbed_disk() if domain is None else domain)
