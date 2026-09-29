"""Mapped domains (single patch).

Package providing mapping classes for single-patch geometries used by Struphy,
one class per module: spline mappings such as `Tokamak`, `GVECunit`, `DESCunit`,
`IGAPolarCylinder` and `IGAPolarTorus`, and analytical mappings such as `Cuboid`
and `HollowTorus`. Mappings transform reference coordinates to Cartesian
coordinates and integrate with spline-based grid constructions and field-line tracing.
All classes are importable from `struphy.geometry.domains` directly.
"""

from struphy.geometry.domains.colella import Colella
from struphy.geometry.domains.cuboid import Cuboid
from struphy.geometry.domains.desc_unit import DESCunit
from struphy.geometry.domains.gvec_unit import GVECunit
from struphy.geometry.domains.hollow_cylinder import HollowCylinder
from struphy.geometry.domains.hollow_torus import HollowTorus
from struphy.geometry.domains.iga_polar_cylinder import IGAPolarCylinder
from struphy.geometry.domains.iga_polar_torus import IGAPolarTorus
from struphy.geometry.domains.orthogonal import Orthogonal
from struphy.geometry.domains.powered_elliptic_cylinder import PoweredEllipticCylinder
from struphy.geometry.domains.shafranov_dshaped_cylinder import ShafranovDshapedCylinder
from struphy.geometry.domains.shafranov_shift_cylinder import ShafranovShiftCylinder
from struphy.geometry.domains.shafranov_sqrt_cylinder import ShafranovSqrtCylinder
from struphy.geometry.domains.tokamak import Tokamak

__all__ = [
    "Colella",
    "Cuboid",
    "DESCunit",
    "GVECunit",
    "HollowCylinder",
    "HollowTorus",
    "IGAPolarCylinder",
    "IGAPolarTorus",
    "Orthogonal",
    "PoweredEllipticCylinder",
    "ShafranovDshapedCylinder",
    "ShafranovShiftCylinder",
    "ShafranovSqrtCylinder",
    "Tokamak",
]
