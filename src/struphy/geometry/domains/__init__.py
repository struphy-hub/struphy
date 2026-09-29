"""Mapped domains (single patch).

Package providing mapping classes for single-patch geometries used by Struphy,
one class per module: spline mappings such as `Tokamak`, `GVECunit`, `DESCunit`,
`IGAPolarCylinder` and `IGAPolarTorus`, and analytical mappings such as `Cuboid`
and `HollowTorus`. Mappings transform reference coordinates to Cartesian
coordinates and integrate with spline-based grid constructions and field-line tracing.
All classes are importable from `struphy.geometry.domains` directly.
"""

import importlib
from typing import TYPE_CHECKING

# class name -> module defining it, resolved on first access by __getattr__ below
_LAZY_IMPORTS = {
    "Colella": "struphy.geometry.domains.colella",
    "Cuboid": "struphy.geometry.domains.cuboid",
    "DESCunit": "struphy.geometry.domains.desc_unit",
    "GVECunit": "struphy.geometry.domains.gvec_unit",
    "HollowCylinder": "struphy.geometry.domains.hollow_cylinder",
    "HollowTorus": "struphy.geometry.domains.hollow_torus",
    "IGAPolarCylinder": "struphy.geometry.domains.iga_polar_cylinder",
    "IGAPolarTorus": "struphy.geometry.domains.iga_polar_torus",
    "Orthogonal": "struphy.geometry.domains.orthogonal",
    "PoweredEllipticCylinder": "struphy.geometry.domains.powered_elliptic_cylinder",
    "ShafranovDshapedCylinder": "struphy.geometry.domains.shafranov_dshaped_cylinder",
    "ShafranovShiftCylinder": "struphy.geometry.domains.shafranov_shift_cylinder",
    "ShafranovSqrtCylinder": "struphy.geometry.domains.shafranov_sqrt_cylinder",
    "Tokamak": "struphy.geometry.domains.tokamak",
}

if TYPE_CHECKING:  # static analysis and IDEs see the eager imports
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


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
