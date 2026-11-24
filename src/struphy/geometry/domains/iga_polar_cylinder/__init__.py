"""Domain package for :class:`struphy.geometry.domains.iga_polar_cylinder.iga_polar_cylinder.IGAPolarCylinder`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

from typing import TYPE_CHECKING

__all__ = ["IGAPolarCylinder"]

if TYPE_CHECKING:  # IDEs and type checkers see the class; no import at run time (see above)
    from struphy.geometry.domains.iga_polar_cylinder.iga_polar_cylinder import IGAPolarCylinder


def __getattr__(name):
    if name == "IGAPolarCylinder":
        from struphy.geometry.domains.iga_polar_cylinder.iga_polar_cylinder import IGAPolarCylinder

        return IGAPolarCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
