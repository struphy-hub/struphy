"""Domain package for :class:`struphy.geometry.domains.shafranov_shift_cylinder.shafranov_shift_cylinder.ShafranovShiftCylinder`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

from typing import TYPE_CHECKING

__all__ = ["ShafranovShiftCylinder"]

if TYPE_CHECKING:  # IDEs and type checkers see the class; no import at run time (see above)
    from struphy.geometry.domains.shafranov_shift_cylinder.shafranov_shift_cylinder import ShafranovShiftCylinder


def __getattr__(name):
    if name == "ShafranovShiftCylinder":
        from struphy.geometry.domains.shafranov_shift_cylinder.shafranov_shift_cylinder import ShafranovShiftCylinder

        return ShafranovShiftCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
