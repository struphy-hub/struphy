"""Domain package for :class:`struphy.geometry.domains.shafranov_dshaped_cylinder.shafranov_dshaped_cylinder.ShafranovDshapedCylinder`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

__all__ = ["ShafranovDshapedCylinder"]


def __getattr__(name):
    if name == "ShafranovDshapedCylinder":
        from struphy.geometry.domains.shafranov_dshaped_cylinder.shafranov_dshaped_cylinder import (
            ShafranovDshapedCylinder,
        )

        return ShafranovDshapedCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
