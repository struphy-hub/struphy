"""Domain package for :class:`struphy.geometry.domains.hollow_cylinder.hollow_cylinder.HollowCylinder`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

__all__ = ["HollowCylinder"]


def __getattr__(name):
    if name == "HollowCylinder":
        from struphy.geometry.domains.hollow_cylinder.hollow_cylinder import HollowCylinder

        return HollowCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
