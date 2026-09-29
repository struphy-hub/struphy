"""Domain package for :class:`struphy.geometry.domains.cuboid.cuboid.Cuboid`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

__all__ = ["Cuboid"]


def __getattr__(name):
    if name == "Cuboid":
        from struphy.geometry.domains.cuboid.cuboid import Cuboid

        return Cuboid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
