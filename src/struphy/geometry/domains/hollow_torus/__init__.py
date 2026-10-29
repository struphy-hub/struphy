"""Domain package for :class:`struphy.geometry.domains.hollow_torus.hollow_torus.HollowTorus`.

The class is imported lazily on first attribute access. This keeps the
package ``__init__`` free of imports of :mod:`struphy.geometry.base`, so that
the pyccel kernel module in this package can be imported from
:mod:`struphy.geometry.evaluation_kernels` (which is itself imported by
``struphy.geometry.base``) without creating a circular import.
"""

from typing import TYPE_CHECKING

__all__ = ["HollowTorus"]

if TYPE_CHECKING:  # IDEs and type checkers see the class; no import at run time (see above)
    from struphy.geometry.domains.hollow_torus.hollow_torus import HollowTorus


def __getattr__(name):
    if name == "HollowTorus":
        from struphy.geometry.domains.hollow_torus.hollow_torus import HollowTorus

        return HollowTorus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
