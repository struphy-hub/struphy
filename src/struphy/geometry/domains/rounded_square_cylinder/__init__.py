"""Domain package for :class:`struphy.geometry.domains.rounded_square_cylinder.rounded_square_cylinder.RoundedSquareCylinder`.

The class is imported lazily on first attribute access (see :mod:`struphy.geometry.domains.hollow_cylinder`).
"""

__all__ = ["RoundedSquareCylinder"]


def __getattr__(name):
    if name == "RoundedSquareCylinder":
        from struphy.geometry.domains.rounded_square_cylinder.rounded_square_cylinder import RoundedSquareCylinder

        return RoundedSquareCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
