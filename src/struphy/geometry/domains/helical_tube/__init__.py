"""Domain package for :class:`struphy.geometry.domains.helical_tube.helical_tube.HelicalTube`.

The class is imported lazily on first attribute access (see :mod:`struphy.geometry.domains.hollow_cylinder`).
"""

__all__ = ["HelicalTube"]


def __getattr__(name):
    if name == "HelicalTube":
        from struphy.geometry.domains.helical_tube.helical_tube import HelicalTube

        return HelicalTube
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
