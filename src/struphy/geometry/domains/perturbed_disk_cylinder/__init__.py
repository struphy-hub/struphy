"""Domain package for :class:`struphy.geometry.domains.perturbed_disk_cylinder.perturbed_disk_cylinder.PerturbedDiskCylinder`.

The class is imported lazily on first attribute access (see :mod:`struphy.geometry.domains.hollow_cylinder`).
"""

__all__ = ["PerturbedDiskCylinder"]


def __getattr__(name):
    if name == "PerturbedDiskCylinder":
        from struphy.geometry.domains.perturbed_disk_cylinder.perturbed_disk_cylinder import PerturbedDiskCylinder

        return PerturbedDiskCylinder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
