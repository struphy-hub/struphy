"""Frozen reference of the PolarFEEC.jl port (test-only).

The 1D B-spline and trigonometric spaces and the disk mappings of the original port. The library builds the polar
complexes on Struphy's Derham spaces and domains (:mod:`struphy.polar.homotopy.derham_blocks`); the tests use this
reference to check them against an independent implementation and to test the homotopy algebra on arbitrary
analytic mappings (``DiskMapping(radius=...)``).
"""

from struphy.polar.tests.reference_core.mappings import (
    CoordinateTransformation2D,
    DiskMapping,
    complex_step,
    perturbed_disk,
)
from struphy.polar.tests.reference_core.spaces1d import (
    ClampedBSplineSpace,
    PeriodicBSplineSpace,
    TrigonometricSpace,
)

__all__ = [
    "ClampedBSplineSpace",
    "CoordinateTransformation2D",
    "DiskMapping",
    "PeriodicBSplineSpace",
    "TrigonometricSpace",
    "complex_step",
    "perturbed_disk",
]
