"""High-order polar FEEC by homotopy operators (serial core).

A port of PolarFEEC.jl (R. Remmerswaal) to struphy: 1D B-spline and trigonometric spaces, 2D tensor-product and
physical form spaces, the high-order polar subcomplex with commuting projections and (boundary-compatible) homotopy
operators, and the homotopy-based solvers.
"""

from struphy.polar.homotopy.complex2d import PolarComplex2D, PolarFormSpace, cumulative_sum_matrix
from struphy.polar.homotopy.forms2d import FormSpace2D, PhysicalFormSpace2D
from struphy.polar.homotopy.mappings import CoordinateTransformation2D, DiskMapping, perturbed_disk
from struphy.polar.homotopy.polynomials import (
    PolynomialBasis,
    center_shifted_basis,
    monomial_basis,
    polynomial_dimension,
    zernike_basis,
)
from struphy.polar.homotopy.spaces1d import ClampedBSplineSpace, PeriodicBSplineSpace, TrigonometricSpace

__all__ = [
    "ClampedBSplineSpace",
    "CoordinateTransformation2D",
    "DiskMapping",
    "FormSpace2D",
    "PeriodicBSplineSpace",
    "PhysicalFormSpace2D",
    "PolarComplex2D",
    "PolarFormSpace",
    "PolynomialBasis",
    "TrigonometricSpace",
    "center_shifted_basis",
    "cumulative_sum_matrix",
    "monomial_basis",
    "perturbed_disk",
    "polynomial_dimension",
    "zernike_basis",
]
