"""High-order polar FEEC by homotopy operators.

A port of PolarFEEC.jl (R. Remmerswaal) to Struphy: the high-order polar subcomplex of regularity ``r`` with commuting
projections and (boundary-compatible) homotopy operators, built on the 1D spaces of a Struphy Derham complex
(B-splines, or trigonometric functions in ``eta2``) and on Struphy domains, and the homotopy-based solvers.
"""

from struphy.polar.homotopy.complex2d import PolarComplex2D, PolarFormSpace, cumulative_sum_matrix
from struphy.polar.homotopy.derham_blocks import DomainCrossSection, derham_polar_complex, poloidal_derham
from struphy.polar.homotopy.derham_spaces import DerhamSpace1D, derham_factors
from struphy.polar.homotopy.forms2d import FormSpace2D, PhysicalFormSpace2D
from struphy.polar.homotopy.polynomials import (
    PolynomialBasis,
    center_shifted_basis,
    monomial_basis,
    polynomial_dimension,
    zernike_basis,
)

__all__ = [
    "DerhamSpace1D",
    "DomainCrossSection",
    "FormSpace2D",
    "PhysicalFormSpace2D",
    "PolarComplex2D",
    "PolarFormSpace",
    "PolynomialBasis",
    "center_shifted_basis",
    "cumulative_sum_matrix",
    "derham_factors",
    "derham_polar_complex",
    "monomial_basis",
    "poloidal_derham",
    "polynomial_dimension",
    "zernike_basis",
]
