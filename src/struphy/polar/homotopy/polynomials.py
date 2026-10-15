"""Graded bases of the Cartesian polynomials of total degree at most ``degree``.

Port of ``src/polynomials.jl`` of PolarFEEC.jl. The functions are vectorized: ``f(x, y)`` accepts arrays of physical
coordinates.
"""

from math import comb, factorial

import numpy as np


def polynomial_dimension(degree):
    """Dimension ``(r + 1)(r + 2)/2`` of the polynomials of total degree at most ``r`` (0 for ``r < 0``)."""
    return (degree + 1) * (degree + 2) // 2 if degree >= 0 else 0


class PolynomialBasis:
    """A graded basis: the first ``(q + 1)(q + 2)/2`` functions span the polynomials of degree ``q``, for every
    ``q <= degree``, and the first function is a nonzero constant."""

    def __init__(self, functions, degree):
        if len(functions) != polynomial_dimension(degree):
            raise ValueError(f"a total-degree-{degree} polynomial basis must contain {polynomial_dimension(degree)}")
        self.functions = list(functions)
        self.degree = int(degree)

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index):
        return self.functions[index]

    def __iter__(self):
        return iter(self.functions)


def _constant_one(x, y):
    return np.ones(np.broadcast(np.asarray(x), np.asarray(y)).shape)


def center_shifted_basis(basis):
    """The basis of the polar construction: ``T_1 = 1`` and ``T_l = P_l - P_l(0, 0)``."""
    samples = [basis[0](np.array(a), np.array(b)) for a, b in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.37, 0.29))]
    if not np.allclose(samples, samples[0]) or samples[0] == 0:
        raise ValueError("the first polynomial basis function must be a nonzero constant")
    functions = [_constant_one]
    for f in basis.functions[1:]:
        center = float(f(np.array(0.0), np.array(0.0)))
        functions.append(lambda x, y, f=f, c=center: f(x, y) - c)
    return PolynomialBasis(functions, basis.degree)


def monomial_basis(order):
    """Monomials ``1, x, y, x^2, x y, y^2, ...`` (by total degree, then decreasing power of ``x``)."""
    functions = []
    for total in range(order + 1):
        for ydeg in range(total + 1):
            xdeg = total - ydeg
            functions.append(lambda x, y, a=xdeg, b=ydeg: np.asarray(x, dtype=float) ** a * np.asarray(y) ** b)
    return PolynomialBasis(functions, order)


def _zernike_radial_coefficients(n, m):
    half = (n - m) // 2
    return [
        ((-1) ** s * factorial(n - s) / (factorial(s) * factorial((n + m) // 2 - s) * factorial(half - s)), half - s)
        for s in range(half + 1)
    ]


def _zernike_function(n, index, radius):
    m = abs(index)
    coefficients = _zernike_radial_coefficients(n, m)
    use_sine = index < 0
    normalization = np.sqrt((1 if m == 0 else 2) * (n + 1) / np.pi) / radius

    def zernike(x, y):
        sx, sy = np.asarray(x) / radius, np.asarray(y) / radius
        if m == 0:
            harmonic = np.ones(np.broadcast(sx, sy).shape)
        else:
            # real/imaginary parts of (sx + i sy)^m, written with binomials (complex-step safe)
            terms = [comb(m, j) * sx ** (m - j) * sy**j for j in range(m + 1)]
            if use_sine:
                harmonic = sum(terms[j] * (-1) ** ((j - 1) // 2) for j in range(1, m + 1, 2))
            else:
                harmonic = sum(terms[j] * (-1) ** (j // 2) for j in range(0, m + 1, 2))
        r2 = sx**2 + sy**2
        result = sum(c * r2**power for c, power in coefficients) * harmonic
        return normalization * result

    return zernike


def zernike_basis(order, radius=1.0, shifted=False):
    """Real Zernike polynomials through radial degree ``order``, ordered by ``n``, then ``m = -n, -n + 2, ..., n``.

    Negative ``m`` is the sine mode. Unnormalised, the ordering is ``1, y, x, 2xy, 2(x^2 + y^2) - 1, x^2 - y^2, ...``.
    Each function has unit L2 norm on the disk of the given ``radius``. With ``shifted=True`` the constant mode is
    ``1`` and the value at the origin is subtracted from the other modes.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    functions = []
    for n in range(order + 1):
        for index in range(-n, n + 1, 2):
            f = _zernike_function(n, index, float(radius))
            if shifted:
                if n == 0:
                    f = _constant_one
                else:
                    center = float(f(np.array(0.0), np.array(0.0)))
                    f = (lambda g, c: lambda x, y: g(x, y) - c)(f, center)
            functions.append(f)
    return PolynomialBasis(functions, order)
