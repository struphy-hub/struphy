"""The 1D spline spaces of a Struphy :class:`~struphy.feec.psydac_derham.Derham` complex, with the data of the polar
construction.

:class:`DerhamSpace1D` wraps a feectools ``SplineSpace`` of the Derham complex (B-splines for 0-forms, M-splines for
1-forms, in Struphy's ordering) and provides what the polar algebra needs: basis values
(``feectools.core.bsplines.collocation_matrix``), the coefficient derivative
(:func:`~struphy.eigenvalue_solvers.derivatives.grad_1d_matrix`), the geometric degrees of freedom on Struphy's
projection grids (:func:`~struphy.feec.psydac_derham.get_pts_and_wts`: values at the Greville points, integrals over
the Greville cells split at the breakpoints), integrals and quadrature on the elements.
"""

from functools import cached_property

import numpy as np
import scipy.sparse as sp


def basis_values_1d(space, x):
    """Values ``(len(x), nbasis)`` of the basis functions of a 1D space of the Derham complex: B-/M-splines (periodic
    splines with all their periodic copies, also when there are fewer functions than ``degree + 1``) or the nodal
    trigonometric functions of :mod:`struphy.feec.trig_space`."""
    from feectools.core.bsplines import collocation_matrix

    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if getattr(space, "is_trigonometric", False):
        return space.basis_values(x)[0]
    periodic, n = bool(space.periodic), int(space.nbasis)
    points = x % 1.0 if periodic else x
    # (feectools returns a 1 x 1 matrix for a single point: evaluate at least two)
    padded = points if points.size > 1 else np.concatenate([points, points])
    # the unrolled basis of the (extended) knot vector; periodic copies are folded onto their index
    values = np.asarray(collocation_matrix(space.knots, space.degree, False, space.basis, padded), dtype=float)
    values = values[: points.size]
    if not periodic:
        values[(x < 0) | (x > 1)] = 0.0
        return values
    folded = np.zeros((points.size, n))
    for j in range(values.shape[1]):
        folded[:, j % n] += values[:, j]
    return folded


def _gauss(n):
    return np.polynomial.legendre.leggauss(max(int(n), 1))


def _drop_small(matrix):
    matrix = sp.csr_matrix(matrix)
    matrix.data[np.abs(matrix.data) < 100 * np.finfo(float).eps * max(np.max(np.abs(matrix.data), initial=0.0), 1)] = 0
    matrix.eliminate_zeros()
    return matrix


class DerhamSpace1D:
    """A uni-variate space of the Derham complex (``space.basis`` ``"B"``: 0-forms, ``"M"``: 1-forms): B-/M-splines,
    or the nodal trigonometric functions of :mod:`struphy.feec.trig_space` (``angular_basis="trig"``).

    Parameters
    ----------
    space : feectools.fem.splines.SplineSpace
        The 1D space, e.g. ``derham.H1_1d_serial[d].spaces[0]`` or ``derham.L2_1d_serial[d].spaces[0]``.
    partner : feectools.fem.splines.SplineSpace, optional
        The other space of the 1D complex (the M-splines of a B-spline space and vice versa).
    """

    is_bspline = True

    def __init__(self, space, partner=None):
        self.space = space
        self._partner = partner
        self.n = int(space.nbasis)
        self.degree = int(space.degree)
        self.periodic = bool(space.periodic)
        self.is_scaled = space.basis == "M"
        self.breaks = np.asarray(space.breaks, dtype=float)
        self.cells = len(self.breaks) - 1

    def __repr__(self):
        kind = "periodic" if self.periodic else "clamped"
        return f"DerhamSpace1D({kind} {self.space.basis}-splines of degree {self.degree}, n = {self.n})"

    def __len__(self):
        return self.n

    @property
    def local_degree(self):
        return self.degree

    @cached_property
    def scaled(self):
        """The M-spline (1-form) companion of a B-spline space."""
        if self.is_scaled or self._partner is None:
            raise ValueError("only B-spline spaces with a partner have a scaled (1-form) companion")
        return DerhamSpace1D(self._partner, self.space)

    @cached_property
    def parent(self):
        """The B-spline space of an M-spline space (``self`` for B-splines)."""
        if not self.is_scaled:
            return self
        return DerhamSpace1D(self._partner, self.space)

    # ------------------------------------------------------------------ basis and derivative
    def basis_matrix(self, x):
        """Sparse matrix of basis values with rows for the points ``x`` and columns for the basis functions."""
        return sp.csr_matrix(basis_values_1d(self.space, x))

    def derivative_matrix(self):
        """Coefficient derivative into the M-splines, ``(D u)_i = u_{i+1} - u_i`` (periodically wrapped)."""
        import cunumpy as xp

        from struphy.eigenvalue_solvers.derivatives import grad_1d_matrix

        if self.is_scaled:
            raise ValueError("the derivative of 1-forms is zero in one dimension")
        return sp.csr_matrix(xp.to_numpy(grad_1d_matrix(self.periodic, self.n)))

    def constant_coefficients(self):
        """Coefficients of the constant function one (partition of unity of the B-splines)."""
        if self.is_scaled:
            raise ValueError("use constant_one_form_coefficients for M-splines")
        return np.ones(self.n)

    def constant_one_form_coefficients(self):
        """M-spline coefficients of the constant 1-form ``dx`` (histopolation of one)."""
        target = self if self.is_scaled else self.scaled
        points, R = target.gdof_rule_matrix(0)
        return sp.linalg.spsolve(sp.csc_matrix(target.histopolation_matrix), R @ np.ones(points.size))

    # ------------------------------------------------------------------ degrees of freedom
    def greville_points(self):
        return np.asarray(self.parent.space.greville, dtype=float)

    def gdof_rule_matrix(self, additional_order=2):
        """Points of the geometric degrees of freedom and the sparse matrix ``R`` with ``dofs = R @ f(points)``.

        Values at the Greville points for B-splines; for M-splines integrals over the Greville cells with
        ``degree + 1 + additional_order`` Gauss points on each part of a cell split at the breakpoints (Struphy's
        projection grid of :func:`~struphy.feec.psydac_derham.get_pts_and_wts`)."""
        from struphy.feec.psydac_derham import get_pts_and_wts

        if not self.is_scaled:
            points = self.greville_points()
            return points, sp.identity(self.n, format="csr")
        n_quad = self.degree + 1 + int(additional_order)
        pts, wts, subs = get_pts_and_wts(self.space, 0, self.n - 1, n_quad=n_quad)
        pts, wts, subs = np.asarray(pts), np.asarray(wts), np.asarray(subs)
        cell = np.cumsum(subs == 0) - 1
        rows = np.repeat(cell, pts.shape[1])
        R = sp.csr_matrix((wts.ravel(), (rows, np.arange(pts.size))), shape=(self.n, pts.size))
        return pts.ravel(), R

    @cached_property
    def histopolation_matrix(self):
        """Square matrix from coefficients to geometric degrees of freedom (exact for the spline integrands; the
        identity for the nodal trigonometric functions, which interpolate and histopolate on the nodes)."""
        from struphy.feec.trig_space import is_trigonometric

        if is_trigonometric(self.space):
            return sp.identity(self.n, format="csr")
        points, R = self.gdof_rule_matrix(0)
        return _drop_small(R @ self.basis_matrix(points))

    def geometric_dofs(self, f, additional_order=2):
        points, R = self.gdof_rule_matrix(additional_order)
        return R @ np.asarray(f(points), dtype=float)

    def commuting_projection(self, f, additional_order=2):
        return sp.linalg.spsolve(sp.csc_matrix(self.histopolation_matrix), self.geometric_dofs(f, additional_order))

    # ------------------------------------------------------------------ quadrature, integrals and masses
    def quadrature_order(self, additional_order):
        return self.degree + 1 + int(additional_order)

    def quadrature(self, order):
        """Gauss points and weights with ``order`` points on each element."""
        values, weights = _gauss(order)
        a, b = self.breaks[:-1, None], self.breaks[1:, None]
        points = (a + b) / 2 + (b - a) / 2 * values[None, :]
        return points.ravel(), ((b - a) / 2 * weights[None, :]).ravel()

    @cached_property
    def coefficient_integrals(self):
        """Integrals of the basis functions over ``[0, 1]``."""
        points, weights = self.quadrature(self.degree + 2)
        values = np.asarray(self.basis_matrix(points).T @ weights).ravel()
        values[np.abs(values) < 100 * np.finfo(float).eps] = 0.0
        return values

    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        additional = (
            0 if coefficient is None else (2 if additional_quadrature_order is None else additional_quadrature_order)
        )
        points, weights = self.quadrature(self.quadrature_order(additional))
        if coefficient is not None:
            weights = weights * coefficient(points)
        B = self.basis_matrix(points)
        return _drop_small(B.T @ sp.diags(weights) @ B)

    def inner_product(self, f, additional_order=2):
        points, weights = self.quadrature(self.quadrature_order(additional_order))
        return np.asarray(self.basis_matrix(points).T @ (weights * f(points))).ravel()


def derham_factors(derham):
    """The radial (``eta1``), angular (``eta2``) and axial (``eta3``) 0-form spaces of a Derham complex."""
    H1, L2 = derham.H1_1d_serial, derham.L2_1d_serial
    return tuple(DerhamSpace1D(H1[d].spaces[0], L2[d].spaces[0]) for d in range(3))
