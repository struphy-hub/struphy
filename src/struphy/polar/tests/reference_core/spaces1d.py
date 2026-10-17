"""One-dimensional 0-form spaces and their derivative-scaled (1-form) companions.

Port of ``src/spaces1d.jl`` and ``src/quadrature.jl`` of PolarFEEC.jl (R. Remmerswaal). All spaces live on the
logical unit interval and all indices are zero-based (the Julia code is one-based):

* clamped and periodic B-splines satisfy ``d/dx B_i = C_{i-1} - C_i``, so the coefficient derivative is the forward
  difference ``(du)_i = u_{i+1} - u_i`` (periodically wrapped for periodic splines);
* trigonometric spaces are ordered ``1, sin(2 pi x), cos(2 pi x), sin(4 pi x), ...``;
* the geometric degrees of freedom are values at Greville points (0-forms) and integrals over Greville edges
  (1-forms).
"""

from functools import cached_property

import numpy as np
import scipy.sparse as sp

GAUSS_RULES = {}


def gauss_rule(order):
    """Gauss-Legendre rule with ``max(order, 2)`` points on ``[-1, 1]``."""
    n = max(int(order), 2)
    if n not in GAUSS_RULES:
        GAUSS_RULES[n] = np.polynomial.legendre.leggauss(n)
    return GAUSS_RULES[n]


def _check_additional_order(order):
    if int(order) != order or order < 0:
        raise ValueError("additional quadrature order must be a nonnegative integer")
    return int(order)


def mass_additional_order(coefficient, requested):
    """Additional quadrature order of a mass matrix: 0 without and 2 (default) with a coefficient."""
    if coefficient is None:
        if requested not in (None, 0):
            raise ValueError("additional_quadrature_order must be zero when no coefficient is supplied")
        return 0
    return 2 if requested is None else _check_additional_order(requested)


class Space1D:
    """Base class of the 1D spaces on the logical unit interval."""

    #: whether the space is periodic
    periodic = False
    #: whether the space is a (scaled) B-spline space
    is_bspline = True
    #: whether the space is a scaled (1-form) space
    is_scaled = False

    def __len__(self):
        return self.n

    @property
    def parent(self):
        """The unscaled parent space (``self`` for unscaled spaces)."""
        return self

    # ------------------------------------------------------------------ quadrature
    def integration_breaks(self):
        """Breakpoints at which quadrature rules are split."""
        return np.linspace(0.0, 1.0, self.parent.cells + 1)

    def minimal_quadrature_order(self):
        """Number of Gauss points per cell that integrates products of two basis functions exactly."""
        return self.local_degree + 1

    def quadrature_order(self, additional_order):
        return self.minimal_quadrature_order() + _check_additional_order(additional_order)

    def quadrature(self, order):
        """Global quadrature points and weights on ``[0, 1]`` with ``order`` Gauss points per cell."""
        values, weights = gauss_rule(order)
        breaks = self.integration_breaks()
        a, b = breaks[:-1, None], breaks[1:, None]
        points = (a + b) / 2 + (b - a) / 2 * values[None, :]
        return points.ravel(), ((b - a) / 2 * weights[None, :]).ravel()

    def basis_matrix(self, x):
        """Sparse matrix of basis values with rows for the points ``x`` and columns for the basis functions."""
        raise NotImplementedError

    # ------------------------------------------------------------------ degrees of freedom
    def greville_points(self):
        raise NotImplementedError

    def edge_bounds(self):
        """Greville edges ``(a, b)`` of the 1-form degrees of freedom (wrapped by one period if periodic)."""
        points = self.greville_points()
        if self.periodic:
            nxt = np.roll(points, -1)
            nxt = np.where(nxt <= points, nxt + 1.0, nxt)
            return np.stack([points, nxt], axis=1)
        return np.stack([points[:-1], points[1:]], axis=1)

    def integration_segments(self, a, b):
        """Split ``[a, b]`` into segments inside ``[0, 1]`` (periodic spaces wrap around)."""
        if not self.periodic:
            return [(a, b)]
        width = b - a
        left = a % 1.0
        right = left + width
        if right <= 1 + 100 * np.finfo(float).eps:
            return [(left, min(right, 1.0))]
        return [(left, 1.0), (0.0, right - 1.0)]

    def edge_rule(self, a, b, values, weights):
        """Quadrature points and weights on ``[a, b]``, split at the element boundaries."""
        breaks = self.integration_breaks()
        points, rule_weights = [], []
        for left_segment, right_segment in self.integration_segments(a, b):
            first = np.clip(np.searchsorted(breaks, left_segment, side="right") - 1, 0, len(breaks) - 2)
            last = np.clip(np.searchsorted(breaks, right_segment, side="left") - 1, first, len(breaks) - 2)
            for cell in range(first, last + 1):
                left = max(left_segment, breaks[cell])
                right = min(right_segment, breaks[cell + 1])
                if left >= right:
                    continue
                scale, midpoint = (right - left) / 2, (right + left) / 2
                points.append(midpoint + scale * values)
                rule_weights.append(scale * weights)
        if not points:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(points), np.concatenate(rule_weights)

    def gdof_rules(self, additional_order=2):
        """For every geometric DOF, the quadrature points and weights that realize it.

        A single unit-weight point at a Greville point for 0-form spaces, and a Gauss rule on the Greville edge (split
        at the element boundaries) for scaled spaces.
        """
        if not self.is_scaled:
            return [(np.array([x]), np.array([1.0])) for x in self.greville_points()]
        values, weights = gauss_rule(self.quadrature_order(additional_order))
        return [self.edge_rule(a, b, values, weights) for a, b in self.parent_edges()]

    def gdof_rule_matrix(self, additional_order=2):
        """Points of all GDOF rules and the sparse matrix ``R`` with ``dofs = R @ f(points)``."""
        rules = self.gdof_rules(additional_order)
        points = np.concatenate([r[0] for r in rules])
        rows = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rules)])
        weights = np.concatenate([r[1] for r in rules])
        R = sp.csr_matrix((weights, (rows, np.arange(len(points)))), shape=(len(rules), len(points)))
        return points, R

    def parent_edges(self):
        return self.parent.edge_bounds()

    def geometric_dofs(self, f, additional_order=2):
        """Geometric degrees of freedom of the function ``f`` (vectorized in its argument)."""
        points, R = self.gdof_rule_matrix(additional_order)
        return R @ np.asarray(f(points), dtype=float)

    @cached_property
    def histopolation_matrix(self):
        """Square matrix mapping coefficients to geometric degrees of freedom."""
        if not self.is_scaled:
            return sp.csr_matrix(self.basis_matrix(self.greville_points()))
        return self._edge_integral_matrix()

    def _edge_integral_matrix(self):
        values, weights = gauss_rule((self.local_degree + 2) // 2)
        rows = []
        for a, b in self.parent_edges():
            points, w = self.edge_rule(a, b, values, weights)
            rows.append(w @ self.basis_matrix(points))
        H = sp.csr_matrix(np.vstack(rows))
        H.eliminate_zeros()
        return H

    def commuting_projection(self, f, additional_order=2):
        return sp.linalg.spsolve(sp.csc_matrix(self.histopolation_matrix), self.geometric_dofs(f, additional_order))

    # ------------------------------------------------------------------ integrals and masses
    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        """Sparse mass matrix, exactly integrated without a coefficient."""
        order = self.quadrature_order(mass_additional_order(coefficient, additional_quadrature_order))
        points, weights = self.quadrature(order)
        if coefficient is not None:
            weights = weights * coefficient(points)
        B = sp.csr_matrix(self.basis_matrix(points))
        M = (B.T @ sp.diags(weights) @ B).tocsr()
        return _drop_small(M)

    def inner_product(self, f, additional_order=2):
        points, weights = self.quadrature(self.quadrature_order(additional_order))
        return np.asarray(sp.csr_matrix(self.basis_matrix(points)).T @ (weights * f(points))).ravel()

    @cached_property
    def coefficient_integrals(self):
        """Integrals of the basis functions over ``[0, 1]``."""
        points, weights = self.quadrature(self.local_degree + 2)
        values = np.asarray(sp.csr_matrix(self.basis_matrix(points)).T @ weights).ravel()
        values[np.abs(values) < 100 * np.finfo(float).eps] = 0.0
        return values


def _drop_small(M):
    M = sp.csr_matrix(M)
    if M.nnz:
        scale = np.max(np.abs(M.data))
        M.data[np.abs(M.data) < 100 * np.finfo(float).eps * scale] = 0.0
        M.eliminate_zeros()
    return M


# ====================================================================== B-spline evaluation
def _de_boor(knots, degree, span, x):
    """Values of the ``degree + 1`` B-splines active on knot span ``span`` (zero-based, vectorized in ``x``).

    Returns an array of shape ``(len(x), degree + 1)`` whose column ``r`` belongs to basis function ``span - degree + r``.
    """
    x = np.asarray(x, dtype=float)
    N = np.zeros((x.size, degree + 1))
    N[:, 0] = 1.0
    left = np.zeros((x.size, degree + 1))
    right = np.zeros((x.size, degree + 1))
    for j in range(1, degree + 1):
        left[:, j] = x - knots[span + 1 - j]
        right[:, j] = knots[span + j] - x
        saved = np.zeros(x.size)
        for r in range(j):
            temp = N[:, r] / (right[:, r + 1] + left[:, j - r])
            N[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        N[:, j] = saved
    return N


class ClampedBSplineSpace(Space1D):
    """Uniform open (clamped) B-spline space of ``degree`` on ``cells`` cells in ``[0, 1]``."""

    def __init__(self, degree, cells):
        if degree < 0 or cells <= 0:
            raise ValueError("degree must be nonnegative and cells positive")
        self.degree = int(degree)
        self.cells = int(cells)
        h = 1.0 / cells
        self.knots = np.concatenate([np.zeros(degree + 1), h * np.arange(1, cells), np.ones(degree + 1)])
        self.n = cells + degree

    def __repr__(self):
        return f"ClampedBSplineSpace({self.degree}, {self.cells})"

    def __eq__(self, other):
        return type(other) is type(self) and (other.degree, other.cells) == (self.degree, self.cells)

    def __hash__(self):
        return hash((type(self), self.degree, self.cells))

    @property
    def local_degree(self):
        return self.degree

    def active_cells(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(np.floor(x * self.cells).astype(int), 0, self.cells - 1)

    def basis_matrix(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        cells = self.active_cells(x)
        p = self.degree
        values = np.zeros((x.size, p + 1))
        for cell in np.unique(cells):
            mask = cells == cell
            values[mask] = _de_boor(self.knots, p, cell + p, x[mask])
        values[(x < 0) | (x > 1)] = 0.0
        rows = np.repeat(np.arange(x.size), p + 1)
        columns = (cells[:, None] + np.arange(p + 1)[None, :]).ravel()
        return sp.csr_matrix((values.ravel(), (rows, columns)), shape=(x.size, self.n))

    def greville_points(self):
        p = self.degree
        if p == 0:
            return np.linspace(0.0, 1.0, self.n)
        return np.array([self.knots[j + 1 : j + p + 1].sum() / p for j in range(self.n)])

    def derivative_matrix(self):
        m = self.n - 1
        rows = np.repeat(np.arange(m), 2)
        columns = np.stack([np.arange(m), np.arange(1, m + 1)], axis=1).ravel()
        values = np.tile([-1.0, 1.0], m)
        return sp.csr_matrix((values, (rows, columns)), shape=(m, self.n))

    @cached_property
    def scaled(self):
        if self.degree == 0:
            raise ValueError("a degree-zero B-spline space has no scaled space")
        return ScaledClampedSpace(self)

    def constant_coefficients(self):
        return np.ones(self.n)


class ScaledClampedSpace(Space1D):
    """``C_i = p / (t_{i+p+1} - t_{i+1}) B^{p-1}_{i+1}``: the 1-form companion of a clamped B-spline space."""

    is_scaled = True

    def __init__(self, parent):
        self._parent = parent
        self.reduced = ClampedBSplineSpace(parent.degree - 1, parent.cells)
        self.n = parent.n - 1
        t, p = parent.knots, parent.degree
        self.scales = np.array([p / (t[i + p + 1] - t[i + 1]) for i in range(self.n)])

    def __repr__(self):
        return f"scaled({self._parent!r})"

    @property
    def parent(self):
        return self._parent

    @property
    def local_degree(self):
        return self.reduced.degree

    def basis_matrix(self, x):
        return sp.csr_matrix(self.reduced.basis_matrix(x) @ sp.diags(self.scales))

    def greville_points(self):
        return self._parent.greville_points()


class PeriodicBSplineSpace(Space1D):
    """Uniform periodic B-spline space of ``degree`` with ``cells`` basis functions.

    Basis function ``i`` (zero-based) is supported on ``[i, i + p + 1] / n``.
    """

    periodic = True

    def __init__(self, degree, cells):
        if degree < 0 or cells <= degree:
            raise ValueError("degree must be nonnegative and periodic cells must exceed the degree")
        self.degree = int(degree)
        self.cells = int(cells)
        self.n = cells
        # extended uniform knots t_k = (k - p) / n
        self.knots = (np.arange(cells + 2 * degree + 1) - degree) / cells

    def __repr__(self):
        return f"PeriodicBSplineSpace({self.degree}, {self.cells})"

    def __eq__(self, other):
        return type(other) is type(self) and (other.degree, other.cells) == (self.degree, self.cells)

    def __hash__(self):
        return hash((type(self), self.degree, self.cells))

    @property
    def local_degree(self):
        return self.degree

    def active_cells(self, x):
        u = np.asarray(x, dtype=float) % 1.0
        return np.clip(np.floor(u * self.cells).astype(int), 0, self.cells - 1)

    def _active(self, x):
        """Cells, values ``(npts, p + 1)`` and zero-based indices of the active B-splines."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        u = x - np.floor(x)
        cells = self.active_cells(u)
        p = self.degree
        values = np.zeros((x.size, p + 1))
        for cell in np.unique(cells):
            mask = cells == cell
            values[mask] = _de_boor(self.knots, p, cell + p, u[mask])
        indices = (cells[:, None] - p + np.arange(p + 1)[None, :]) % self.cells
        return values, indices

    def basis_matrix(self, x):
        values, indices = self._active(x)
        rows = np.repeat(np.arange(values.shape[0]), self.degree + 1)
        return sp.csr_matrix((values.ravel(), (rows, indices.ravel())), shape=(values.shape[0], self.n))

    def greville_points(self):
        n, p = self.n, self.degree
        return ((np.arange(n) + (p + 1) / 2) % n) / n

    def derivative_matrix(self):
        n = self.n
        rows = np.repeat(np.arange(n), 2)
        columns = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).ravel()
        return sp.csr_matrix((np.tile([-1.0, 1.0], n), (rows, columns)), shape=(n, n))

    @cached_property
    def scaled(self):
        if self.degree == 0:
            raise ValueError("a degree-zero periodic B-spline space has no scaled space")
        return ScaledPeriodicSpace(self)

    def constant_coefficients(self):
        return np.ones(self.n)

    def constant_one_form_coefficients(self):
        return np.full(self.n, 1.0 / self.n)


class ScaledPeriodicSpace(Space1D):
    """``C_i = n B^{p-1}_{i+1}``: the 1-form companion of a periodic B-spline space (supported on ``[i+1, i+p+1]/n``)."""

    periodic = True
    is_scaled = True

    def __init__(self, parent):
        self._parent = parent
        self.reduced = PeriodicBSplineSpace(parent.degree - 1, parent.cells)
        self.n = parent.n

    def __repr__(self):
        return f"scaled({self._parent!r})"

    @property
    def parent(self):
        return self._parent

    @property
    def local_degree(self):
        return self.reduced.degree

    def basis_matrix(self, x):
        values, reduced_indices = self.reduced._active(x)
        indices = (reduced_indices - 1) % self.n
        rows = np.repeat(np.arange(values.shape[0]), self.reduced.degree + 1)
        return sp.csr_matrix(
            (self.n * values.ravel(), (rows, indices.ravel())),
            shape=(values.shape[0], self.n),
        )

    def greville_points(self):
        return self._parent.greville_points()


# ====================================================================== trigonometric spaces
class TrigonometricSpace(Space1D):
    """Real Fourier space with ``2 modes + 1`` functions ``1, sin(2 pi x), cos(2 pi x), sin(4 pi x), ...``.

    Zero-based index ``2k - 1`` is ``sin(2 pi k x)`` and ``2k`` is ``cos(2 pi k x)``.
    """

    periodic = True
    is_bspline = False

    def __init__(self, modes):
        if modes < 0:
            raise ValueError("modes must be nonnegative")
        self.modes = int(modes)
        self.n = 2 * modes + 1

    def __repr__(self):
        return f"TrigonometricSpace({self.modes})"

    def __eq__(self, other):
        return type(other) is type(self) and other.modes == self.modes

    def __hash__(self):
        return hash((type(self), self.modes))

    @property
    def local_degree(self):
        return 2 * self.modes

    def integration_breaks(self):
        return np.array([0.0, 1.0])

    def minimal_quadrature_order(self):
        return 2 * self.modes + 1

    def quadrature(self, order):
        """Midpoint rule with ``max(order, 2 modes + 1)`` points (exact for the trigonometric products)."""
        n = max(order, 2 * self.modes + 1)
        return (np.arange(n) + 0.5) / n, np.full(n, 1.0 / n)

    def _values(self, x, scale):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        values = np.empty((x.size, self.n))
        values[:, 0] = 1.0
        for k in range(1, self.modes + 1):
            factor = 2 * np.pi * k if scale else 1.0
            values[:, 2 * k - 1] = factor * np.sin(2 * np.pi * k * x)
            values[:, 2 * k] = factor * np.cos(2 * np.pi * k * x)
        return values

    def basis_matrix(self, x):
        return sp.csr_matrix(self._values(x, False))

    def greville_points(self):
        return np.arange(self.n) / self.n

    def derivative_matrix(self):
        rows, columns, values = [], [], []
        for k in range(1, self.modes + 1):
            sine, cosine = 2 * k - 1, 2 * k
            rows += [cosine, sine]
            columns += [sine, cosine]
            values += [1.0, -1.0]
        return sp.csr_matrix((values, (rows, columns)), shape=(self.n, self.n))

    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        if coefficient is None:
            mass_additional_order(None, additional_quadrature_order)
            diagonal = np.full(self.n, 0.5)
            diagonal[0] = 1.0
            return sp.diags(diagonal).tocsr()
        return super().mass_matrix(coefficient, additional_quadrature_order)

    @cached_property
    def scaled(self):
        return ScaledTrigonometricSpace(self)

    def constant_coefficients(self):
        e = np.zeros(self.n)
        e[0] = 1.0
        return e

    def constant_one_form_coefficients(self):
        return self.constant_coefficients()


class ScaledTrigonometricSpace(TrigonometricSpace):
    """``1, 2 pi k sin(2 pi k x), 2 pi k cos(2 pi k x), ...``: the 1-form companion of a trigonometric space."""

    is_scaled = True

    def __init__(self, parent):
        super().__init__(parent.modes)
        self._parent = parent

    def __repr__(self):
        return f"scaled({self._parent!r})"

    @property
    def parent(self):
        return self._parent

    def parent_edges(self):
        return self.edge_bounds()

    def basis_matrix(self, x):
        return sp.csr_matrix(self._values(x, True))

    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        if coefficient is None:
            mass_additional_order(None, additional_quadrature_order)
            diagonal = np.ones(self.n)
            for index in range(1, self.n):
                frequency = 2 * np.pi * ((index + 1) // 2)
                diagonal[index] = frequency**2 / 2
            return sp.diags(diagonal).tocsr()
        return Space1D.mass_matrix(self, coefficient, additional_quadrature_order)

    def _edge_integral_matrix(self):
        rows = []
        for a, b in self.edge_bounds():
            row = np.empty(self.n)
            segments = self.integration_segments(a, b)
            row[0] = sum(right - left for left, right in segments)
            for k in range(1, self.modes + 1):
                w = 2 * np.pi * k
                row[2 * k - 1] = sum(np.cos(w * left) - np.cos(w * right) for left, right in segments)
                row[2 * k] = sum(np.sin(w * right) - np.sin(w * left) for left, right in segments)
            rows.append(row)
        H = np.vstack(rows)
        H[np.abs(H) < 100 * np.finfo(float).eps] = 0.0
        return sp.csr_matrix(H)

    @cached_property
    def scaled(self):
        raise ValueError("scaled spaces have no scaled companion")
