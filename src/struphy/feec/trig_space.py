r"""Trigonometric 1D spaces for the Derham complex (periodic directions).

The space of trigonometric polynomials of degree ``M`` on ``[0, 1)`` has dimension ``n = 2 M + 1``. Struphy uses the
*nodal* bases on the points :math:`x_i = i / n`:

* 0-forms (``basis="B"``): the cardinal functions (Dirichlet kernels)
  :math:`L_i(x) = \sin(n \pi (x - x_i)) / (n \sin(\pi (x - x_i)))`, :math:`L_i(x_j) = \delta_{ij}`;
* 1-forms (``basis="M"``): the cell histopolants
  :math:`H_i(x) = 1 + \sum_{k=1}^{M} \frac{2 \pi k}{n \sin(\pi k / n)} \cos(2 \pi k (x - x_i - \tfrac{1}{2n}))`,
  :math:`\int_{x_j}^{x_{j+1}} H_i = \delta_{ij}`.

They satisfy :math:`L_i' = H_{i-1} - H_i`, so the coefficient derivative is the forward difference of the B-splines
and the interpolation and histopolation matrices are the identity: the trigonometric spaces form the same 1D
complex as periodic splines, with the geometric degrees of freedom at the nodes and on the cells.

For the FEEC data structures the spaces present themselves as periodic splines on ``n`` cells (the cells
``[x_i, x_{i+1}]``) whose basis functions are all active on every cell: the 0-form space has "degree" ``n`` (``n + 1``
local slots per cell, one of them a zero duplicate), the 1-form space degree ``n - 1``, both with ``n`` pads. The
assembly grids (:class:`TrigonometricAssemblyGrid`) provide the basis values, so the mass-matrix kernels of Struphy
work unchanged. Particles, local projectors and spline-kernel evaluation do not support trigonometric spaces.
"""

import numpy as np
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.fem.grid import FemAssemblyGrid
from feectools.fem.partitioning import partition_coefficients
from feectools.fem.splines import SplineSpace
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.direct_solvers import SparseSolver
from feectools.linalg.stencil import StencilVectorSpace
from scipy.sparse import csc_matrix, identity


# ---------------------------------------------------------------------------------------------------------- bases
def cardinal_values(x, n, nderiv=0):
    """Values (and derivatives) of the cardinal trigonometric functions ``L_i``, shape ``(nderiv + 1, len(x), n)``."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    modes = (n - 1) // 2
    k = np.arange(1, modes + 1)
    t = x[:, None, None] - np.arange(n)[None, :, None] / n  # (points, functions, 1)
    phase = 2 * np.pi * k[None, None, :] * t
    out = np.empty((nderiv + 1, x.size, n))
    for d in range(nderiv + 1):
        factor = (2 * np.pi * k) ** d
        # d-th derivative of cos is cos(. + d pi / 2)
        values = np.sum(factor * np.cos(phase + d * np.pi / 2), axis=2)
        out[d] = (2 * values + (1.0 if d == 0 else 0.0)) / n
    return out


def histopolant_values(x, n, nderiv=0):
    """Values (and derivatives) of the cell histopolants ``H_i``, shape ``(nderiv + 1, len(x), n)``."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    modes = (n - 1) // 2
    k = np.arange(1, modes + 1)
    weight = 2 * np.pi * k / (n * np.sin(np.pi * k / n))
    t = x[:, None, None] - (np.arange(n)[None, :, None] + 0.5) / n
    phase = 2 * np.pi * k[None, None, :] * t
    out = np.empty((nderiv + 1, x.size, n))
    for d in range(nderiv + 1):
        factor = weight * (2 * np.pi * k) ** d
        out[d] = np.sum(factor * np.cos(phase + d * np.pi / 2), axis=2) + (1.0 if d == 0 else 0.0)
    return out


# ---------------------------------------------------------------------------------------------------------- spaces
class TrigonometricSpace(SplineSpace):
    """Nodal trigonometric space with ``n`` (odd) functions on the periodic interval, a 0-form (``basis="B"``) or
    1-form (``basis="M"``) space of the Derham complex (see the module documentation)."""

    is_trigonometric = True

    def __init__(self, n, basis="B"):
        if n % 2 == 0 or n < 3:
            raise ValueError(f"the number of trigonometric functions must be odd and at least 3 (got {n})")
        if basis not in ("B", "M"):
            raise ValueError("basis must be 'B' (0-forms) or 'M' (1-forms)")
        n = int(n)
        self.n_functions = n
        self.modes = (n - 1) // 2
        nodes = np.arange(n + 1) / n
        degree = n if basis == "B" else n - 1
        self._degree = degree
        self._pads = n
        # knots of a uniform periodic spline of the same "degree" (only for code that reads them)
        self._knots = np.arange(-degree, n + degree + 1) / n
        self._periodic = True
        self._multiplicity = 1
        self._parent_multiplicity = 1
        self._dirichlet = (False, False)
        self._basis = basis
        self._nbasis = n
        self._breaks = nodes
        self._ncells = n
        self._greville = nodes[:-1].copy()
        self._ext_greville = nodes.copy()
        self._scaling_array = None
        self._histopolation_grid = nodes.copy()
        domain_decomposition = DomainDecomposition([n], [True])
        cart = CartDecomposition(domain_decomposition, [n], [np.array([0])], [np.array([n - 1])], [self._pads], [1])
        self._coeff_space = StencilVectorSpace(cart)
        self._interpolation_ready = False
        self._histopolation_ready = False
        self._symbolic_space = None

    def __repr__(self):
        return f"TrigonometricSpace(n={self.n_functions}, basis={self._basis!r})"

    def basis_values(self, x, nderiv=0):
        """Basis values and derivatives at the points ``x``, shape ``(nderiv + 1, len(x), n)``."""
        values = cardinal_values if self._basis == "B" else histopolant_values
        return values(np.asarray(x, dtype=float) % 1.0, self.n_functions, nderiv)

    def partner(self):
        """The other space of the 1D complex (1-forms of a 0-form space and vice versa)."""
        return TrigonometricSpace(self.n_functions, "M" if self._basis == "B" else "B")

    def init_interpolation(self, dtype=float):
        # the cardinal functions interpolate at the nodes
        self.imat = np.eye(self.n_functions)
        self._interpolator = SparseSolver(csc_matrix(identity(self.n_functions)))
        self._interpolation_ready = True

    def init_histopolation(self, dtype=float):
        # the histopolants histopolate on the cells
        self.hmat = np.eye(self.n_functions)
        self._histopolator = SparseSolver(csc_matrix(identity(self.n_functions)))
        self._histopolation_ready = True


def is_trigonometric(space):
    return getattr(space, "is_trigonometric", False)


class TrigonometricAssemblyGrid(FemAssemblyGrid):
    """Quadrature data of a trigonometric space on the cells ``[x_i, x_{i+1}]`` (the interface of feectools'
    ``FemAssemblyGrid``): all functions are active on every cell; the 0-form space has one zero duplicate slot."""

    def __init__(self, space, start, end, *, nquads, nderiv=1):
        n = space.n_functions
        u, w = np.polynomial.legendre.leggauss(nquads)
        breaks = space.breaks
        a, b = breaks[:-1, None], breaks[1:, None]
        points = (a + b) / 2 + (b - a) / 2 * u[None, :]
        weights = (b - a) / 2 * w[None, :]
        degree = space.degree
        slots = degree + 1
        values = space.basis_values(points.ravel(), nderiv).reshape(nderiv + 1, n, nquads, n)
        basis = np.zeros((n, slots, nderiv + 1, nquads))
        for cell in range(n):
            for slot in range(min(slots, n)):
                function = (cell + slot) % n
                basis[cell, slot] = values[:, cell, :, function]
        spans = np.arange(n) + degree
        cells = slice(start, end + 1)
        self._num_elements = end - start + 1
        self._num_quad_pts = nquads
        self._spans = spans[cells].copy()
        self._basis = basis[cells].copy()
        self._points = points[cells].copy()
        self._weights = weights[cells].copy()
        self._indices = tuple(range(start, end + 1))
        self._quad_rule_x, self._quad_rule_w = u, w
        self._local_element_start = 0
        self._local_element_end = self._num_elements - 1

    num_elements = property(lambda self: self._num_elements)
    num_quad_pts = property(lambda self: self._num_quad_pts)
    spans = property(lambda self: self._spans)
    basis = property(lambda self: self._basis)
    points = property(lambda self: self._points)
    weights = property(lambda self: self._weights)
    indices = property(lambda self: self._indices)
    quad_rule_x = property(lambda self: self._quad_rule_x)
    quad_rule_w = property(lambda self: self._quad_rule_w)
    local_element_start = property(lambda self: self._local_element_start)
    local_element_end = property(lambda self: self._local_element_end)


class TrigonometricTensorFemSpace(TensorFemSpace):
    """A ``TensorFemSpace`` with trigonometric directions: assembly grids and degree reduction (0-forms to 1-forms)
    of the trigonometric spaces."""

    def get_assembly_grids(self, *nquads):
        assert len(nquads) == self.ldim
        grids = []
        for i, (V, nq) in enumerate(zip(self.spaces, nquads)):
            cache = self._assembly_grids[i]
            if nq not in cache:
                s, e = int(self._element_starts[i]), int(self._element_ends[i])
                if is_trigonometric(V):
                    cache[nq] = TrigonometricAssemblyGrid(V, s, e, nderiv=1, nquads=nq)
                else:
                    cache[nq] = FemAssemblyGrid(V, s, e, nderiv=V.degree, nquads=nq)
            grids.append(cache[nq])
        return tuple(grids)

    def reduce_degree(self, axes, multiplicity=None, basis="B"):
        if isinstance(axes, int):
            axes = [axes]
        trig_axes = [a for a in axes if is_trigonometric(self.spaces[a])]
        other_axes = [a for a in axes if a not in trig_axes]
        if multiplicity is None:
            multiplicity = [self.multiplicity[i] for i in axes]
        elif isinstance(multiplicity, int):
            multiplicity = [multiplicity]
        other_multiplicity = [m for m, a in zip(multiplicity, axes) if a in other_axes]
        reduced = super().reduce_degree(other_axes, other_multiplicity, basis) if other_axes else self
        spaces = list(reduced.spaces)
        for axis in trig_axes:
            spaces[axis] = TrigonometricSpace(self.spaces[axis].n_functions, basis="M")
        v = self.coeff_space
        npts = [s.nbasis for s in spaces]
        global_starts, global_ends = partition_coefficients(v.cart.domain_decomposition, spaces)
        cart = v.cart.reduce_npts(npts, global_starts, global_ends, shifts=[s.multiplicity for s in spaces])
        space = TrigonometricTensorFemSpace(self._domain_decomposition, *spaces, cart=cart, dtype=v.dtype)
        space._interpolation_ready = False
        return space


def tensor_fem_space(domain_decomposition, *spaces, **kwargs):
    """``TensorFemSpace`` of the 1D spaces (with trigonometric support if one of them is trigonometric)."""
    cls = TrigonometricTensorFemSpace if any(is_trigonometric(s) for s in spaces) else TensorFemSpace
    return cls(domain_decomposition, *spaces, **kwargs)
