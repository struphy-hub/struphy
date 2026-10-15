"""Tensor-product form spaces on the logical unit square and their push-forwards to a physical domain.

Port of ``src/tensor_forms.jl`` and ``src/physical_forms.jl`` of PolarFEEC.jl, restricted to two dimensions.

Coefficient layout: components are stacked (``0``-forms: one, ``1``-forms: ``x``- then ``y``-component, ``2``-forms:
one); within a component the radial index ``i`` varies fastest, linear index ``i + j * n_x``. Kronecker products are
written ``kron(A_y, A_x)``. The derivatives are ``G = [I (x) dx; dy (x) I]`` and ``C = [-dy (x) I, I (x) dx]``.
"""

from functools import cached_property

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from struphy.polar.homotopy.spaces1d import _drop_small, mass_additional_order


class FormComponent:
    """One vector-proxy component: its two 1D factors and the mask of scaled directions."""

    def __init__(self, fx, fy, mask):
        self.fx, self.fy, self.mask = fx, fy, mask

    @property
    def factors(self):
        return (self.fx, self.fy)

    @property
    def shape(self):
        return (self.fx.n, self.fy.n)

    @property
    def size(self):
        return self.fx.n * self.fy.n


class FormSpace2D:
    """Logical tensor-product ``k``-form space ``(k = 0, 1, 2)`` built from the 1D 0-form spaces ``x`` and ``y``."""

    def __init__(self, k, x, y):
        if k not in (0, 1, 2):
            raise ValueError("form degree must be 0, 1 or 2")
        self.k, self.x, self.y = k, x, y
        if k == 0:
            self.components = [FormComponent(x, y, (False, False))]
        elif k == 1:
            self.components = [FormComponent(x.scaled, y, (True, False)), FormComponent(x, y.scaled, (False, True))]
        else:
            self.components = [FormComponent(x.scaled, y.scaled, (True, True))]
        self.sizes = [c.size for c in self.components]
        self.offsets = list(np.cumsum([0] + self.sizes[:-1]))
        self.n = int(sum(self.sizes))

    def __repr__(self):
        return f"FormSpace2D({self.k}, {self.x!r}, {self.y!r})"

    def __len__(self):
        return self.n

    def with_degree(self, k):
        return FormSpace2D(k, self.x, self.y)

    @property
    def shape(self):
        """``(n_x, n_y)`` of a single-component space."""
        if len(self.components) != 1:
            raise ValueError("shape is defined for single-component spaces")
        return self.components[0].shape

    def component_range(self, index):
        return slice(self.offsets[index], self.offsets[index] + self.sizes[index])

    # ------------------------------------------------------------------ exterior derivative
    @cached_property
    def derivative_matrix(self):
        """Coefficient matrix of the exterior derivative ``d: V^k -> V^{k+1}``."""
        x, y = self.x, self.y
        if self.k == 0:
            Ix, Iy = sp.identity(x.n), sp.identity(y.n)
            return sp.vstack([sp.kron(Iy, x.derivative_matrix()), sp.kron(y.derivative_matrix(), Ix)]).tocsr()
        if self.k == 1:
            Ix1, Iy1 = sp.identity(x.scaled.n), sp.identity(y.scaled.n)
            return sp.hstack([-sp.kron(y.derivative_matrix(), Ix1), sp.kron(Iy1, x.derivative_matrix())]).tocsr()
        raise ValueError("the top-degree form space has no exterior derivative")

    # ------------------------------------------------------------------ geometric degrees of freedom
    def _component_rules(self, index, additional_order):
        c = self.components[index]
        return c.fx.gdof_rule_matrix(additional_order), c.fy.gdof_rule_matrix(additional_order)

    @cached_property
    def _histopolation_factorizations(self):
        result = []
        for c in self.components:
            result.append(tuple(spla.splu(sp.csc_matrix(f.histopolation_matrix)) for f in c.factors))
        return result

    def histopolation_matrix(self):
        blocks = [sp.kron(c.fy.histopolation_matrix, c.fx.histopolation_matrix) for c in self.components]
        return sp.block_diag(blocks).tocsr()

    def solve_histopolation(self, dofs):
        """Solve ``histopolation_matrix @ u = dofs`` direction by direction (``dofs`` may have several columns)."""
        dofs = np.asarray(dofs, dtype=float)
        vector = dofs.ndim == 1
        dofs = dofs.reshape(self.n, -1)
        result = np.empty_like(dofs)
        for index, c in enumerate(self.components):
            lu_x, lu_y = self._histopolation_factorizations[index]
            nx, ny = c.shape
            block = dofs[self.component_range(index)].reshape(nx, ny, -1, order="F")
            block = lu_x.solve(block.reshape(nx, -1, order="F")).reshape(nx, ny, -1, order="F")
            block = np.moveaxis(block, 1, 0).reshape(ny, -1, order="F")
            block = lu_y.solve(block).reshape(ny, nx, -1, order="F")
            block = np.moveaxis(block, 0, 1)
            result[self.component_range(index)] = block.reshape(nx * ny, -1, order="F")
        return result[:, 0] if vector else result

    def geometric_dofs(self, f, additional_order=2):
        """GDOFs of the logical proxy ``f(x, y)`` (scalar, or a tuple of components for 1-forms)."""
        result = np.empty(self.n)
        for index in range(len(self.components)):
            (px, Rx), (py, Ry) = self._component_rules(index, additional_order)
            X, Y = np.meshgrid(px, py, indexing="ij")
            values = f(X, Y)
            if self.k == 1:
                values = values[index]
            values = np.broadcast_to(np.asarray(values, dtype=float), X.shape)
            result[self.component_range(index)] = (Rx @ values @ Ry.T).ravel(order="F")
        return result

    def commuting_projection(self, f, additional_order=2):
        return self.solve_histopolation(self.geometric_dofs(f, additional_order))

    # ------------------------------------------------------------------ quadrature
    def quadrature_points(self, additional_order, component=None):
        """Tensor quadrature shared by the components (or of one ``component``): 1D points and weights per direction."""
        components = self.components if component is None else [self.components[component]]
        rules = []
        for d in range(2):
            factor = components[0].factors[d]
            order = max(c.factors[d].quadrature_order(additional_order) for c in components)
            rules.append(factor.quadrature(order))
        return rules

    def basis_at(self, px, py, component):
        """Sparse matrix of the component's basis values at the tensor points (``x`` fastest)."""
        c = self.components[component]
        return sp.kron(c.fy.basis_matrix(py), c.fx.basis_matrix(px)).tocsr()

    def evaluate(self, coefficients, x, y):
        """Logical proxy of the element with the given coefficients at the tensor grid ``x (x) y``.

        Returns arrays of shape ``(len(x), len(y))`` (a list of two for 1-forms).
        """
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        values = []
        for index in range(len(self.components)):
            B = self.basis_at(x, y, index)
            values.append((B @ coefficients[self.component_range(index)]).reshape(len(x), len(y), order="F"))
        return values if self.k == 1 else values[0]

    def evaluate_points(self, coefficients, x, y):
        """Logical proxy at the scattered logical points ``(x[k], y[k])`` (a list of two arrays for 1-forms)."""
        x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        values = []
        for index, c in enumerate(self.components):
            C = np.asarray(coefficients[self.component_range(index)]).reshape(c.fx.n, c.fy.n, order="F")
            Bx, By = c.fx.basis_matrix(x), c.fy.basis_matrix(y)
            values.append(np.asarray(By.multiply(Bx @ C).sum(axis=1)).reshape(shape))
        return values if self.k == 1 else values[0]

    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        """Logical mass matrix (Kronecker products of 1D masses without a coefficient)."""
        if coefficient is None:
            mass_additional_order(None, additional_quadrature_order)
            return sp.block_diag([sp.kron(c.fy.mass_matrix(), c.fx.mass_matrix()) for c in self.components]).tocsr()
        additional = mass_additional_order(coefficient, additional_quadrature_order)
        blocks = []
        for index in range(len(self.components)):
            (px, wx), (py, wy) = self.quadrature_points(additional, index)
            X, Y = np.meshgrid(px, py, indexing="ij")
            w = (np.outer(wx, wy) * coefficient(X, Y)).ravel(order="F")
            B = self.basis_at(px, py, index)
            blocks.append(B.T @ sp.diags(w) @ B)
        return _drop_small(sp.block_diag(blocks))


class PhysicalFormSpace2D:
    """Push-forward of a logical form space by a 2D mapping.

    ======  ======================================  ====================
    degree  push-forward of a logical proxy ``v``   metric weight ``W``
    ======  ======================================  ====================
    0       ``v o F^{-1}``                          ``|J|``
    1       ``(J^{-T} v) o F^{-1}`` (covariant)     ``|J| G^{-1}``
    2       ``(v / det J) o F^{-1}``                ``1 / |J|``
    ======  ======================================  ====================

    Physical targets are functions of the physical coordinates ``(X, Y)``; elements are evaluated at logical points.
    """

    def __init__(self, logical, mapping):
        self.logical = logical
        self.mapping = mapping

    @classmethod
    def from_factors(cls, k, x, y, mapping):
        return cls(FormSpace2D(k, x, y), mapping)

    def __repr__(self):
        return f"PhysicalFormSpace2D({self.logical!r}, {type(self.mapping).__name__})"

    @property
    def k(self):
        return self.logical.k

    @property
    def n(self):
        return self.logical.n

    def __len__(self):
        return self.n

    def with_degree(self, k):
        return PhysicalFormSpace2D(self.logical.with_degree(k), self.mapping)

    @property
    def derivative_matrix(self):
        return self.logical.derivative_matrix

    # ------------------------------------------------------------------ Piola maps
    def pullback_values(self, values, x, y):
        """Pull back physical proxy values given at ``F(x, y)`` to logical proxies at ``(x, y)``."""
        if self.k == 0:
            return values
        if self.k == 2:
            return self.mapping.jacobian(x, y) * values
        J = self.mapping.jacobian_matrix(x, y)
        vx, vy = values
        return [J[..., 0, 0] * vx + J[..., 1, 0] * vy, J[..., 0, 1] * vx + J[..., 1, 1] * vy]

    def pullback(self, f):
        """Logical proxy ``(x, y) -> pullback of f`` of a physical target ``f(X, Y)``."""

        def target(x, y):
            X, Y = self.mapping(x, y)
            return self.pullback_values(f(X, Y), x, y)

        return target

    def pushforward_values(self, values, x, y):
        """Push logical proxy values at ``(x, y)`` forward to the physical proxy at ``F(x, y)``."""
        if self.k == 0:
            return values
        if self.k == 2:
            return values / self.mapping.jacobian(x, y)
        J = self.mapping.jacobian_matrix(x, y)
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        ax, ay = values
        # J^{-T} = adj(J)^T / det
        return [(J[..., 1, 1] * ax - J[..., 1, 0] * ay) / det, (-J[..., 0, 1] * ax + J[..., 0, 0] * ay) / det]

    def metric_weight(self, x, y):
        """``|J|``, ``|J| G^{-1}`` (array ``(..., 2, 2)``) or ``1/|J|``."""
        if self.k == 0:
            return np.abs(self.mapping.jacobian(x, y))
        if self.k == 2:
            return 1.0 / np.abs(self.mapping.jacobian(x, y))
        J = self.mapping.jacobian_matrix(x, y)
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        G = np.einsum("...ki,...kj->...ij", J, J)
        Ginv = np.empty_like(G)
        Ginv[..., 0, 0] = G[..., 1, 1]
        Ginv[..., 1, 1] = G[..., 0, 0]
        Ginv[..., 0, 1] = -G[..., 0, 1]
        Ginv[..., 1, 0] = -G[..., 1, 0]
        Ginv /= (det**2)[..., None, None]
        W = np.abs(det)[..., None, None] * Ginv
        return (W + np.swapaxes(W, -1, -2)) / 2

    # ------------------------------------------------------------------ quadrature-based operations
    def _quadrature(self, additional_order):
        """Shared quadrature (coupled 1-form components) or the single component's quadrature."""
        return self.logical.quadrature_points(additional_order)

    def mass_matrix(self, coefficient=None, additional_quadrature_order=None):
        """Physical mass matrix ``int u . v dX`` (``coefficient(X, Y)`` optional)."""
        additional = mass_additional_order(coefficient, additional_quadrature_order)
        (px, wx), (py, wy) = self._quadrature(additional)
        X, Y = np.meshgrid(px, py, indexing="ij")
        w = np.outer(wx, wy)
        if coefficient is not None:
            w = w * coefficient(*self.mapping(X, Y))
        W = self.metric_weight(X, Y)
        bases = [self.logical.basis_at(px, py, c) for c in range(len(self.logical.components))]
        if self.k != 1:
            weights = (w * W).ravel(order="F")
            return _drop_small(bases[0].T @ sp.diags(weights) @ bases[0])
        blocks = [[None, None], [None, None]]
        for a in range(2):
            for b in range(2):
                weights = (w * W[..., a, b]).ravel(order="F")
                blocks[a][b] = bases[a].T @ sp.diags(weights) @ bases[b]
        return _drop_small(sp.bmat(blocks))

    def inner_product(self, f, additional_order=2, representation="physical"):
        """Load vector ``(Lambda_i, f)`` of a physical (or logical) target."""
        target = self.pullback(f) if representation == "physical" else f
        (px, wx), (py, wy) = self._quadrature(additional_order)
        X, Y = np.meshgrid(px, py, indexing="ij")
        w = np.outer(wx, wy)
        W = self.metric_weight(X, Y)
        values = target(X, Y)
        result = np.empty(self.n)
        if self.k != 1:
            weighted = (w * W * np.broadcast_to(values, X.shape)).ravel(order="F")
            return self.logical.basis_at(px, py, 0).T @ weighted
        vx, vy = (np.broadcast_to(v, X.shape) for v in values)
        weighted = [w * (W[..., 0, 0] * vx + W[..., 0, 1] * vy), w * (W[..., 1, 0] * vx + W[..., 1, 1] * vy)]
        for c in range(2):
            result[self.logical.component_range(c)] = self.logical.basis_at(px, py, c).T @ weighted[c].ravel(order="F")
        return result

    def geometric_dofs(self, f, additional_order=2, representation="physical"):
        target = self.pullback(f) if representation == "physical" else f
        return self.logical.geometric_dofs(target, additional_order)

    def commuting_projection(self, f, additional_order=2, representation="physical"):
        return self.logical.solve_histopolation(self.geometric_dofs(f, additional_order, representation))

    def l2_projection(self, f, additional_quadrature_order=None, mass=None):
        additional = mass_additional_order(None, additional_quadrature_order)
        M = self.mass_matrix() if mass is None else mass
        return spla.spsolve(sp.csc_matrix(M), self.inner_product(f, additional))

    def l2_error(self, coefficients, f, additional_order=2, representation="physical"):
        """``|| u_h - f ||_{L2}`` by Gauss quadrature on the logical cells."""
        target = self.pullback(f) if representation == "physical" else f
        (px, wx), (py, wy) = self._quadrature(additional_order)
        X, Y = np.meshgrid(px, py, indexing="ij")
        w = np.outer(wx, wy)
        W = self.metric_weight(X, Y)
        values = self.logical.evaluate(coefficients, px, py)
        exact = target(X, Y)
        if self.k != 1:
            difference = values - exact
            return np.sqrt(max(np.sum(w * W * difference**2), 0.0))
        dx, dy = values[0] - exact[0], values[1] - exact[1]
        quadratic = W[..., 0, 0] * dx**2 + 2 * W[..., 0, 1] * dx * dy + W[..., 1, 1] * dy**2
        return np.sqrt(max(np.sum(w * quadratic), 0.0))

    def evaluate(self, coefficients, x, y):
        """Physical proxy of the element at the logical tensor grid ``x (x) y``."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return self.pushforward_values(self.logical.evaluate(coefficients, x, y), X, Y)

    def evaluate_points(self, coefficients, x, y):
        """Physical proxy of the element at the scattered logical points ``(x[k], y[k])``."""
        x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)))
        return self.pushforward_values(self.logical.evaluate_points(coefficients, x, y), x, y)

    def evaluate_physical(self, coefficients, X, Y):
        """Physical proxy of the element at physical points (requires ``mapping.inverse``)."""
        x, y = self.mapping.inverse(X, Y)
        return self.evaluate_points(coefficients, x, y)

    def polynomial_projection_columns(self, functions, method="commuting", additional_order=None):
        """Projections (one column each) of polynomials into a single-component (0- or 2-form) space.

        ``method="commuting"``: the GDOFs of every polynomial (pulled back with the factor 1 or ``det J``) followed by
        the histopolation solve, as in the paper; ``"l2"``: the full tensor-product L2 projection.
        """
        if not functions:
            return np.zeros((self.n, 0))
        if method == "l2":
            if additional_order not in (None, 0):
                raise ValueError("additional_quadrature_order must be zero for the L2 polynomial projection")
            loads = np.column_stack([self.inner_product(f, 0) for f in functions])
            lu = spla.splu(sp.csc_matrix(self.mass_matrix()))
            return lu.solve(loads)
        if method != "commuting":
            raise ValueError("polynomial_projection must be 'l2' or 'commuting'")
        additional = 2 if additional_order is None else additional_order
        (px, Rx), (py, Ry) = self.logical._component_rules(0, additional)
        Xl, Yl = np.meshgrid(px, py, indexing="ij")
        X, Y = self.mapping(Xl, Yl)
        factor = 1.0 if self.k == 0 else self.mapping.jacobian(Xl, Yl)
        dofs = np.column_stack(
            [(Rx @ (factor * np.broadcast_to(f(X, Y), X.shape)) @ Ry.T).ravel(order="F") for f in functions]
        )
        return self.logical.solve_histopolation(dofs)
