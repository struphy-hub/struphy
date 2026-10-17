"""Solvers for ``d* d A = J`` based on the homotopy identities (port of ``src/solvers.jl`` of PolarFEEC.jl).

In the homotopy gauge the solution is ``A = h b``, where ``b = dA`` is the ``M``-orthogonal projection of
``M^{-1} h^T l`` onto ``range(d_k)``. For the 2D curl-curl problem (``k = 1``) this is a single polar 2-form mass solve
(the "free two-dimensional lunch"), ``A = h_2 (I - B) M_2^{-1} h_2^T l`` with the mean projection ``B`` for Dirichlet
boundary conditions. For ``k = 0`` (Poisson) the projection is a sparse saddle-point solve.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from struphy.polar.homotopy.linalg import SparsePlusLowRank, as_dense


def factor_basis(P, side):
    """Orthonormal basis of the span of the nontrivial ``left`` or ``right`` factor columns of ``P = L R^T``."""
    n = P.shape[0] if side == "left" else P.shape[1]
    if not isinstance(P, SparsePlusLowRank) or P.rank == 0:
        return np.zeros((n, 0))
    L, R = P.left, P.right
    active = [j for j in range(L.shape[1]) if np.linalg.norm(L[:, j]) > 0 and np.linalg.norm(R[:, j]) > 0]
    if not active:
        return np.zeros((n, 0))
    factors = (L if side == "left" else R)[:, active]
    Q, Rq, _ = sla.qr(factors, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(Rq))
    rank = int(np.sum(diagonal > 1e-12 * diagonal.max()))
    return Q[:, :rank]


def source_load(complex_, k, source, additional_order=None, representation="physical"):
    """Full tensor-product load vector ``(Lambda_i, J)`` of a callable, a full or a polar coefficient vector."""
    space = complex_.spaces[k]
    if callable(source):
        return space.inner_product(source, 0 if additional_order is None else additional_order, representation)
    source = np.asarray(source, dtype=float)
    if source.size == space.n:
        return complex_.full_mass_matrix(k) @ source
    if source.size == complex_.dimension(k):
        return complex_.full_mass_matrix(k) @ (complex_.extraction(k) @ source)
    raise ValueError("the source has neither full nor polar coefficient length")


class StiffnessSolver:
    """Solver of ``(d_k A, d_k Phi) + shift (A, Phi) = (J, Phi)`` on the polar space of degree ``k`` (2D).

    ``gauge="homotopy"`` (requires ``shift = 0``) returns the solution ``A = h b`` in the homotopy gauge
    (``h A = 0``); ``gauge="none"`` solves the (shifted) stiffness system directly.
    """

    def __init__(self, complex_, k, gauge="homotopy", shift=0.0):
        top = complex_.top_degree
        if not 0 <= k < top:
            raise ValueError(f"k must lie between 0 and {top - 1}")
        if gauge not in ("homotopy", "none"):
            raise ValueError("gauge must be 'homotopy' or 'none'")
        if gauge == "homotopy" and shift != 0:
            raise ValueError("the homotopy method solves the unshifted problem; use gauge='none' for shift > 0")
        if gauge == "none" and shift == 0:
            raise ValueError("the direct method needs an invertible operator: use shift > 0 or gauge='homotopy'")
        self.complex, self.k, self.gauge, self.shift = complex_, k, gauge, shift
        if gauge == "none":
            D = complex_.derivative(k)
            A = D.T @ complex_.mass_matrix(k + 1) @ D + shift * complex_.mass_matrix(k)
            self._lu = spla.splu(sp.csc_matrix((A + A.T) / 2))
        else:
            self._factorize_range_projection()

    # ------------------------------------------------------------------ range projection
    def _factorize_range_projection(self):
        """Sparse LU of the saddle system ``[M d^T 0 V ...; d 0 d^T 0 V ...; ...]`` over degrees ``k+1..top``."""
        c, k = self.complex, self.k
        degrees = list(range(k + 1, c.top_degree + 1))
        sizes = [c.dimension(j) for j in degrees]
        constraints = [factor_basis(c.cohomology_projection(j), "right") for j in degrees]
        ranks = [V.shape[1] for V in constraints]
        nblocks = len(degrees) + len(degrees)
        blocks = [[None] * nblocks for _ in range(nblocks)]
        all_sizes = sizes + ranks
        blocks[0][0] = c.mass_matrix(k + 1)
        for index, j in enumerate(degrees):
            if j < c.top_degree:
                D = c.derivative(j)
                blocks[index + 1][index] = D
                blocks[index][index + 1] = D.T
            if ranks[index] > 0:
                constraint = len(degrees) + index
                blocks[index][constraint] = sp.csr_matrix(constraints[index])
                blocks[constraint][index] = sp.csr_matrix(constraints[index].T)
        for i, size in enumerate(all_sizes):
            if blocks[i][i] is None:
                blocks[i][i] = sp.csr_matrix((size, size))
        system = sp.bmat(blocks, format="csc")
        self._range_lu = spla.splu(system)
        self._range_size = system.shape[0]

    def range_projection(self, load):
        """``M``-orthogonal projection of ``M^{-1} load`` onto ``range(d_k)`` (polar coefficients of degree k+1)."""
        rhs = np.zeros(self._range_size)
        rhs[: load.size] = load
        return self._range_lu.solve(rhs)[: load.size]

    # ------------------------------------------------------------------ solves
    def solve(self, load):
        """Polar coefficients of the solution for a polar load vector ``E^T (Lambda, J)``."""
        load = np.asarray(load, dtype=float)
        if self.gauge == "none":
            return self._lu.solve(load)
        h = self.complex.homotopy(self.k + 1)
        b = self.range_projection(h.T @ load)
        A = h @ b
        defect = np.max(np.abs(self.complex.derivative(self.k) @ A - b)) if b.size else 0.0
        if defect > 1e-8 * max(1.0, np.max(np.abs(b))):
            raise ArithmeticError("the polar spaces do not satisfy the homotopy identity required by the solver")
        return A

    def compatibility_residual(self, load):
        """``max |(D_{k-1}^T l, V^T l)|``: the source must be orthogonal to ``ker d_k``."""
        c, k = self.complex, self.k
        parts = []
        if k >= 1:
            parts.append(c.derivative(k - 1).T @ load)
        parts.append(factor_basis(c.cohomology_projection(k), "left").T @ load)
        parts = np.concatenate(parts)
        return float(np.max(np.abs(parts))) if parts.size else 0.0

    def __call__(self, source, additional_order=None, compatibility_tolerance=None, representation="physical"):
        """Full tensor-product coefficients of the solution for a callable or coefficient source."""
        c, k = self.complex, self.k
        load = c.extraction(k).T @ source_load(c, k, source, additional_order, representation)
        if self.shift == 0:
            residual = self.compatibility_residual(load)
            tolerance = (
                1e-10 * max(np.max(np.abs(load)), 1.0) if compatibility_tolerance is None else (compatibility_tolerance)
            )
            if residual > tolerance:
                raise ValueError(
                    f"the source is not L2-orthogonal to the kernel of the derivative (residual {residual:.3e}, "
                    f"tolerance {tolerance:.3e})"
                )
        return c.extraction(k) @ self.solve(load)


def stiffness_solver(complex_, k, gauge="homotopy", shift=0.0):
    """Cached :class:`StiffnessSolver` of ``complex_``."""
    key = ("stiffness_solver", k, gauge, float(shift))
    if key not in complex_._cache:
        complex_._cache[key] = StiffnessSolver(complex_, k, gauge, shift)
    return complex_._cache[key]


def range_projection(complex_, k, load):
    """The projection ``b = dA`` of the homotopy method for polar coefficients ``load`` of degree ``k + 1``."""
    return stiffness_solver(complex_, k).range_projection(np.asarray(load, dtype=float))


def curl_curl_free_lunch(complex_, load):
    """2D curl-curl in the homotopy gauge with one 2-form mass solve: ``A = h_2 (I - B) M_2^{-1} h_2^T l``.

    Identical to ``stiffness_solver(complex_, 1).solve(load)``; written out to show the algorithm of the paper.
    """
    h = complex_.homotopy(2)
    x = complex_.mass_solver(2).solve(h.T @ load)
    if complex_.boundary == "dirichlet":
        x = x - as_dense(complex_.cohomology_projection(2)) @ x
    return h @ x
