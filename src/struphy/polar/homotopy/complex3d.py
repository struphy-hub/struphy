"""The 3D polar complex as the total complex of the 2D polar complex and the 1D periodic (toroidal) complex.

Port of ``src/complexes.jl`` and ``src/polar3d.jl`` of PolarFEEC.jl at the coefficient level::

    W^m = sum_{p + q = m} A^p (x) B^q,
    d(V (x) U) = d_A V (x) U + (-1)^p V (x) d_B U,
    h(V (x) U) = h_A V (x) U + (-1)^p P_A V (x) h_B U   (ordered homotopy),
    P = P_A (x) P_B.

Within degree ``m`` the blocks are ordered by decreasing ``p``; each block uses the layout ``kron(B, A)`` (coefficients
of ``A`` vary fastest). Operators are returned as dense or sparse matrices (intended for verification and small
problems; the distributed 3D operators live in :class:`~struphy.feec.psydac_derham.Derham`).
"""

import numpy as np
import scipy.sparse as sp

from struphy.polar.homotopy.linalg import SparsePlusLowRank, as_dense
from struphy.polar.homotopy.spaces1d import PeriodicBSplineSpace, TrigonometricSpace, periodic_primitive_matrix


def _sparse(A):
    """Sparse matrix of a sparse, dense or factored matrix (low-rank parts are multiplied out)."""
    if isinstance(A, SparsePlusLowRank):
        if A.rank == 0:
            return A.sparse
        return sp.csr_matrix(A.sparse + sp.csr_matrix(A.left @ A.right.T))
    return sp.csr_matrix(A)


class PeriodicComplex:
    """The 1D periodic complex ``space -> scaled(space)`` with the homotopy normalized at ``base_point``.

    ``h A = int_{base}^{x} (A - int A)``, ``P0 u = u(base)`` and ``P1 A = int A`` (as the paper's 3D construction).
    """

    top_degree = 1

    def __init__(self, space, base_point=None):
        if not isinstance(space, (PeriodicBSplineSpace, TrigonometricSpace)):
            raise TypeError("the periodic complex needs a periodic B-spline or trigonometric space")
        self.space = space
        self.base_point = float(space.greville_points()[0] if base_point is None else base_point)

    def dimension(self, k):
        return self.space.n

    def derivative(self, k):
        assert k == 0
        return self.space.derivative_matrix()

    def extraction(self, k):
        return sp.identity(self.space.n, format="csr")

    duals = extraction

    def cohomology_projection(self, k):
        if k == 0:
            evaluation = self.space.basis_matrix(np.array([self.base_point])).toarray().ravel()
            return SparsePlusLowRank.rank_one(self.space.constant_coefficients(), evaluation)
        return SparsePlusLowRank.rank_one(
            self.space.constant_one_form_coefficients(), self.space.scaled.coefficient_integrals
        )

    def homotopy(self, k):
        assert k == 1
        primitive = periodic_primitive_matrix(self.space, normalization=self.base_point)
        mean = self.cohomology_projection(1)
        return SparsePlusLowRank(primitive) @ SparsePlusLowRank(sp.identity(self.space.n, format="csr") - _sparse(mean))


class TensorProductComplex:
    """The total complex ``W^m = sum_{p+q=m} A^p (x) B^q`` of two complexes with contracting homotopies."""

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.top_degree = first.top_degree + second.top_degree

    def bidegree_blocks(self, m):
        top_a, top_b = self.first.top_degree, self.second.top_degree
        return [(p, m - p) for p in range(min(m, top_a), max(0, m - top_b) - 1, -1)]

    def _block_sizes(self, m):
        return [self.first.dimension(p) * self.second.dimension(q) for p, q in self.bidegree_blocks(m)]

    def dimension(self, m):
        return sum(self._block_sizes(m))

    def derivative(self, m):
        source, target = self.bidegree_blocks(m), self.bidegree_blocks(m + 1)
        blocks = [[None] * len(source) for _ in target]
        for s, (p, q) in enumerate(source):
            if p < self.first.top_degree:
                t = target.index((p + 1, q))
                blocks[t][s] = sp.kron(sp.identity(self.second.dimension(q)), _sparse(self.first.derivative(p)))
            if q < self.second.top_degree:
                t = target.index((p, q + 1))
                block = (-1) ** p * sp.kron(_sparse(self.second.derivative(q)), sp.identity(self.first.dimension(p)))
                blocks[t][s] = block if blocks[t][s] is None else blocks[t][s] + block
        return _bmat(blocks, self._block_sizes(m + 1), self._block_sizes(m))

    def homotopy(self, m):
        source, target = self.bidegree_blocks(m), self.bidegree_blocks(m - 1)
        blocks = [[None] * len(source) for _ in target]
        for s, (p, q) in enumerate(source):
            if p >= 1:
                t = target.index((p - 1, q))
                blocks[t][s] = sp.kron(sp.identity(self.second.dimension(q)), _sparse(self.first.homotopy(p)))
            if q >= 1:
                projection = self.first.cohomology_projection(p)
                if isinstance(projection, SparsePlusLowRank) and projection.is_zero():
                    continue
                t = target.index((p, q - 1))
                block = (-1) ** p * sp.kron(_sparse(self.second.homotopy(q)), _sparse(projection))
                blocks[t][s] = block if blocks[t][s] is None else blocks[t][s] + block
        return _bmat(blocks, self._block_sizes(m - 1), self._block_sizes(m))

    def cohomology_projection(self, m):
        blocks = [
            sp.kron(_sparse(self.second.cohomology_projection(q)), _sparse(self.first.cohomology_projection(p)))
            for p, q in self.bidegree_blocks(m)
        ]
        return sp.block_diag(blocks).tocsr()

    def extraction(self, m):
        blocks = [
            sp.kron(_sparse(self.second.extraction(q)), _sparse(self.first.extraction(p)))
            for p, q in self.bidegree_blocks(m)
        ]
        return sp.block_diag(blocks).tocsr()

    def duals(self, m):
        blocks = [
            sp.kron(_sparse(self.second.duals(q)), _sparse(self.first.duals(p))) for p, q in self.bidegree_blocks(m)
        ]
        return sp.block_diag(blocks).tocsr()

    def cohomology_dimensions(self):
        ranks = [np.linalg.matrix_rank(as_dense(self.derivative(k))) for k in range(self.top_degree)] + [0]
        return [self.dimension(k) - ranks[k] - (ranks[k - 1] if k > 0 else 0) for k in range(self.top_degree + 1)]


def _bmat(blocks, row_sizes, column_sizes):
    # scipy.sparse.bmat needs at least one block per block row/column to infer sizes: pad with zero blocks
    padded = [
        [
            blocks[i][j] if blocks[i][j] is not None else sp.csr_matrix((row_sizes[i], column_sizes[j]))
            for j in range(len(column_sizes))
        ]
        for i in range(len(row_sizes))
    ]
    return sp.bmat(padded, format="csr")


class PolarComplex3D(TensorProductComplex):
    """The 3D polar complex: the 2D polar complex of the cross-section (x) the periodic toroidal complex.

    Parameters
    ----------
    polar2d : PolarComplex2D
        Polar complex of the reference cross-section.
    toroidal : PeriodicBSplineSpace or TrigonometricSpace
        Periodic toroidal 0-form space.
    """

    def __init__(self, polar2d, toroidal):
        super().__init__(polar2d, PeriodicComplex(toroidal))
        self.polar2d, self.toroidal = polar2d, toroidal
