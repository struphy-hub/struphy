"""Small linear-algebra helpers: matrices kept as a sparse part plus a factored low-rank part."""

import numpy as np
import scipy.sparse as sp


class SparsePlusLowRank:
    """The matrix ``S + U V^T`` with sparse ``S`` and dense ``U`` (n x r), ``V`` (m x r)."""

    __array_priority__ = 100

    def __init__(self, sparse, left=None, right=None):
        self.sparse = sp.csr_matrix(sparse)
        n, m = self.sparse.shape
        self.left = np.zeros((n, 0)) if left is None else np.asarray(left, dtype=float).reshape(n, -1)
        self.right = np.zeros((m, 0)) if right is None else np.asarray(right, dtype=float).reshape(m, -1)

    @classmethod
    def rank_one(cls, left, right):
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        return cls(sp.csr_matrix((left.size, right.size)), left[:, None], right[:, None])

    @classmethod
    def zero(cls, n, m=None):
        return cls(sp.csr_matrix((n, n if m is None else m)))

    @property
    def shape(self):
        return self.sparse.shape

    @property
    def rank(self):
        return self.left.shape[1]

    @property
    def T(self):
        return SparsePlusLowRank(self.sparse.T, self.right, self.left)

    def toarray(self):
        return self.sparse.toarray() + self.left @ self.right.T

    def is_zero(self):
        return self.sparse.nnz == 0 and (self.rank == 0 or not np.any(self.left) or not np.any(self.right))

    def __matmul__(self, other):
        if isinstance(other, SparsePlusLowRank):
            # (S1 + U1 V1^T)(S2 + U2 V2^T) = S1 S2 + [S1 U2 + U1 (V1^T U2)] V2^T + U1 (S2^T V1)^T
            sparse = self.sparse @ other.sparse
            left = np.hstack([self.sparse @ other.left + self.left @ (self.right.T @ other.left), self.left])
            right = np.hstack([other.right, other.sparse.T @ self.right])
            return SparsePlusLowRank(sparse, left, right)
        if sp.issparse(other):
            return SparsePlusLowRank(self.sparse @ other, self.left, other.T @ self.right)
        other = np.asarray(other)
        return self.sparse @ other + self.left @ (self.right.T @ other)

    def __rmatmul__(self, other):
        if sp.issparse(other):
            return SparsePlusLowRank(other @ self.sparse, other @ self.left, self.right)
        other = np.asarray(other)
        return other @ self.sparse + (other @ self.left) @ self.right.T

    def __add__(self, other):
        if isinstance(other, SparsePlusLowRank):
            return SparsePlusLowRank(
                self.sparse + other.sparse, np.hstack([self.left, other.left]), np.hstack([self.right, other.right])
            )
        if sp.issparse(other):
            return SparsePlusLowRank(self.sparse + other, self.left, self.right)
        return self.toarray() + other

    __radd__ = __add__

    def __neg__(self):
        return SparsePlusLowRank(-self.sparse, -self.left, self.right)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, scalar):
        return SparsePlusLowRank(scalar * self.sparse, scalar * self.left, self.right)

    __rmul__ = __mul__


def identity_minus(P):
    """``I - P`` for a square :class:`SparsePlusLowRank` ``P``."""
    n = P.shape[0]
    return SparsePlusLowRank(sp.identity(n, format="csr") - P.sparse, -P.left, P.right)


def as_dense(A):
    if isinstance(A, SparsePlusLowRank) or sp.issparse(A):
        return A.toarray()
    return np.asarray(A)


def clean(A, tolerance=100 * np.finfo(float).eps):
    """Drop entries below ``tolerance`` of a sparse matrix (absolute, as in PolarFEEC.jl)."""
    A = sp.csr_matrix(A, copy=True)
    A.data[np.abs(A.data) < tolerance] = 0.0
    A.eliminate_zeros()
    return A


def max_abs(A):
    if sp.issparse(A):
        return float(np.max(np.abs(A.data))) if A.nnz else 0.0
    A = as_dense(A)
    return float(np.max(np.abs(A))) if A.size else 0.0


def kron_factored(A, B):
    """``kron(A, B)`` where ``A`` and/or ``B`` may be :class:`SparsePlusLowRank`."""
    if not isinstance(A, SparsePlusLowRank) and not isinstance(B, SparsePlusLowRank):
        return sp.kron(A, B, format="csr")
    A = A if isinstance(A, SparsePlusLowRank) else SparsePlusLowRank(A)
    B = B if isinstance(B, SparsePlusLowRank) else SparsePlusLowRank(B)
    # (SA + UA VA^T) (x) (SB + UB VB^T): the mixed terms kron(UA VA^T, SB) are not low rank and are densified
    sparse = sp.kron(A.sparse, B.sparse, format="csr")
    dense = np.zeros(sparse.shape)
    any_dense = False
    if A.rank and B.sparse.nnz:
        dense += np.kron(A.left @ A.right.T, B.sparse.toarray())
        any_dense = True
    if B.rank and A.sparse.nnz:
        dense += np.kron(A.sparse.toarray(), B.left @ B.right.T)
        any_dense = True
    left = np.zeros((sparse.shape[0], A.rank * B.rank))
    right = np.zeros((sparse.shape[1], A.rank * B.rank))
    column = 0
    for i in range(A.rank):
        for j in range(B.rank):
            left[:, column] = np.kron(A.left[:, i], B.left[:, j])
            right[:, column] = np.kron(A.right[:, i], B.right[:, j])
            column += 1
    if any_dense:
        sparse = sp.csr_matrix(sparse.toarray() + dense)
    return SparsePlusLowRank(sparse, left, right)


def matmul(A, B):
    """``A @ B`` for any combination of sparse, dense and :class:`SparsePlusLowRank` operands."""
    if isinstance(A, SparsePlusLowRank):
        return A @ B
    if isinstance(B, SparsePlusLowRank):
        if sp.issparse(A):
            return SparsePlusLowRank(A @ B.sparse, A @ B.left, B.right)
        return np.asarray(A) @ B.sparse + (np.asarray(A) @ B.left) @ B.right.T
    return A @ B
