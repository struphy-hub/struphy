"""Fast serial quadrature kernels for the H1 (0-form) spline space.

Used by the Poisson–Boltzmann Newton solve of ``IonOpticsElectrostatic``, whose Hessian
``S + M[w]`` needs a new weighted mass matrix ``M[w]`` at every iteration. Struphy's generic
``WeightedMassOperator.assemble`` costs about 0.6 s on a 128 x 48 x 1 mesh; the sum-factorized
element-wise assembly here takes a few tens of ms. It relies on the quadrature tables of
``derham.spline_attributes["0"]`` (points, weights, knot spans, basis values) and needs
non-periodic directions in a serial run.

**Invariant directions.** A direction with one element and degree one (2 coefficients, both
basis functions summing to one) carries no structure for a field that does not depend on it,
e.g. the invariant ``z`` of a slit or the wedge angle of an axisymmetric problem. With
``reduce_invariant=True`` such a direction is collapsed: coefficients are shared by both
z-dofs (``c_full = P c_red``), the reduced matrices are ``P^T A P``, and reduced gradients are
``P^T g``. Gradient, Hessian and energy of the restricted problem are exactly the restrictions
of the full ones (chain rule), so nothing but the number of unknowns changes.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


class H1QuadratureAssembler:
    """Quadrature transfers and weighted mass matrices of the serial H1 space.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence (serial, non-periodic).

    space : StencilVectorSpace
        Coefficient space of H1, defining the flat ordering (``Vector.toarray()``).

    reduce_invariant : bool
        Collapse directions with one element and degree one (see the module docstring).
    """

    def __init__(self, derham, space, reduce_invariant=True):
        if any(space.periods):
            raise NotImplementedError("Periodic directions are not supported.")
        attributes = derham.spline_attributes["0"]
        self.npts_full = tuple(int(n) for n in space.npts)
        self._weights, self._values, self._first, self._nloc = [], [], [], []
        self.reduced = []
        self.npts = []
        for d in range(3):
            wts = np.asarray(attributes.quad_grid_wts[0][d])  # (n_el, nq)
            bases = np.asarray(attributes.quad_grid_bases[0][d])[:, :, 0, :]  # (n_el, p + 1, nq)
            spans = np.asarray(attributes.quad_grid_spans[0][d])
            n_el, n_loc = bases.shape[0], bases.shape[1]
            degree = n_loc - 1
            first = spans - degree
            reduce_d = bool(reduce_invariant and n_el == 1 and degree == 1 and self.npts_full[d] == 2)
            if reduce_d:
                bases, first, n_loc = np.ones((1, 1, bases.shape[2])), np.zeros(1, dtype=int), 1
            self.reduced.append(reduce_d)
            self._weights.append(wts)
            self._values.append(bases)
            self._first.append(first.astype(int))
            self._nloc.append(n_loc)
            self.npts.append(1 if reduce_d else self.npts_full[d])
        self.npts = tuple(self.npts)
        self.size = int(np.prod(self.npts))
        self.n_elements = tuple(w.shape[0] for w in self._weights)
        self.n_quad = tuple(w.shape[1] for w in self._weights)
        self.shape_q = tuple(n * q for n, q in zip(self.n_elements, self.n_quad))

        idx = [self._first[d][:, None] + np.arange(self._nloc[d])[None, :] for d in range(3)]
        self._idx = idx
        self._pair = [v[:, :, None, :] * v[:, None, :, :] for v in self._values]
        # global flat index of every local coefficient: layout (i, a, j, c, k, e)
        self._lin = np.ravel_multi_index(
            (
                idx[0][:, :, None, None, None, None],
                idx[1][None, None, :, :, None, None],
                idx[2][None, None, None, None, :, :],
            ),
            self.npts,
        ).reshape(
            self.n_elements[0], self._nloc[0], self.n_elements[1], self._nloc[1], self.n_elements[2], self._nloc[2]
        )
        self._pattern = None
        self._wq = (
            self._weights[0][:, :, None, None, None, None]
            * self._weights[1][None, None, :, :, None, None]
            * self._weights[2][None, None, None, None, :, :]
        )

        # projection P: reduced -> full coefficients, and its transposes
        if any(self.reduced):
            factors = [
                sps.csr_matrix(np.ones((self.npts_full[d], 1)))
                if self.reduced[d]
                else sps.identity(self.npts_full[d], format="csr")
                for d in range(3)
            ]
            self.projection = sps.kron(sps.kron(factors[0], factors[1]), factors[2], format="csr")
        else:
            self.projection = sps.identity(self.size, format="csr")

    # --- coefficient <-> quadrature values -------------------------------------------------

    def values(self, coefficients):
        """Spline values at all quadrature points, shape ``shape_q``, from (reduced) coefficients."""
        cl = np.asarray(coefficients)[self._lin]  # (i, a, j, c, k, e)
        t1 = np.einsum("iaq,iajcke->iqjcke", self._values[0], cl)
        t2 = np.einsum("jcr,iqjcke->iqjrke", self._values[1], t1)
        v = np.einsum("kes,iqjrke->iqjrks", self._values[2], t2)
        return v.reshape(self.shape_q)

    def project(self, f):
        """Load vector ``g_i = sum_q w_q f_q Lambda_i(x_q)`` for quadrature-point values ``f`` (adjoint of :meth:`values`)."""
        n, q = self.n_elements, self.n_quad
        F = f.reshape(n[0], q[0], n[1], q[1], n[2], q[2]) * self._wq
        t1 = np.einsum("iaq,iqjrks->iajrks", self._values[0], F)
        t2 = np.einsum("jcr,iajrks->iajcks", self._values[1], t1)
        g_local = np.einsum("kes,iajcks->iajcke", self._values[2], t2)
        return np.bincount(self._lin.ravel(), weights=g_local.ravel(), minlength=self.size)

    def integrate(self, f):
        """``sum_q w_q f_q``: the integral of a function given at the quadrature points."""
        n, q = self.n_elements, self.n_quad
        return float(np.sum(f.reshape(n[0], q[0], n[1], q[1], n[2], q[2]) * self._wq))

    # --- weighted mass matrix ----------------------------------------------------------------

    def _build_pattern(self):
        n, loc = self.n_elements, self._nloc
        rows = np.ravel_multi_index(
            (
                self._idx[0][:, :, None, None, None, None, None, None, None],
                self._idx[1][None, None, None, :, :, None, None, None, None],
                self._idx[2][None, None, None, None, None, None, :, :, None],
            ),
            self.npts,
        )
        cols = np.ravel_multi_index(
            (
                self._idx[0][:, None, :, None, None, None, None, None, None],
                self._idx[1][None, None, None, :, None, :, None, None, None],
                self._idx[2][None, None, None, None, None, None, :, None, :],
            ),
            self.npts,
        )
        shape = (n[0], loc[0], loc[0], n[1], loc[1], loc[1], n[2], loc[2], loc[2])
        keys = (np.broadcast_to(rows, shape).astype(np.int64) * self.size + np.broadcast_to(cols, shape)).ravel()
        unique, inverse = np.unique(keys, return_inverse=True)
        indptr = np.zeros(self.size + 1, dtype=np.int64)
        np.cumsum(np.bincount(unique // self.size, minlength=self.size), out=indptr[1:])
        self._pattern = (inverse, (unique % self.size).astype(np.int32), indptr.astype(np.int32), unique.size)

    def matrix(self, w):
        """CSR matrix ``M_ij = int w Lambda_i Lambda_j`` for weights ``w`` at the quadrature points (``shape_q``)."""
        if self._pattern is None:
            self._build_pattern()
        inverse, indices, indptr, nnz = self._pattern
        n, q = self.n_elements, self.n_quad
        W = w.reshape(n[0], q[0], n[1], q[1], n[2], q[2]) * self._wq
        t3 = np.einsum("iqjrks,kabs->iqjrkab", W, self._pair[2])
        t2 = np.einsum("iqjrkab,jcdr->iqjcdkab", t3, self._pair[1])
        m = np.einsum("iefq,iqjcdkab->iefjcdkab", self._pair[0], t2)
        data = np.bincount(inverse, weights=m.ravel(), minlength=nnz)
        return sps.csr_matrix((data, indices, indptr), shape=(self.size, self.size))

    # --- reduction helpers -------------------------------------------------------------------

    def reduce_matrix(self, matrix):
        """``P^T A P`` of a full-space sparse matrix."""
        return (self.projection.T @ matrix @ self.projection).tocsr()

    def reduce_load(self, vector):
        """``P^T g``: sum of a full-space load vector over the collapsed dofs."""
        return self.projection.T @ vector

    def reduce_coefficients(self, coefficients):
        """Average of full-space coefficients over the collapsed dofs (exact if they are equal)."""
        multiplicity = np.asarray(self.projection.sum(axis=0)).ravel()
        return (self.projection.T @ coefficients) / multiplicity

    def expand(self, reduced):
        """``P c``: full-space coefficients from reduced ones."""
        return self.projection @ reduced


class BandedSPDSolver:
    """Cholesky factorization of a symmetric positive definite sparse matrix by banded storage.

    Falls back to SuperLU if the band is wider than a quarter of the matrix.

    The bandwidth of a tensor-product FEEC matrix is ``p * n_fast``, with ``n_fast`` the size of the
    faster-varying directions; LAPACK's banded Cholesky exploits both symmetry and band structure and
    is several times faster than a general sparse LU here.
    """

    def __init__(self, matrix):
        matrix = sps.csr_matrix(matrix)
        coo = sps.tril(matrix).tocoo()
        self._bandwidth = int((coo.row - coo.col).max())
        self._n = matrix.shape[0]
        self._lu = None
        if self._bandwidth > self._n // 4:
            self._lu = spsl.splu(matrix.tocsc())
            return
        ab = np.zeros((self._bandwidth + 1, self._n))
        ab[coo.row - coo.col, coo.col] = coo.data
        self._factor = sla.cholesky_banded(ab, lower=True, check_finite=False)

    def solve(self, b):
        if self._lu is not None:
            return self._lu.solve(b)
        return sla.cho_solve_banded((self._factor, True), b, check_finite=False)
