r"""
2D x 1D Kronecker structure of linear operators on polar spline spaces.

Polar extraction operators only act in the poloidal plane :math:`(\eta_1, \eta_2)` and are the identity
in :math:`\eta_3`. (Axisymmetric) weighted mass matrices thus have the structure

.. math::

    \mathbb A = \bigoplus_G \mathbb A^G_\textnormal{pol} \otimes \mathbb A^G_3 \,,

where the direct sum runs over groups :math:`G` of vector components (components which share the same
spline basis in :math:`\eta_3` and are coupled at the pole), :math:`\mathbb A^G_\textnormal{pol}` is a
sparse matrix acting on the polar unknowns of the poloidal plane (polar coefficients and tensor-product
rings :math:`i \geq` ``n_rings``) and :math:`\mathbb A^G_3` is a 1d (banded) matrix.

The inverse :math:`(\mathbb A_\textnormal{pol} \otimes \mathbb A_3)^{-1} = (\mathbb A_\textnormal{pol}^{-1} \otimes I)(I \otimes \mathbb A_3^{-1})`
is applied in two steps: a 1d solve along :math:`\eta_3` and a (batched) 2d solve in the poloidal plane.
For the latter, the following solvers are available (see :func:`make_poloidal_solver`):

* ``"lu"``: sparse LU factorization (exact),
* ``"ilu"``: incomplete LU factorization (approximate inverse),
* ``"schur"``: block elimination of the pole block (polar coefficients and first tensor-product ring),
  with a dense LU of the small Schur complement and an exact (Kronecker or sparse LU) or incomplete (ILU)
  solve of the outer block.

In parallel runs, the data is redistributed with ``MPI_Alltoallv``: along :math:`\eta_3` within the
sub-communicator of processes sharing the same poloidal sub-domain, and in the poloidal plane within the
sub-communicator of processes sharing the same :math:`\eta_3` sub-domain.
"""

import logging
from dataclasses import dataclass, field

import cunumpy as xp
import numpy as np
import scipy.sparse as spa
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import LinearOperator
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector
from maybempi import MPI
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.sparse.linalg import spilu, splu

from struphy.polar.basic import PolarDerhamSpace, PolarVector
from struphy.polar.linear_operators import DeviceSparseMatrix, device_sparse_module

logger = logging.getLogger("struphy")

POLOIDAL_SOLVERS = ("schur", "ilu", "lu")

# default parameters of the incomplete LU factorization
ILU_DROP_TOL = 1e-4
ILU_FILL_FACTOR = 10.0


# --------------------------------------------------------------------------------------
# Index helpers
# --------------------------------------------------------------------------------------
def component_groups(space: PolarDerhamSpace) -> tuple[tuple[int, ...], ...]:
    """Groups of vector components which share the spline basis in eta_3 and are coupled at the pole."""
    groups = {
        "H1": ((0,),),
        "L2": ((0,),),
        "Hcurl": ((0, 1), (2,)),
        "Hdiv": ((0, 1), (2,)),
        "H1vec": ((0, 1), (2,)),
    }
    return groups[space.space_id]


def tp_space(space: PolarDerhamSpace, c: int):
    """StencilVectorSpace of component c of the parent (tensor-product) space."""
    parent = space.parent_space
    return parent if space.n_comps == 1 else parent.spaces[c]


def tp_component(v, c: int) -> StencilVector:
    """Component c of a Stencil-/BlockVector."""
    return v if isinstance(v, StencilVector) else v[c]


def n_pol_unknowns(space: PolarDerhamSpace, c: int) -> int:
    """Number of polar unknowns of component c in the poloidal plane."""
    n1, n2 = tp_space(space, c).npts[:2]
    return space.n_polar[c] + (n1 - space.n_rings[c]) * n2


def has_essential_bc(space_id: str, basis: str, c: int, d: int) -> bool:
    """Whether boundary conditions in direction d apply to component c (same rule as the boundary operators)."""
    if space_id == "H1vec":
        return c == d
    return basis == "B"


def bc_masks(
    space: PolarDerhamSpace, comps: tuple[int, ...], bases: list, bc: tuple | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Masks of the unknowns that are free of essential boundary conditions.

    Parameters
    ----------
    space : PolarDerhamSpace
        The polar space.

    comps : tuple[int]
        Components of the group.

    bases : list
        ``bases[c][d]`` is the spline basis (``"B"`` or ``"M"``) of component c in direction d.

    bc : tuple | None
        ``bc[d] = (left, right)`` flags of essential boundary conditions (None: no boundary conditions).

    Returns
    -------
    free_pol : np.ndarray[bool]
        Mask of the poloidal unknowns of the group (polar coefficients are always free).

    free_3 : np.ndarray[bool]
        Mask of the unknowns in eta_3.
    """
    free_pol = []
    free_3 = None
    for c in comps:
        n1, n2, n3 = tp_space(space, c).npts
        nr = space.n_rings[c]
        mask = np.ones(space.n_polar[c] + (n1 - nr) * n2, dtype=bool)
        mask3 = np.ones(n3, dtype=bool)
        if bc is not None:
            # the left boundary in eta_1 is the pole
            if bc[0][1] and has_essential_bc(space.space_id, bases[c][0], c, 0):
                mask[-n2:] = False
            assert not any(bc[1]), "Polar splines are periodic in eta_2."
            if has_essential_bc(space.space_id, bases[c][2], c, 2):
                if bc[2][0]:
                    mask3[0] = False
                if bc[2][1]:
                    mask3[-1] = False
        free_pol.append(mask)
        if free_3 is None:
            free_3 = mask3
        else:
            assert np.all(free_3 == mask3), "Components of a group must have the same boundary conditions in eta_3."
    return np.concatenate(free_pol), free_3


def apply_mask(A: spa.spmatrix | np.ndarray, free: np.ndarray, identity: bool):
    """D A D (+ I - D if identity), with D the diagonal matrix of the mask ``free``."""
    if np.all(free):
        return A
    D = spa.diags(free.astype(float))
    out = D @ spa.csr_matrix(A) @ D
    if identity:
        out = out + spa.diags((~free).astype(float))
    return out.tocsr() if spa.issparse(A) else out.toarray()


# --------------------------------------------------------------------------------------
# Poloidal part of polar extraction operators
# --------------------------------------------------------------------------------------
def _pad_cols(B, n_rows: int, n_cols: int) -> spa.csr_matrix:
    """Sparse matrix B (or zeros if B is None) padded with zero columns to n_cols columns."""
    if B is None:
        return spa.csr_matrix((n_rows, n_cols))
    B = spa.csr_matrix(B)
    assert B.shape[0] == n_rows
    if B.shape[1] < n_cols:
        B = spa.hstack([B, spa.csr_matrix((n_rows, n_cols - B.shape[1]))])
    return B.tocsr()


def poloidal_extraction_matrix(
    space: PolarDerhamSpace, comps: tuple[int, ...], blocks_ten_to_pol: list, blocks_ten_to_ten: list | None = None
) -> spa.csr_matrix:
    r"""
    Poloidal part of a :class:`~struphy.polar.linear_operators.PolarExtractionOperator`, restricted to a group of components.

    Maps the tensor-product coefficients of the poloidal plane (component by component, C-ordering of
    :math:`(i, j)`) to the polar unknowns (component by component: polar coefficients, then the
    tensor-product rings :math:`i \geq` ``n_rings`` in C-ordering).
    """
    rows = []
    for m in comps:
        n1m, n2m = tp_space(space, m).npts[:2]
        nrm = space.n_rings[m]
        row_pol = []
        row_tp = []
        for n in comps:
            n1n, n2n = tp_space(space, n).npts[:2]
            ncols = n1n * n2n

            # polar coefficients <- inner rings
            B = blocks_ten_to_pol[m][n] if blocks_ten_to_pol is not None else None
            row_pol.append(_pad_cols(B, space.n_polar[m], ncols))

            # tensor-product rings i >= n_rings (identity, possibly with modified first ring)
            ident = spa.eye((n1m - nrm) * n2m, ncols, k=nrm * n2m, format="csr") if m == n else None
            if blocks_ten_to_ten is None:
                row_tp.append(ident if ident is not None else spa.csr_matrix(((n1m - nrm) * n2m, ncols)))
            else:
                first = _pad_cols(blocks_ten_to_ten[m][n], n2m, ncols)
                if ident is not None:
                    rest = ident[n2m:]
                else:
                    rest = spa.csr_matrix(((n1m - nrm - 1) * n2m, ncols))
                row_tp.append(spa.vstack([first, rest]))
        rows.append(row_pol)
        rows.append(row_tp)
    return spa.bmat(rows, format="csr")


def poloidal_average_basis(space: PolarDerhamSpace, E_op) -> spa.csr_matrix:
    r"""
    Basis of the polar 0-forms of the poloidal plane that are constant in :math:`\eta_2` (functions of :math:`\eta_1` only),
    as columns in the polar unknowns (polar coefficients, then the tensor-product rings :math:`i \geq` ``n_rings``):
    the polar coefficients of the function that is 1 on the inner rings, and the constant in :math:`\eta_2` on each
    outer ring.

    Parameters
    ----------
    space : PolarDerhamSpace
        The polar 0-form space (``"H1"``).

    E_op : PolarExtractionOperator
        The basis extraction operator of the space.
    """
    assert space.space_id == "H1", "Averages in eta_2 are only implemented for 0-forms."
    n1, n2 = tp_space(space, 0).npts[:2]
    npol, nr = space.n_polar[0], space.n_rings[0]

    # polar coefficients of the function that is 1 on the inner rings (partition of unity of the polar splines)
    E = poloidal_extraction_matrix(space, (0,), E_op.blocks_ten_to_pol, None)
    E_inner = E[:npol, : nr * n2].toarray().T
    ones = np.ones(nr * n2)
    p0 = np.linalg.lstsq(E_inner, ones, rcond=None)[0]
    assert np.allclose(E_inner @ p0, ones), "The polar splines must reproduce constants on the inner rings."

    cols = [np.concatenate((p0, np.zeros((n1 - nr) * n2)))[:, None]]
    B = spa.hstack([spa.csr_matrix(np.column_stack(cols))] + [_ring_constant(npol, n1 - nr, n2)])
    return B.tocsr()


def _ring_constant(npol: int, n_rings_outer: int, n2: int) -> spa.csr_matrix:
    """Columns that are 1 on one outer ring each (rows: polar unknowns)."""
    rows = npol + np.arange(n_rings_outer * n2)
    cols = np.repeat(np.arange(n_rings_outer), n2)
    return spa.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(npol + n_rings_outer * n2, n_rings_outer))


# --------------------------------------------------------------------------------------
# Poloidal solvers
# --------------------------------------------------------------------------------------
def _matrix_key(matrix):
    """Exact numerical contents, independent of sparse storage order and explicit zeros."""
    matrix = spa.csc_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return (
        matrix.shape,
        matrix.dtype.str,
        matrix.indptr.astype(np.int64, copy=False).tobytes(),
        matrix.indices.astype(np.int64, copy=False).tobytes(),
        matrix.data.tobytes(),
    )


def _cache_context():
    """Keep device copies separate when solvers are built on different backends/devices."""
    backend = xp.get_backend()
    if backend == "cupy":
        import cupy

        return backend, cupy.cuda.runtime.getDevice()
    return backend, None


def _factorized_1d(matrix, cache=None):
    key = ("1d", _cache_context(), _matrix_key(matrix))
    factor = None if cache is None else cache.get(key)
    if factor is None:
        factor = _FactorizedSolve(splu(spa.csc_matrix(matrix)), dense=True)
        if cache is not None:
            cache[key] = factor
    return factor


class _FactorizedSolve:
    """Host factorization with a cached device solve; dense inverses are only used for small 1d factors."""

    def __init__(self, factor, dense=False):
        self._factor = factor
        self._dense = dense
        self._device = None

    def solve(self, Y):
        if Y.shape[-1] == 0:
            return Y.copy()
        if not xp.is_gpu(Y):
            return self._factor.solve(np.asarray(Y, order="F"))
        if self._device is None:
            if self._dense:
                self._device = xp.asarray(self._factor.solve(np.eye(self._factor.shape[0])))
            else:
                # CuPy wraps the existing SciPy factors: factorization stays at
                # setup, while permutations and batched triangular solves run on device.
                self._device = device_sparse_module().linalg.SuperLU(self._factor)
        return self._device @ Y if self._dense else self._device.solve(Y)


class _SparseLU:
    """Exact (LU) or incomplete (ILU) factorization of a sparse matrix."""

    def __init__(self, A: spa.spmatrix, kind: str = "lu", drop_tol=ILU_DROP_TOL, fill_factor=ILU_FILL_FACTOR):
        assert kind in ("lu", "ilu")
        A = spa.csc_matrix(A)
        if kind == "lu":
            self._fac = splu(A)
        else:
            self._fac = spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        self._solve = _FactorizedSolve(self._fac)
        self.exact = kind == "lu"

    def solve(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape[-1] == 0:
            return Y.copy()
        return self._solve.solve(Y)


class _KroneckerBlockDiagonal:
    """Exact solver for a block diagonal matrix with Kronecker blocks A1 x A2 (C-ordering of the unknowns)."""

    def __init__(self, factors: list[tuple[np.ndarray, np.ndarray]], cache=None):
        self._sizes = []
        self._lus = []
        for A1, A2 in factors:
            self._sizes.append((A1.shape[0], A2.shape[0]))
            self._lus.append(
                (
                    _factorized_1d(A1, cache),
                    _factorized_1d(A2, cache),
                )
            )
        self.exact = True

    def solve(self, Y: np.ndarray) -> np.ndarray:
        arrays = xp if xp.is_gpu(Y) else np
        out = arrays.empty_like(Y)
        m = Y.shape[-1]
        off = 0
        for (n1, n2), (lu1, lu2) in zip(self._sizes, self._lus):
            X = Y[off : off + n1 * n2].reshape(n1, n2 * m)
            X = lu1.solve(X).reshape(n1, n2, m)
            X = arrays.ascontiguousarray(X.transpose(1, 0, 2)).reshape(n2, n1 * m)
            X = lu2.solve(X).reshape(n2, n1, m)
            out[off : off + n1 * n2] = X.transpose(1, 0, 2).reshape(n1 * n2, m)
            off += n1 * n2
        return out


class _SchurSolver:
    r"""
    Block elimination of the pole block :math:`p` (polar coefficients and first tensor-product ring) of a
    poloidal matrix :math:`A = \begin{pmatrix} A_{pp} & A_{po} \\ A_{op} & A_{oo} \end{pmatrix}`,
    with Schur complement :math:`S = A_{pp} - A_{po} A_{oo}^{-1} A_{op}` (small and dense, LU factorized).
    """

    def __init__(self, A: spa.spmatrix, pole: np.ndarray, outer_solver, batch: int = 64):
        A = spa.csr_matrix(A)
        n = A.shape[0]
        self._p = np.flatnonzero(pole)
        self._o = np.flatnonzero(~pole)
        App = A[self._p][:, self._p].toarray()
        self._Apo = A[self._p][:, self._o].tocsr()
        self._Aop = A[self._o][:, self._p].tocsc()
        self._outer = outer_solver
        self.exact = outer_solver.exact

        # Schur complement (columns of A_op in batches to limit memory)
        S = App
        for k0 in range(0, self._p.size, batch):
            k1 = min(k0 + batch, self._p.size)
            W = self._outer.solve(self._Aop[:, k0:k1].toarray())
            S[:, k0:k1] -= self._Apo @ W
        self._S = lu_factor(S)
        self._device = None
        self._n = n
        logger.debug(f"Schur solver: {n = }, pole block {self._p.size}, outer block {self._o.size}")

    def solve(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape[-1] == 0:
            return Y.copy()
        if xp.is_gpu(Y):
            if self._device is None:
                self._device = (
                    xp.asarray(self._p),
                    xp.asarray(self._o),
                    DeviceSparseMatrix(self._Apo),
                    DeviceSparseMatrix(self._Aop),
                    xp.asarray(lu_solve(self._S, np.eye(self._p.size))),
                )
            p, o, Apo, Aop, Sinv = self._device
            pole_solve = lambda rhs: Sinv @ rhs
            arrays = xp
        else:
            p, o, Apo, Aop = self._p, self._o, self._Apo, self._Aop
            pole_solve = lambda rhs: lu_solve(self._S, rhs)
            arrays = np
        Yp = Y[p]
        Yo = Y[o]
        t = self._outer.solve(Yo)
        Xp = pole_solve(Yp - Apo.dot(t))
        Xo = self._outer.solve(Yo - Aop.dot(Xp))
        X = arrays.empty_like(Y)
        X[p] = Xp
        X[o] = Xo
        return X


def make_poloidal_solver(
    A: spa.spmatrix,
    kind: str,
    *,
    pole: np.ndarray | None = None,
    outer_kronecker: list | None = None,
    schur_outer: str = "lu",
    drop_tol: float = ILU_DROP_TOL,
    fill_factor: float = ILU_FILL_FACTOR,
    _cache=None,
):
    """
    Solver for the poloidal matrix A.

    Parameters
    ----------
    A : scipy.sparse matrix
        The poloidal matrix.

    kind : str
        ``"lu"`` (exact), ``"ilu"`` (approximate inverse) or ``"schur"`` (block elimination of the pole block).

    pole : np.ndarray[bool]
        Mask of the pole block (needed for ``"schur"``).

    outer_kronecker : list | None
        For ``"schur"``: Kronecker factors ``[(A1, A2), ...]`` (one pair per component) of the outer block,
        which is then solved exactly with 1d solves. If None, the outer block is factorized as a sparse matrix.

    schur_outer : str
        Factorization of the outer block if it is not Kronecker: ``"lu"`` (exact) or ``"ilu"``.

    drop_tol, fill_factor : float
        Parameters of the incomplete LU factorization (``scipy.sparse.linalg.spilu``).
    """
    assert kind in POLOIDAL_SOLVERS, f"Unknown poloidal solver {kind!r}, must be one of {POLOIDAL_SOLVERS}."
    if kind == "schur":
        assert pole is not None
        settings = (
            np.asarray(pole, dtype=bool).tobytes(),
            None if outer_kronecker is None else tuple((_matrix_key(a), _matrix_key(b)) for a, b in outer_kronecker),
            schur_outer,
            float(drop_tol),
            float(fill_factor),
        )
    else:
        settings = (float(drop_tol), float(fill_factor)) if kind == "ilu" else ()
    key = ("poloidal", _cache_context(), kind, _matrix_key(A), settings)
    if _cache is not None:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    if kind in ("lu", "ilu"):
        solver = _SparseLU(A, kind, drop_tol=drop_tol, fill_factor=fill_factor)
    else:
        if outer_kronecker is not None:
            outer = _KroneckerBlockDiagonal(outer_kronecker, cache=_cache)
            if __debug__:
                # the outer block must match the Kronecker factors
                o = np.flatnonzero(~pole)
                Aoo = spa.csr_matrix(A)[o][:, o]
                K = spa.block_diag([spa.kron(spa.csr_matrix(A1), spa.csr_matrix(A2)) for A1, A2 in outer_kronecker])
                diff = abs(Aoo - K).max() if Aoo.nnz + K.nnz > 0 else 0.0
                assert diff <= 1e-12 * max(abs(Aoo).max(), 1.0), f"Outer block is not Kronecker ({diff = })."
        else:
            o = np.flatnonzero(~pole)
            outer = make_poloidal_solver(
                spa.csr_matrix(A)[o][:, o],
                schur_outer,
                drop_tol=drop_tol,
                fill_factor=fill_factor,
                _cache=_cache,
            )
        solver = _SchurSolver(A, pole, outer)
    if _cache is not None:
        _cache[key] = solver
    return solver


def pole_mask(space: PolarDerhamSpace, comps: tuple[int, ...]) -> np.ndarray:
    """Mask of the pole block (polar coefficients and first tensor-product ring) of the poloidal unknowns."""
    masks = []
    for c in comps:
        n2 = tp_space(space, c).npts[1]
        mask = np.zeros(n_pol_unknowns(space, c), dtype=bool)
        mask[: space.n_polar[c] + n2] = True
        masks.append(mask)
    return np.concatenate(masks)


# --------------------------------------------------------------------------------------
# MPI redistribution
# --------------------------------------------------------------------------------------
class _BoxTranspose:
    """
    Redistribution of data within a communicator: before, every process holds a box (along the
    "gathered" axes) for all ``m_local`` columns; after, every process holds the full extent of the
    gathered axes for a chunk of the columns. The columns are the same on all processes of the communicator.
    """

    def __init__(self, comm, starts: tuple, ends: tuple, global_shape: tuple, m_local: int):
        self._comm = comm if (comm is not None and comm.Get_size() > 1) else None
        self._local_shape = tuple(int(e - s + 1) for s, e in zip(starts, ends))
        self._global_shape = tuple(int(n) for n in global_shape)
        self._m_local = int(m_local)

        size = 1 if self._comm is None else self._comm.Get_size()
        rank = 0 if self._comm is None else self._comm.Get_rank()
        bounds = np.linspace(0, self._m_local, size + 1).astype(int)
        self._chunks = [(int(bounds[r]), int(bounds[r + 1])) for r in range(size)]
        self._my_chunk = self._chunks[rank]

        if self._comm is None:
            assert self._local_shape == self._global_shape
            return

        self._boxes = self._comm.allgather((tuple(int(s) for s in starts), tuple(int(e) for e in ends)))
        loc = int(np.prod(self._local_shape))
        mc = self._my_chunk[1] - self._my_chunk[0]
        self._send_counts = np.array([loc * (c1 - c0) for c0, c1 in self._chunks], dtype=int)
        self._recv_counts = np.array(
            [int(np.prod([e - s + 1 for s, e in zip(*box)])) * mc for box in self._boxes], dtype=int
        )
        self._send_displs = np.concatenate(([0], np.cumsum(self._send_counts)[:-1]))
        self._recv_displs = np.concatenate(([0], np.cumsum(self._recv_counts)[:-1]))

    @property
    def my_chunk(self) -> tuple[int, int]:
        """Range of the columns held by this process after the forward redistribution."""
        return self._my_chunk

    def forward(self, local: np.ndarray) -> np.ndarray:
        """(*local_shape, m_local) -> (*global_shape, m_chunk)."""
        c0, c1 = self._my_chunk
        if self._comm is None:
            return local[..., c0:c1]

        sendbuf = xp.concatenate([local[..., a:b].ravel() for a, b in self._chunks])
        recvbuf = xp.empty(int(self._recv_counts.sum()), dtype=float)
        with xp.mpi.mpi_buffer(sendbuf) as send, xp.mpi.mpi_buffer(recvbuf, send=False, recv=True) as recv:
            self._comm.Alltoallv(
                [send, (self._send_counts, self._send_displs), MPI.DOUBLE],
                [recv, (self._recv_counts, self._recv_displs), MPI.DOUBLE],
            )
        full = xp.empty((*self._global_shape, c1 - c0), dtype=float)
        for (s, e), d, n in zip(self._boxes, self._recv_displs, self._recv_counts):
            shape = tuple(ei - si + 1 for si, ei in zip(s, e))
            full[tuple(slice(si, ei + 1) for si, ei in zip(s, e))] = recvbuf[d : d + n].reshape(*shape, c1 - c0)
        return full

    def backward(self, full: np.ndarray) -> np.ndarray:
        """(*global_shape, m_chunk) -> (*local_shape, m_local)."""
        if self._comm is None:
            return full

        sendbuf = xp.concatenate(
            [full[tuple(slice(si, ei + 1) for si, ei in zip(s, e))].ravel() for s, e in self._boxes],
        )
        recvbuf = xp.empty(int(self._send_counts.sum()), dtype=float)
        with xp.mpi.mpi_buffer(sendbuf) as send, xp.mpi.mpi_buffer(recvbuf, send=False, recv=True) as recv:
            self._comm.Alltoallv(
                [send, (self._recv_counts, self._recv_displs), MPI.DOUBLE],
                [recv, (self._send_counts, self._send_displs), MPI.DOUBLE],
            )
        local = xp.empty((*self._local_shape, self._m_local), dtype=float)
        for (a, b), d, n in zip(self._chunks, self._send_displs, self._send_counts):
            local[..., a:b] = recvbuf[d : d + n].reshape(*self._local_shape, b - a)
        return local


def _interior(u: StencilVector) -> np.ndarray:
    """View of the data of a StencilVector without ghost regions."""
    sp = u.space
    idx = tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(sp.pads, sp.shifts, sp.starts, sp.ends))
    return u._data[idx]


# communicators of processes sharing the same eta_3 sub-domain, cached per Cartesian communicator
_PLANE_COMMS = {}


def _plane_comm(cart):
    """Sub-communicator of the processes with the same eta_3 sub-domain (None in serial runs)."""
    comm = cart.comm_cart
    if comm is None or comm.Get_size() == 1:
        return None
    key = id(comm)
    if key not in _PLANE_COMMS:
        _PLANE_COMMS[key] = (comm, comm.Sub([True, True, False]))
    return _PLANE_COMMS[key][1]


def _line_comm(cart):
    """Sub-communicator of the processes with the same poloidal sub-domain (None in serial runs)."""
    comm = cart.comm_cart
    if comm is None or comm.Get_size() == 1:
        return None
    return cart.subcomm[2]


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------
@dataclass
class Kron2Dx1DGroup:
    """Factors of one group of components: A = A_pol x A_3."""

    comps: tuple[int, ...]
    A_pol: spa.spmatrix
    A_3: np.ndarray
    # Kronecker factors (A1, A2) of the outer block per component (for the Schur solver), optional
    outer_kronecker: list | None = None
    pole: np.ndarray | None = field(default=None, repr=False)

    def transpose(self) -> "Kron2Dx1DGroup":
        outer = None if self.outer_kronecker is None else [(A1.T, A2.T) for A1, A2 in self.outer_kronecker]
        return Kron2Dx1DGroup(self.comps, self.A_pol.T.tocsr(), self.A_3.T.copy(), outer, self.pole)

    def masked(self, free_pol: np.ndarray, free_3: np.ndarray, identity: bool, space: PolarDerhamSpace):
        """Factors with essential boundary conditions (D A D, plus identity on the constrained unknowns if identity)."""
        if self.outer_kronecker is None or not identity:
            outer = None if self.outer_kronecker is None else list(self.outer_kronecker)
            A_pol = apply_mask(self.A_pol, free_pol, identity)
        else:
            # keep the Kronecker structure of the outer block: (D_1 A_1 D_1 + I - D_1) x A_2, i.e. the block
            # 1 x A_2 (instead of the identity) for the constrained unknowns (which are decoupled from the others)
            outer = []
            fill = []
            off = 0
            for c, (A1, A2) in zip(self.comps, self.outer_kronecker):
                n_c = n_pol_unknowns(space, c)
                n2 = tp_space(space, c).npts[1]
                n1o = A1.shape[0]
                # eta_1 mask of the outer rings (constant in eta_2)
                free_1 = free_pol[off : off + n_c][-n1o * n2 :].reshape(n1o, n2)[:, 0]
                outer.append((apply_mask(A1, free_1, True), A2))
                fill.append(spa.csr_matrix((n_c - n1o * n2, n_c - n1o * n2)))
                fill.append(spa.kron(spa.diags((~free_1).astype(float)), spa.csr_matrix(A2)))
                off += n_c
            A_pol = (apply_mask(self.A_pol, free_pol, False) + spa.block_diag(fill)).tocsr()
        return Kron2Dx1DGroup(
            self.comps,
            A_pol,
            apply_mask(self.A_3, free_3, identity),
            outer,
            self.pole,
        )


class _PolarKron2Dx1DBase(LinearOperator):
    """Common data layout and parallel application of 2D x 1D Kronecker operators on a PolarDerhamSpace."""

    def __init__(self, space: PolarDerhamSpace, groups: list["Kron2Dx1DGroup | KronSum2Dx1DGroup"]):
        assert isinstance(space, PolarDerhamSpace)
        self._space = space
        self._groups = groups
        assert sorted(c for g in groups for c in g.comps) == list(range(space.n_comps))

        # redistribution along eta_3 (line) and in the poloidal plane (plane), per component
        self._t3 = {}
        self._tpl = {}
        for g in groups:
            for c in g.comps:
                sp = tp_space(space, c)
                s, e = sp.starts, sp.ends
                l1, l2, l3 = (e[d] - s[d] + 1 for d in range(3))
                self._t3[c] = _BoxTranspose(_line_comm(sp.cart), (s[2],), (e[2],), (sp.npts[2],), l1 * l2)
                self._tpl[c] = _BoxTranspose(_plane_comm(sp.cart), s[:2], e[:2], sp.npts[:2], l3)

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return float

    @property
    def groups(self) -> list[Kron2Dx1DGroup]:
        return self._groups

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def _apply(
        self, v: PolarVector, out: PolarVector | None, ops_3: list, ops_pol: list, ops_3_post: list | None = None
    ) -> PolarVector:
        """Apply ops_3[g] (to (n3, m) arrays) along eta_3, then ops_pol[g] (to (N_pol, m) arrays, called with the
        global eta_3 indices of the m columns) in the poloidal plane, and finally ops_3_post[g] along eta_3 (if given)."""
        assert isinstance(v, PolarVector)
        assert v.space == self._space
        if out is None:
            out = PolarVector(self._space)
        else:
            assert isinstance(out, PolarVector)
            assert out.space == self._space

        V = self._space
        out_pol = [xp.zeros_like(p) for p in v.pol]

        for g, op_3, op_pol in zip(self._groups, ops_3, ops_pol):
            # 1. eta_3 direction (tensor-product part redistributed, polar coeffs replicated)
            tmp_tp = {}
            tmp_pol = {}
            for c in g.comps:
                tmp_tp[c] = self._apply_3(_interior(tp_component(v.tp, c)), c, op_3)
                tmp_pol[c] = op_3(v.pol[c].T).T if V.n_polar[c] > 0 else v.pol[c].copy()

            # 2. poloidal plane
            s3 = tp_space(V, g.comps[0]).starts[2]
            k0, k1 = self._tpl[g.comps[0]].my_chunk
            full = {}
            blocks = []
            for c in g.comps:
                full[c] = self._tpl[c].forward(tmp_tp[c])
                nr = V.n_rings[c]
                blocks.append(tmp_pol[c][:, s3 + k0 : s3 + k1])
                n1, n2 = tp_space(V, c).npts[:2]
                blocks.append(full[c][nr:].reshape((n1 - nr) * n2, k1 - k0))
            Y = xp.concatenate(blocks, axis=0)
            Z = op_pol(Y, np.arange(s3 + k0, s3 + k1)) if k1 > k0 else Y

            off = 0
            for c in g.comps:
                npol = V.n_polar[c]
                nr = V.n_rings[c]
                n1, n2 = tp_space(V, c).npts[:2]
                out_pol[c][:, s3 + k0 : s3 + k1] = Z[off : off + npol]
                off += npol
                F = full[c]
                F[:nr] = 0.0
                F[nr:] = Z[off : off + (n1 - nr) * n2].reshape(n1 - nr, n2, k1 - k0)
                off += (n1 - nr) * n2
                u = tp_component(out.tp, c)
                _interior(u)[...] = self._tpl[c].backward(F)
                u.ghost_regions_in_sync = False

        # polar coefficients are held by all processes (each column was computed on exactly one process)
        comm = V.comm
        for c in range(V.n_comps):
            if comm is not None and comm.Get_size() > 1 and out_pol[c].size > 0:
                with xp.mpi.mpi_buffer(out_pol[c], recv=True) as buf:
                    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
            out.pol[c][:] = out_pol[c]

        # 3. eta_3 direction again
        if ops_3_post is not None:
            for g, op_3 in zip(self._groups, ops_3_post):
                for c in g.comps:
                    L = _interior(tp_component(out.tp, c))
                    L[...] = self._apply_3(L, c, op_3)
                    if V.n_polar[c] > 0:
                        out.pol[c][:] = op_3(out.pol[c].T).T

        return out

    def _apply_3(self, L: np.ndarray, c: int, op_3) -> np.ndarray:
        """Apply op_3 along eta_3 to the local data L (shape (l1, l2, l3)) of component c."""
        l1, l2, l3 = L.shape
        T3 = self._t3[c]
        X = T3.forward(xp.ascontiguousarray(L.reshape(l1 * l2, l3).T))
        return xp.ascontiguousarray(T3.backward(op_3(X)).T).reshape(l1, l2, l3)


class PolarKron2Dx1DSolver(_PolarKron2Dx1DBase):
    r"""
    (Approximate) inverse :math:`\bigoplus_G (\mathbb A^G_\textnormal{pol})^{-1} \otimes (\mathbb A^G_3)^{-1}` on a PolarDerhamSpace.

    The 1d matrices are factorized exactly, the poloidal ones with :func:`make_poloidal_solver`.

    Parameters
    ----------
    space : PolarDerhamSpace
        Domain and codomain.

    groups : list[Kron2Dx1DGroup]
        The factors of each group of components.

    poloidal_solver : str
        ``"schur"``, ``"ilu"`` or ``"lu"``.

    **solver_kwargs
        Passed to :func:`make_poloidal_solver` (``schur_outer``, ``drop_tol``, ``fill_factor``).
    """

    def __init__(
        self, space: PolarDerhamSpace, groups: list[Kron2Dx1DGroup], poloidal_solver: str = "schur", **solver_kwargs
    ):
        super().__init__(space, groups)
        self._kind = poloidal_solver
        self._solver_kwargs = solver_kwargs
        self._lu_3 = [_factorized_1d(g.A_3, space._factorization_cache) for g in groups]
        self._pol_solvers = [
            make_poloidal_solver(
                g.A_pol,
                poloidal_solver,
                pole=g.pole if g.pole is not None else pole_mask(space, g.comps),
                outer_kronecker=g.outer_kronecker,
                _cache=space._factorization_cache,
                **solver_kwargs,
            )
            for g in groups
        ]

    @property
    def exact(self) -> bool:
        """Whether the poloidal solves are exact (else the solver is an approximate inverse, e.g. a preconditioner)."""
        return all(s.exact for s in self._pol_solvers)

    @property
    def poloidal_solver(self) -> str:
        return self._kind

    def dot(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        ops_3 = [lu.solve for lu in self._lu_3]
        ops_pol = [lambda Y, modes, s=s: s.solve(Y) for s in self._pol_solvers]
        return self._apply(v, out, ops_3, ops_pol)

    def solve(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        return self.dot(v, out=out)

    def transpose(self, conjugate=False) -> "PolarKron2Dx1DSolver":
        return PolarKron2Dx1DSolver(
            self._space, [g.transpose() for g in self._groups], self._kind, **self._solver_kwargs
        )


@dataclass
class KronSum2Dx1DGroup:
    r"""
    Factors of one group of components: :math:`\mathbb A = \sum_t \mathbb A^t_\textnormal{pol} \otimes \mathbb B^t_3`
    with :math:`\mathbb B^t_3` the 1d mass matrix ``M_3`` (term ``"M"``), the 1d stiffness matrix ``S_3`` (term ``"S"``)
    or another 1d matrix of ``B_3`` (e.g. the identity), plus low-rank terms
    :math:`c \, \mathbb S \mathbb B (\mathbb B^\top \mathbb S \mathbb B)^{-1} \mathbb B^\top \mathbb S \otimes \mathbb B^t_3`
    (e.g. a weighted average in :math:`\eta_2`).
    """

    comps: tuple[int, ...]
    # poloidal factor of each term ("M", "S" or a key of B_3), without boundary conditions
    A_pol: dict[str, spa.spmatrix]
    M_3: np.ndarray
    S_3: np.ndarray | None = None
    # further 1d factors in eta_3 (e.g. "I": identity), approximated by their diagonal in the eigenbasis
    B_3: dict[str, np.ndarray] = field(default_factory=dict)
    # low-rank terms coeff * (S B)(B^T S B)^{-1}(S B)^T x B_3[key] (or M_3 for key "M"), as (coeff, S, B, key)
    low_rank: list = field(default_factory=list)
    # masks of the unknowns free of essential boundary conditions (None: all free)
    free_pol: np.ndarray | None = None
    free_3: np.ndarray | None = None
    pole: np.ndarray | None = field(default=None, repr=False)


def fast_diagonalization_3(
    M_3: np.ndarray, S_3: np.ndarray | None, free_3: np.ndarray | None = None, B_3: dict | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    r"""
    Basis :math:`V` of :math:`\eta_3` in which the 1d factors of a :class:`KronSum2Dx1DGroup` are diagonal.

    With ``S_3``, :math:`V` solves the generalized eigenvalue problem :math:`S_3 V = M_3 V \Lambda`
    (:math:`V^\top M_3 V = I`, :math:`V^\top S_3 V = \Lambda`). Without ``S_3``, :math:`V` are the orthonormal
    eigenvectors of :math:`M_3`. Further 1d matrices ``B_3`` (e.g. the identity) are approximated by the diagonal of
    :math:`V^\top B V` (exact if they are diagonal in this basis, e.g. the M_3-weighted average for periodic :math:`\eta_3`).
    The unknowns with essential boundary conditions (``~free_3``) are decoupled: each of them is a mode of its own,
    on which the inverse is the identity.

    Returns
    -------
    V : np.ndarray
        The basis (columns), with :math:`\mathbb A^{-1} = (I \otimes V) \bigoplus_k (\sum_t q_{t,k} \mathbb A^t_\textnormal{pol})^{-1} (I \otimes V^\top)`.

    q : dict[str, np.ndarray]
        Diagonals :math:`q_t` of the 1d factors ``"M"``, ``"S"`` and those of ``B_3`` in this basis.

    constrained : np.ndarray[bool]
        The modes of the constrained unknowns.
    """
    n3 = M_3.shape[0]
    free = np.ones(n3, dtype=bool) if free_3 is None else np.asarray(free_3, dtype=bool)
    f = np.flatnonzero(free)
    nf = f.size
    Mf = M_3[np.ix_(f, f)]
    if S_3 is None:
        mu, Vf = eigh(Mf)
        qf = {"M": mu, "S": np.zeros(nf)}
    else:
        lam, Vf = eigh(S_3[np.ix_(f, f)], Mf)
        qf = {"M": np.ones(nf), "S": lam}
    for key, B in (B_3 or {}).items():
        q_key = np.einsum("ik,ij,jk->k", Vf, B[np.ix_(f, f)], Vf)
        # round-off to zero, so that modes with equal coefficients share their factorization
        q_key[np.abs(q_key) <= 1e-12 * max(np.abs(q_key).max(), 1e-300)] = 0.0
        qf[key] = q_key

    V = np.zeros((n3, n3))
    V[np.ix_(f, np.arange(nf))] = Vf
    V[np.flatnonzero(~free), np.arange(nf, n3)] = 1.0
    constrained = np.zeros(n3, dtype=bool)
    constrained[nf:] = True
    q = {t: np.concatenate((qf[t], np.full(n3 - nf, np.nan))) for t in qf}
    return V, q, constrained


def _low_rank_factors(S: spa.spmatrix, B: spa.spmatrix, free: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """U = D S D B and C = B^T D S D B (identity on empty rows) of a low-rank term, with the mask D of ``free``."""
    S = spa.csr_matrix(S) if free is None else apply_mask(spa.csr_matrix(S), free, False)
    U = (S @ B).toarray() if spa.issparse(B) else S @ B
    C = np.asarray(B.T @ U)
    diag = np.abs(np.diag(C))
    empty = diag <= 1e-14 * max(diag.max(), 1e-300)
    C[empty, empty] = 1.0
    return U, C


class _WoodburySolver:
    """Solver for A + alpha U C^{-1} U^T (Woodbury identity), given a solver for A."""

    def __init__(self, base, U: np.ndarray, C: np.ndarray, alpha: float):
        self._base = base
        self._U = U
        self._W = base.solve(U)
        self._T = lu_factor(C / alpha + U.T @ self._W)
        self.exact = base.exact

    def solve(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape[-1] == 0:
            return Y.copy()
        X = self._base.solve(Y)
        return X - self._W @ lu_solve(self._T, self._U.T @ X)


class PolarKronSum2Dx1DSolver(_PolarKron2Dx1DBase):
    r"""
    (Approximate) inverse of :math:`\bigoplus_G \sum_t \mathbb A^{G,t}_\textnormal{pol} \otimes \mathbb B^{G,t}_3`
    (see :class:`KronSum2Dx1DGroup`) on a PolarDerhamSpace, by fast diagonalization in :math:`\eta_3`
    (see :func:`fast_diagonalization_3`):

    .. math::

        \mathbb A^{-1} = (I \otimes V) \bigoplus_k \Big(\sum_t q_{t,k} \mathbb A^t_\textnormal{pol}\Big)^{-1} (I \otimes V^\top)\,.

    The poloidal matrices of the modes :math:`k` are factorized with :func:`make_poloidal_solver`, each
    process only those of the modes it holds after the redistribution in the poloidal plane; modes
    with the same coefficients share their factorization. Essential boundary conditions are imposed
    as identity rows.

    Parameters
    ----------
    space : PolarDerhamSpace
        Domain and codomain.

    groups : list[KronSum2Dx1DGroup]
        The factors of each group of components.

    poloidal_solver : str
        ``"schur"``, ``"ilu"`` or ``"lu"``.

    **solver_kwargs
        Passed to :func:`make_poloidal_solver` (``schur_outer``, ``drop_tol``, ``fill_factor``).
    """

    def __init__(
        self, space: PolarDerhamSpace, groups: list[KronSum2Dx1DGroup], poloidal_solver: str = "schur", **solver_kwargs
    ):
        super().__init__(space, groups)
        self._kind = poloidal_solver
        self._solver_kwargs = solver_kwargs

        self._V = []
        self._mode_solvers = []
        for g in groups:
            V, q, constrained = fast_diagonalization_3(g.M_3, g.S_3, g.free_3, g.B_3)
            self._V.append(V)
            pole = g.pole if g.pole is not None else pole_mask(space, g.comps)
            low_rank = [(coeff, *_low_rank_factors(S, B, g.free_pol), key) for coeff, S, B, key in g.low_rank]

            # factorizations of the modes held by this process (key None: constrained mode, identity)
            s3 = tp_space(space, g.comps[0]).starts[2]
            k0, k1 = self._tpl[g.comps[0]].my_chunk
            solvers = {}
            keys = {}
            for k in range(s3 + k0, s3 + k1):
                if constrained[k]:
                    keys[k] = None
                    continue
                coeffs = {t: q[t][k] for t in g.A_pol}
                alphas = [coeff * q[key][k] for coeff, _, _, key in low_rank]
                key = tuple(f"{coeffs[t]:.12e}" for t in sorted(coeffs)) + tuple(f"{a:.12e}" for a in alphas)
                keys[k] = key
                if key not in solvers:
                    A = sum(c * g.A_pol[t] for t, c in coeffs.items() if c != 0.0)
                    if g.free_pol is not None:
                        A = apply_mask(spa.csr_matrix(A), g.free_pol, True)
                    solver = make_poloidal_solver(spa.csr_matrix(A), poloidal_solver, pole=pole, **solver_kwargs)
                    for alpha, (_, U, C, _) in zip(alphas, low_rank):
                        if alpha != 0.0:
                            solver = _WoodburySolver(solver, U, C, alpha)
                    solvers[key] = solver
            self._mode_solvers.append((solvers, keys))
            logger.debug(f"PolarKronSum2Dx1DSolver: group {g.comps}, {len(solvers)} poloidal factorizations")

    @property
    def exact(self) -> bool:
        """Whether the poloidal solves are exact (else the solver is an approximate inverse, e.g. a preconditioner).
        Given by the configuration, the same on all processes (also on those without factorizations)."""
        return self._kind != "ilu" and not (self._kind == "schur" and self._solver_kwargs.get("schur_outer") == "ilu")

    @property
    def poloidal_solver(self) -> str:
        return self._kind

    @staticmethod
    def _solve_modes(Y: np.ndarray, modes: np.ndarray, solvers: dict, keys: dict) -> np.ndarray:
        out = Y.copy()
        by_key = {}
        for j, k in enumerate(modes):
            by_key.setdefault(keys[int(k)], []).append(j)
        for key, cols in by_key.items():
            if key is not None:
                out[:, cols] = solvers[key].solve(Y[:, cols])
        return out

    def dot(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        ops_3 = [lambda X, V=V: V.T @ X for V in self._V]
        ops_pol = [lambda Y, modes, s=s, k=k: self._solve_modes(Y, modes, s, k) for s, k in self._mode_solvers]
        ops_3_post = [lambda X, V=V: V @ X for V in self._V]
        return self._apply(v, out, ops_3, ops_pol, ops_3_post)

    def solve(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        return self.dot(v, out=out)

    def transpose(self, conjugate=False) -> "PolarKronSum2Dx1DSolver":
        """The factors are symmetric, so is the inverse."""
        return self


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def _scalar_spaces(fem_space) -> tuple[TensorFemSpace, ...]:
    return (fem_space,) if isinstance(fem_space, TensorFemSpace) else fem_space.spaces


def spline_bases(fem_space) -> list[list[str]]:
    """``bases[c][d]``: spline basis (``"B"`` or ``"M"``) of component c in direction d of a Tensor-/VectorFemSpace."""
    return [[s.basis for s in comp.spaces] for comp in _scalar_spaces(fem_space)]


# --------------------------------------------------------------------------------------
# Weighted mass matrices
# --------------------------------------------------------------------------------------
# number of Gauss-Legendre points in eta_3 for averaging callable weights
_N_AVG_3 = 16


def _serial_attrs_1d(derham, basis: str, d: int):
    """Spline attributes of the serial 1d space (B- or M-splines) in direction d."""
    return derham.H1_1d_serial_splines[d] if basis == "B" else derham.L2_1d_serial_splines[d]


def _global_quad_pts(derham, d: int) -> np.ndarray:
    """Global quadrature points (flattened over the cells) in direction d."""
    return xp.to_numpy(derham.H1_1d_serial_splines[d].quad_grid_pts[0][0]).ravel()


def reduced_weight_2d(weight, derham, reduction: str = "midpoint") -> np.ndarray | None:
    r"""
    Weight of a 3d mass matrix reduced in :math:`\eta_3`, at the global quadrature points of the poloidal plane.

    Parameters
    ----------
    weight : callable | np.ndarray | None
        A block of the weights of the 3d mass operator: a function of the logical coordinates,
        or its values at the local quadrature points (then this function is collective on ``derham.comm``).

    derham : Derham
        The discrete de Rham sequence.

    reduction : str
        ``"midpoint"`` (value at :math:`\eta_3 = 0.5`) or ``"average"`` (mean over :math:`\eta_3`).

    Returns
    -------
    np.ndarray | None
        The reduced weight of shape (number of global quadrature points in eta_1, in eta_2).
    """
    assert reduction in ("midpoint", "average"), f"Unknown weight reduction {reduction!r}."
    if weight is None:
        return None

    pts1 = _global_quad_pts(derham, 0)
    pts2 = _global_quad_pts(derham, 1)

    if callable(weight):
        if reduction == "midpoint":
            pts3, wts3 = np.array([0.5]), np.array([1.0])
        else:
            nodes, wts = np.polynomial.legendre.leggauss(_N_AVG_3)
            pts3, wts3 = (nodes + 1.0) / 2.0, wts / 2.0
        E1, E2, E3 = np.meshgrid(pts1, pts2, pts3, indexing="ij")
        return xp.to_numpy(weight(xp.asarray(E1), xp.asarray(E2), xp.asarray(E3))).reshape(E1.shape) @ wts3

    assert isinstance(weight, np.ndarray) or xp.is_gpu(weight), (
        f"Weight must be callable, an array or None, but is {type(weight)}."
    )
    weight = xp.to_numpy(weight)
    dom = derham.domain_decomposition
    nq = derham.nquads
    s1, s2, s3 = (dom.starts[d] * nq[d] for d in range(3))
    l1, l2, l3 = weight.shape

    # reduced weight (index 0) and normalization (index 1), summed over all processes
    sums = np.zeros((2, pts1.size, pts2.size), dtype=float)
    if reduction == "midpoint":
        mid = (derham.num_elements[2] * nq[2]) // 2 - s3
        if 0 <= mid < l3:
            sums[0, s1 : s1 + l1, s2 : s2 + l2] = weight[:, :, mid]
            sums[1, s1 : s1 + l1, s2 : s2 + l2] = 1.0
    else:
        wts3 = xp.to_numpy(derham.spline_attributes["H1"].quad_grid_wts[0][2]).ravel()
        assert wts3.size == l3
        sums[0, s1 : s1 + l1, s2 : s2 + l2] = weight @ wts3
        sums[1, s1 : s1 + l1, s2 : s2 + l2] = wts3.sum()

    comm = derham.comm
    if comm is not None and comm.Get_size() > 1:
        comm.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
    return sums[0] / sums[1]


def poloidal_mass_matrix(derham, bases_row: tuple, bases_col: tuple, weight: np.ndarray | None) -> spa.csr_matrix:
    r"""
    Weighted mass matrix :math:`\int \Lambda_{i_1 i_2} \, w \, \Lambda_{j_1 j_2} \, \textnormal d\eta_1 \textnormal d\eta_2`
    of the poloidal plane (C-ordering of :math:`(i_1, i_2)`), assembled on the serial 1d spline spaces.

    Parameters
    ----------
    derham : Derham
        The discrete de Rham sequence.

    bases_row, bases_col : tuple[str]
        Spline bases (``"B"`` or ``"M"``) in eta_1 and eta_2 of the row (test) and column (trial) functions.

    weight : np.ndarray | None
        Weight at the global quadrature points of the poloidal plane (None: unit weight).
    """
    att_i = [_serial_attrs_1d(derham, bases_row[d], d) for d in range(2)]
    att_j = [_serial_attrs_1d(derham, bases_col[d], d) for d in range(2)]

    def data(att):
        spans = xp.to_numpy(att.quad_grid_spans[0][0])
        basis = xp.to_numpy(att.quad_grid_bases[0][0])[:, :, 0, :]  # (cells, p+1, quad points), no derivatives
        wts = xp.to_numpy(att.quad_grid_wts[0][0])  # (cells, quad points)
        n = att.nbasis[0][0]
        p = basis.shape[1] - 1
        return spans, basis, wts, n, p

    (si1, bi1, w1, ni1, pi1), (si2, bi2, w2, ni2, pi2) = (data(a) for a in att_i)
    (sj1, bj1, _, nj1, pj1), (sj2, bj2, _, nj2, pj2) = (data(a) for a in att_j)

    ne1, nq1 = w1.shape
    ne2, nq2 = w2.shape
    f = np.ones((ne1, nq1, ne2, nq2)) if weight is None else weight.reshape(ne1, nq1, ne2, nq2)
    f = f * w1[:, :, None, None] * w2[None, None, :, :]

    # element matrices (cells1, cells2, a1, a2, b1, b2)
    loc = np.einsum("xqyr,xaq,ybr,xcq,ydr->xyabcd", f, bi1, bi2, bj1, bj2, optimize=True)

    def global_index(spans, p, n):
        # (cells, p+1) global basis indices (periodic wrap-around)
        return (spans[:, None] - p + np.arange(p + 1)[None, :]) % n

    gi1 = global_index(si1, pi1, ni1)
    gi2 = global_index(si2, pi2, ni2)
    gj1 = global_index(sj1, pj1, nj1)
    gj2 = global_index(sj2, pj2, nj2)

    rows = (gi1[:, None, :, None] * ni2 + gi2[None, :, None, :])[:, :, :, :, None, None]
    cols = (gj1[:, None, :, None] * nj2 + gj2[None, :, None, :])[:, :, None, None, :, :]
    rows, cols = np.broadcast_arrays(rows, cols)
    M = spa.coo_matrix((loc.ravel(), (rows.ravel(), cols.ravel())), shape=(ni1 * ni2, nj1 * nj2))
    return M.tocsr()
