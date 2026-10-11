r"""
2D x 1D Kronecker structure of linear operators on polar spline spaces.

Polar extraction operators only act in the poloidal plane :math:`(\eta_1, \eta_2)` and are the identity
in :math:`\eta_3`. Inter-/histopolation matrices of the commuting projectors and (axisymmetric) weighted
mass matrices thus have the structure

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

import numpy as np
import scipy.sparse as spa
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import LinearOperator
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector
from maybempi import MPI
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spilu, splu

from struphy.polar.basic import PolarDerhamSpace, PolarVector

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


# --------------------------------------------------------------------------------------
# Poloidal solvers
# --------------------------------------------------------------------------------------
class _SparseLU:
    """Exact (LU) or incomplete (ILU) factorization of a sparse matrix."""

    def __init__(self, A: spa.spmatrix, kind: str = "lu", drop_tol=ILU_DROP_TOL, fill_factor=ILU_FILL_FACTOR):
        assert kind in ("lu", "ilu")
        A = spa.csc_matrix(A)
        if kind == "lu":
            self._fac = splu(A)
        else:
            self._fac = spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        self.exact = kind == "lu"

    def solve(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape[-1] == 0:
            return Y.copy()
        return self._fac.solve(np.asarray(Y, order="F"))


class _KroneckerBlockDiagonal:
    """Exact solver for a block diagonal matrix with Kronecker blocks A1 x A2 (C-ordering of the unknowns)."""

    def __init__(self, factors: list[tuple[np.ndarray, np.ndarray]]):
        self._sizes = []
        self._lus = []
        for A1, A2 in factors:
            self._sizes.append((A1.shape[0], A2.shape[0]))
            self._lus.append((splu(spa.csc_matrix(A1)), splu(spa.csc_matrix(A2))))
        self.exact = True

    def solve(self, Y: np.ndarray) -> np.ndarray:
        out = np.empty_like(Y)
        m = Y.shape[-1]
        off = 0
        for (n1, n2), (lu1, lu2) in zip(self._sizes, self._lus):
            X = Y[off : off + n1 * n2].reshape(n1, n2 * m)
            X = lu1.solve(np.asarray(X, order="F")).reshape(n1, n2, m)
            X = np.ascontiguousarray(X.transpose(1, 0, 2)).reshape(n2, n1 * m)
            X = lu2.solve(np.asarray(X, order="F")).reshape(n2, n1, m)
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
        self._n = n
        logger.debug(f"Schur solver: {n = }, pole block {self._p.size}, outer block {self._o.size}")

    def solve(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape[-1] == 0:
            return Y.copy()
        Yp = Y[self._p]
        Yo = Y[self._o]
        t = self._outer.solve(Yo)
        Xp = lu_solve(self._S, Yp - self._Apo @ t)
        Xo = self._outer.solve(Yo - self._Aop @ Xp)
        X = np.empty_like(Y)
        X[self._p] = Xp
        X[self._o] = Xo
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
    if kind in ("lu", "ilu"):
        return _SparseLU(A, kind, drop_tol=drop_tol, fill_factor=fill_factor)

    assert pole is not None
    if outer_kronecker is not None:
        outer = _KroneckerBlockDiagonal(outer_kronecker)
        if __debug__:
            # the outer block must match the Kronecker factors
            o = np.flatnonzero(~pole)
            Aoo = spa.csr_matrix(A)[o][:, o]
            K = spa.block_diag([spa.kron(spa.csr_matrix(A1), spa.csr_matrix(A2)) for A1, A2 in outer_kronecker])
            diff = abs(Aoo - K).max() if Aoo.nnz + K.nnz > 0 else 0.0
            assert diff <= 1e-12 * max(abs(Aoo).max(), 1.0), f"Outer block is not Kronecker ({diff = })."
    else:
        o = np.flatnonzero(~pole)
        outer = _SparseLU(spa.csr_matrix(A)[o][:, o], schur_outer, drop_tol=drop_tol, fill_factor=fill_factor)
    return _SchurSolver(A, pole, outer)


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

        sendbuf = np.concatenate([local[..., a:b].ravel() for a, b in self._chunks])
        recvbuf = np.empty(int(self._recv_counts.sum()), dtype=float)
        self._comm.Alltoallv(
            [sendbuf, (self._send_counts, self._send_displs), MPI.DOUBLE],
            [recvbuf, (self._recv_counts, self._recv_displs), MPI.DOUBLE],
        )
        full = np.empty((*self._global_shape, c1 - c0), dtype=float)
        for (s, e), d, n in zip(self._boxes, self._recv_displs, self._recv_counts):
            shape = tuple(ei - si + 1 for si, ei in zip(s, e))
            full[tuple(slice(si, ei + 1) for si, ei in zip(s, e))] = recvbuf[d : d + n].reshape(*shape, c1 - c0)
        return full

    def backward(self, full: np.ndarray) -> np.ndarray:
        """(*global_shape, m_chunk) -> (*local_shape, m_local)."""
        if self._comm is None:
            return full

        sendbuf = np.concatenate(
            [full[tuple(slice(si, ei + 1) for si, ei in zip(s, e))].ravel() for s, e in self._boxes],
        )
        recvbuf = np.empty(int(self._send_counts.sum()), dtype=float)
        self._comm.Alltoallv(
            [sendbuf, (self._recv_counts, self._recv_displs), MPI.DOUBLE],
            [recvbuf, (self._send_counts, self._send_displs), MPI.DOUBLE],
        )
        local = np.empty((*self._local_shape, self._m_local), dtype=float)
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

    def __init__(self, space: PolarDerhamSpace, groups: list[Kron2Dx1DGroup]):
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

    def _apply(self, v: PolarVector, out: PolarVector | None, ops_3: list, ops_pol: list) -> PolarVector:
        """Apply ops_3[g] (to (n3, m) arrays) along eta_3 and ops_pol[g] (to (N_pol, m) arrays) in the poloidal plane."""
        assert isinstance(v, PolarVector)
        assert v.space == self._space
        if out is None:
            out = PolarVector(self._space)
        else:
            assert isinstance(out, PolarVector)
            assert out.space == self._space

        V = self._space
        out_pol = [np.zeros_like(p) for p in v.pol]

        for g, op_3, op_pol in zip(self._groups, ops_3, ops_pol):
            # 1. eta_3 direction (tensor-product part redistributed, polar coeffs replicated)
            tmp_tp = {}
            tmp_pol = {}
            for c in g.comps:
                L = _interior(tp_component(v.tp, c))
                l1, l2, l3 = L.shape
                T3 = self._t3[c]
                X = T3.forward(np.ascontiguousarray(L.reshape(l1 * l2, l3).T))
                tmp_tp[c] = np.ascontiguousarray(T3.backward(op_3(X)).T).reshape(l1, l2, l3)
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
                blocks.append(full[c][nr:].reshape(-1, k1 - k0))
            Z = op_pol(np.concatenate(blocks, axis=0)) if k1 > k0 else np.concatenate(blocks, axis=0)

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
                comm.Allreduce(MPI.IN_PLACE, out_pol[c], op=MPI.SUM)
            out.pol[c][:] = out_pol[c]

        return out


class PolarKron2Dx1DOperator(_PolarKron2Dx1DBase):
    r"""
    Linear operator :math:`\bigoplus_G \mathbb A^G_\textnormal{pol} \otimes \mathbb A^G_3` on a PolarDerhamSpace.

    Parameters
    ----------
    space : PolarDerhamSpace
        Domain and codomain.

    groups : list[Kron2Dx1DGroup]
        The factors of each group of components.
    """

    def dot(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        ops_3 = [lambda X, A=g.A_3: A @ X for g in self._groups]
        ops_pol = [lambda Y, A=g.A_pol: A @ Y for g in self._groups]
        return self._apply(v, out, ops_3, ops_pol)

    def transpose(self, conjugate=False) -> "PolarKron2Dx1DOperator":
        return PolarKron2Dx1DOperator(self._space, [g.transpose() for g in self._groups])


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
        self._lu_3 = [splu(spa.csc_matrix(g.A_3)) for g in groups]
        self._pol_solvers = [
            make_poloidal_solver(
                g.A_pol,
                poloidal_solver,
                pole=g.pole if g.pole is not None else pole_mask(space, g.comps),
                outer_kronecker=g.outer_kronecker,
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
        ops_3 = [lambda X, lu=lu: lu.solve(np.asarray(X, order="F")) for lu in self._lu_3]
        ops_pol = [s.solve for s in self._pol_solvers]
        return self._apply(v, out, ops_3, ops_pol)

    def solve(self, v: PolarVector, out: PolarVector | None = None) -> PolarVector:
        return self.dot(v, out=out)

    def transpose(self, conjugate=False) -> "PolarKron2Dx1DSolver":
        return PolarKron2Dx1DSolver(
            self._space, [g.transpose() for g in self._groups], self._kind, **self._solver_kwargs
        )


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def _scalar_spaces(fem_space) -> tuple[TensorFemSpace, ...]:
    return (fem_space,) if isinstance(fem_space, TensorFemSpace) else fem_space.spaces


def spline_bases(fem_space) -> list[list[str]]:
    """``bases[c][d]``: spline basis (``"B"`` or ``"M"``) of component c in direction d of a Tensor-/VectorFemSpace."""
    return [[s.basis for s in comp.spaces] for comp in _scalar_spaces(fem_space)]


def _spline_space_1d(fem_space, c: int, d: int):
    return _scalar_spaces(fem_space)[c].spaces[d]


def collocation_matrix_1d(fem_space, c: int, d: int) -> np.ndarray:
    """1d interpolation (B-splines) or histopolation (M-splines) matrix of component c in direction d."""
    S = _spline_space_1d(fem_space, c, d)
    return np.asarray(S.imat if S.basis == "B" else S.hmat, dtype=float)


def projector_groups(
    space: PolarDerhamSpace, fem_space, dofs_extraction_op, base_extraction_op
) -> list[Kron2Dx1DGroup]:
    r"""
    Factors of the polar inter-/histopolation matrix :math:`\mathbb P \mathcal I \mathbb E^T = \bigoplus_G (\mathbb P_\textnormal{pol} (\mathcal I_1 \otimes \mathcal I_2) \mathbb E_\textnormal{pol}^T) \otimes \mathcal I_3`.

    Parameters
    ----------
    space : PolarDerhamSpace
        The polar space.

    fem_space : TensorFemSpace | VectorFemSpace
        The tensor-product FE space of the projector.

    dofs_extraction_op, base_extraction_op : PolarExtractionOperator
        DOF and basis extraction operators.
    """
    groups = []
    for comps in component_groups(space):
        P = poloidal_extraction_matrix(
            space, comps, dofs_extraction_op.blocks_ten_to_pol, dofs_extraction_op.blocks_ten_to_ten
        )
        E = poloidal_extraction_matrix(space, comps, base_extraction_op.blocks_ten_to_pol, None)
        tp_blocks = []
        outer = []
        for c in comps:
            A1 = collocation_matrix_1d(fem_space, c, 0)
            A2 = collocation_matrix_1d(fem_space, c, 1)
            tp_blocks.append(spa.kron(spa.csr_matrix(A1), spa.csr_matrix(A2)))
            nr = space.n_rings[c]
            outer.append((A1[nr + 1 :, nr + 1 :], A2))
        A_pol = (P @ spa.block_diag(tp_blocks, format="csr") @ E.T).tocsr()
        A_pol.eliminate_zeros()
        A_3 = collocation_matrix_1d(fem_space, comps[0], 2)
        groups.append(Kron2Dx1DGroup(comps, A_pol, A_3, outer, pole_mask(space, comps)))
    return groups


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
    return np.asarray(derham.H1_1d_serial_splines[d].quad_grid_pts[0][0]).ravel()


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
        return np.asarray(weight(E1, E2, E3)).reshape(E1.shape) @ wts3

    assert isinstance(weight, np.ndarray), f"Weight must be callable, np.ndarray or None, but is {type(weight)}."
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
        wts3 = np.asarray(derham.spline_attributes["H1"].quad_grid_wts[0][2]).ravel()
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
        spans = np.asarray(att.quad_grid_spans[0][0])
        basis = np.asarray(att.quad_grid_bases[0][0])[:, :, 0, :]  # (cells, p+1, quad points), no derivatives
        wts = np.asarray(att.quad_grid_wts[0][0])  # (cells, quad points)
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
