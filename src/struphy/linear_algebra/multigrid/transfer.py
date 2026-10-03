r"""Grid-transfer operators between nested spline spaces.

On uniformly refined grids the spline spaces are nested, :math:`V_H \subset V_h`, hence every coarse
basis function is a linear combination of fine ones. The prolongation :math:`P: V_H \to V_h` maps the
coefficients of a coarse spline to the coefficients of the *same* function in the fine basis; the
restriction is its transpose :math:`R = P^\top`. Both are tensor products of 1D matrices, applied
component-wise for vector-valued spaces.
"""

import numpy as np
import scipy.sparse as spa
from feectools.core.bsplines import collocation_matrix
from feectools.fem.splines import SplineSpace
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import IdentityOperator, LinearOperator, Vector
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector, StencilVectorSpace

from struphy.feec.psydac_derham import Derham


def prolongation_matrix_1d(coarse: SplineSpace, fine: SplineSpace) -> np.ndarray:
    r"""Dense 1D prolongation matrix :math:`P \in \mathbb R^{n_h \times n_H}` between nested spline spaces.

    Column ``j`` holds the fine coefficients of the coarse basis function ``j``, i.e.
    :math:`\Lambda^H_j = \sum_i P_{ij} \Lambda^h_i`. It is computed by collocation at ``degree + 1``
    points per fine cell (an overdetermined, exactly solvable system). Works for periodic and clamped
    splines and for both normalizations (B-splines and D-splines/M-splines).

    Parameters
    ----------
    coarse, fine : SplineSpace
        1D spaces of the same degree, kind and normalization; the breaks of ``coarse`` are a subset of those of ``fine``.
    """
    assert coarse.degree == fine.degree
    assert coarse.periodic == fine.periodic
    assert coarse.basis == fine.basis

    if coarse.ncells == fine.ncells:
        return np.eye(fine.nbasis)

    breaks = np.asarray(fine.breaks)
    nq = fine.degree + 1
    s = (np.arange(nq) + 0.5) / nq
    x = (breaks[:-1, None] + np.diff(breaks)[:, None] * s[None, :]).ravel()

    Bh = collocation_matrix(fine.knots, fine.degree, fine.periodic, fine.basis, x)
    BH = collocation_matrix(coarse.knots, coarse.degree, coarse.periodic, coarse.basis, x)
    P = np.linalg.lstsq(Bh, BH, rcond=None)[0]

    assert np.allclose(Bh @ P, BH, atol=1e-10), "Spline spaces are not nested."
    P[np.abs(P) < 1e-13 * np.abs(P).max()] = 0.0
    return P


def local_matrix_1d(
    A: np.ndarray,
    out_start: int,
    out_end: int,
    in_start: int,
    in_end: int,
    in_ghost: int,
    periodic: bool,
) -> spa.csr_matrix:
    r"""Restrict a global 1D matrix to the rows owned by this process and to local (ghosted) input columns.

    Row ``r`` of the result is row ``out_start + r`` of ``A``. Global column ``c`` is mapped to the index
    of the local ghosted input array, ``c - in_start + in_ghost``, using periodic images if ``periodic``.
    If a column is present several times (owned and as ghost) the owned copy is used.

    Parameters
    ----------
    A : numpy.ndarray
        Global matrix of shape ``(n_out, n_in)``.

    out_start, out_end : int
        Global indices of the first and last output entry owned by this process.

    in_start, in_end : int
        Global indices of the first and last input entry owned by this process.

    in_ghost : int
        Width of the ghost region on each side of the local input array (``pads * shifts``).

    periodic : bool
        Whether the input index is periodic.
    """
    n_in = A.shape[1]
    n_loc = in_end - in_start + 1 + 2 * in_ghost

    rows, cols, vals = [], [], []
    for r, row in enumerate(range(out_start, out_end + 1)):
        for c in np.flatnonzero(A[row]):
            images = [c - n_in, c, c + n_in] if periodic else [c]
            local = [g - in_start + in_ghost for g in images]
            owned = [l for l in local if in_ghost <= l < n_loc - in_ghost]
            ghost = [l for l in local if 0 <= l < n_loc]
            if owned:
                loc = owned[0]
            elif ghost:
                loc = ghost[0]
            else:
                raise ValueError(f"Column {c} of row {row} is outside the local ghost region.")
            rows.append(r)
            cols.append(loc)
            vals.append(A[row, c])

    return spa.csr_matrix((vals, (rows, cols)), shape=(out_end - out_start + 1, n_loc))


class _KronTransfer:
    """Tensor product of three local 1D matrices mapping a ghosted input StencilVector to the owned part of the output."""

    def __init__(self, mats: list[spa.csr_matrix], W: StencilVectorSpace):
        self._mats = mats
        self._out_slice = tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(W.pads, W.shifts, W.starts, W.ends))

    def dot(self, v: StencilVector, out: StencilVector) -> None:
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()
        x = v._data
        for axis, L in enumerate(self._mats):
            x = np.moveaxis(x, axis, 0)
            shp = x.shape
            x = (L @ x.reshape(shp[0], -1)).reshape((L.shape[0],) + shp[1:])
            x = np.moveaxis(x, 0, axis)
        out._data[...] = 0.0
        out._data[self._out_slice] = x
        out.ghost_regions_in_sync = False


def _scalar_spaces(V) -> list[TensorFemSpace]:
    """Scalar components of a (vector) FEM space."""
    return [V] if isinstance(V, TensorFemSpace) else list(V.spaces)


def _coeff_spaces(W) -> list[StencilVectorSpace]:
    """Scalar components of a (block) coefficient space."""
    return [W] if isinstance(W, StencilVectorSpace) else list(W.spaces)


class SplineProlongation(LinearOperator):
    r"""Prolongation :math:`P: V_H \to V_h` (or, with ``transposed=True``, restriction :math:`R = P^\top`)
    between the same space of two nested Derham sequences.

    The MPI decompositions must be aligned (each process owns the coarse elements covering its fine
    elements), as produced by :meth:`DomainDecomposition.coarsen`. With homogeneous Dirichlet boundary
    conditions, the operator is :math:`\mathbb B_h P \mathbb B_H^\top` (resp. its transpose), which is the
    exact embedding of the coarse into the fine space with boundary conditions.

    Parameters
    ----------
    coarse, fine : Derham
        Coarse and fine level.

    space_id : str
        Space key of ``Derham.fem_spaces`` ("0", "1", "2", "3", "v" or "H1", "Hcurl", "Hdiv", "L2", "H1vec").

    transposed : bool
        If True, the restriction :math:`R = P^\top` (fine to coarse) is created.
    """

    def __init__(self, coarse: Derham, fine: Derham, space_id: str, *, transposed: bool = False):
        if coarse.polar_splines or fine.polar_splines:
            raise NotImplementedError("Grid transfer is not yet implemented for polar splines.")

        self._coarse = coarse
        self._fine = fine
        self._space_id = space_id
        self._transposed = transposed

        form = coarse.space_to_form.get(space_id, space_id)
        self._form = form
        VH = coarse.fem_spaces[form]
        Vh = fine.fem_spaces[form]

        self._Bc = coarse.boundary_ops[form]
        self._Bf = fine.boundary_ops[form]
        self._apply_bc = not (isinstance(self._Bc, IdentityOperator) and isinstance(self._Bf, IdentityOperator))

        if transposed:
            self._domain, self._codomain = fine.coeff_spaces[form], coarse.coeff_spaces[form]
            V_in, V_out = Vh, VH
        else:
            self._domain, self._codomain = coarse.coeff_spaces[form], fine.coeff_spaces[form]
            V_in, V_out = VH, Vh

        self._kron = []
        for cH, ch, Win, Wout in zip(
            _scalar_spaces(VH),
            _scalar_spaces(Vh),
            _coeff_spaces(self._domain),
            _coeff_spaces(self._codomain),
        ):
            mats = []
            for axis, (sH, sh) in enumerate(zip(cH.spaces, ch.spaces)):
                P = prolongation_matrix_1d(sH, sh)
                A = P.T if transposed else P
                mats.append(
                    local_matrix_1d(
                        A,
                        int(Wout.starts[axis]),
                        int(Wout.ends[axis]),
                        int(Win.starts[axis]),
                        int(Win.ends[axis]),
                        int(Win.pads[axis] * Win.shifts[axis]),
                        sh.periodic,
                    )
                )
            self._kron.append(_KronTransfer(mats, Wout))

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._domain.dtype

    @property
    def space_id(self) -> str:
        return self._space_id

    @property
    def transposed(self) -> bool:
        return self._transposed

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        """Apply the operator, ``out = P v`` (or ``out = R v`` if transposed)."""
        assert isinstance(v, Vector) and v.space == self.domain
        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector) and out.space == self.codomain

        B_in, B_out = (self._Bf, self._Bc) if self._transposed else (self._Bc, self._Bf)
        if self._apply_bc:
            v = B_in.dot(v)

        if isinstance(v, BlockVector):
            for k, vk, ok in zip(self._kron, v.blocks, out.blocks):
                k.dot(vk, ok)
        else:
            self._kron[0].dot(v, out)

        if self._apply_bc:
            B_out.dot(out, out=out)
        return out

    def transpose(self, conjugate: bool = False) -> "SplineProlongation":
        return SplineProlongation(self._coarse, self._fine, self._space_id, transposed=not self._transposed)
