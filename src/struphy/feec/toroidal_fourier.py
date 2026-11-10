r"""Exact solver for linear operators that are invariant under shifts in the periodic direction :math:`\eta_3`.

On a uniform, periodic grid in :math:`\eta_3`, the spline coefficients of each vector component are indexed by
:math:`(p, j)`, with the index :math:`p` in the poloidal plane :math:`(\eta_1, \eta_2)` (including the polar
coefficients of polar splines) and :math:`j = 0, \dots, N_3 - 1` along :math:`\eta_3`. If the weights of the operator do
not depend on :math:`\eta_3` (e.g. an axisymmetric equilibrium in a torus or a cylinder), the operator commutes with the
shifts :math:`j \mapsto j + 1`, i.e. it is block-circulant, :math:`A_{jk} = a_{j - k}`, and the discrete Fourier
transform along :math:`\eta_3` splits it into one poloidal matrix per toroidal mode number :math:`n`,

.. math::

    \hat A(n) = \sum_{l=0}^{N_3 - 1} a_l \, e^{-2 \pi i n l / N_3}\,, \qquad n = 0, \dots, N_3 / 2\,,

(:math:`\hat A(N_3 - n) = \overline{\hat A(n)}` for a real operator). Here the matrices :math:`\hat A(n)` are
assembled by probing: :math:`A` is applied to the unit vectors :math:`e_p \otimes \delta_{j0}` of all poloidal
unknowns :math:`p`, whose images are the columns :math:`p` of all :math:`a_l`. Each :math:`\hat A(n)` is factorized
(Cholesky for Hermitian positive definite matrices, else LU), and the inverse is applied by a real FFT along
:math:`\eta_3`, one poloidal solve per toroidal mode number and the inverse FFT. This is the exact inverse of
:math:`A` (to round-off) whenever :math:`A` is shift invariant, including all couplings between the poloidal and the
toroidal directions and between the vector components; otherwise it is an approximation (its quality is checked on
a random vector, see :attr:`ToroidalFourierSolver.residual`).

Cost: the setup needs one application of :math:`A` per poloidal unknown and dense factorizations of size
:math:`N_\textnormal{pol}` (the number of unknowns in a poloidal plane, summed over the components); the toroidal mode
numbers are distributed over the processes. Each application gathers the vector on all processes (allgather of
:math:`N_\textnormal{pol} N_3` numbers). It is meant for moderate poloidal resolution (e.g. up to about 32 x 33
elements with three components).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg as sla
from feectools.linalg.basic import LinearOperator, Vector
from feectools.linalg.stencil import StencilVector

logger = logging.getLogger("struphy")


def _interior_slices(sv: StencilVector) -> tuple[slice, ...]:
    """Slices of the owned data of a StencilVector (without ghost regions) in its local array."""
    sp = sv.space
    return tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(sp.pads, sp.shifts, sp.starts, sp.ends))


class _Layout:
    """Global layout of the unknowns of a vector space as an array of shape (N_pol, N_3): per component, the polar
    coefficients (polar splines) followed by the tensor-product coefficients outside the polar rings."""

    def __init__(self, space):
        self.polar = hasattr(space, "parent_space")
        tp = space.parent_space if self.polar else space
        sample = tp.zeros()
        self.tp_spaces = [sample.space] if isinstance(sample, StencilVector) else [b.space for b in sample.blocks]
        self.n_comps = len(self.tp_spaces)
        self.n_rings = tuple(space.n_rings) if self.polar else (0,) * self.n_comps
        self.n_polar = tuple(space.n_polar) if self.polar else (0,) * self.n_comps
        n3 = {sp.npts[2] for sp in self.tp_spaces}
        assert len(n3) == 1, f"All components need the same number of coefficients along eta_3, got {n3}."
        self.n3 = n3.pop()
        for sp in self.tp_spaces:
            assert sp.periods[2], "The toroidal Fourier solver needs a periodic eta_3 direction."
        # row offsets of the components and of their tensor-product parts
        self.offsets, self.tp_offsets, self.tp_shapes = [], [], []
        row = 0
        for c, sp in enumerate(self.tp_spaces):
            n1, n2 = sp.npts[:2]
            self.offsets.append(row)
            self.tp_offsets.append(row + self.n_polar[c])
            self.tp_shapes.append((n1 - self.n_rings[c], n2))
            row += self.n_polar[c] + (n1 - self.n_rings[c]) * n2
        self.n_pol = row
        comm = self.tp_spaces[0].cart.comm_cart if self.tp_spaces[0].cart is not None else None
        self.comm = comm if comm is not None and comm.Get_size() > 1 else None

    def tp_blocks(self, v: Vector) -> list[StencilVector]:
        tp = v.tp if self.polar else v
        return [tp] if isinstance(tp, StencilVector) else list(tp.blocks)

    def gather(self, v: Vector) -> np.ndarray:
        """The global array (N_pol, N_3) of the unknowns of v (on all processes)."""
        X = np.zeros((self.n_pol, self.n3))
        pieces = []
        for c, sv in enumerate(self.tp_blocks(v)):
            sp = sv.space
            pieces.append((c, tuple(sp.starts), sv._data[_interior_slices(sv)].copy()))
        if self.comm is not None:
            pieces = [p for chunk in self.comm.allgather(pieces) for p in chunk]
        for c, starts, data in pieces:
            r = self.n_rings[c]
            s1, s2, s3 = starts
            e1 = s1 + data.shape[0]
            lo = max(s1, r)
            if lo >= e1:
                continue
            n1r, n2 = self.tp_shapes[c]
            block = X[self.tp_offsets[c] : self.tp_offsets[c] + n1r * n2].reshape(n1r, n2, self.n3)
            block[lo - r : e1 - r, s2 : s2 + data.shape[1], s3 : s3 + data.shape[2]] = data[lo - s1 :]
        if self.polar:
            for c, pol in enumerate(v.pol):
                X[self.offsets[c] : self.offsets[c] + self.n_polar[c]] = pol
        return X

    def scatter(self, X: np.ndarray, v: Vector) -> Vector:
        """Write the global array X (N_pol, N_3) into v (each process its own part; the polar rings are zero)."""
        for c, sv in enumerate(self.tp_blocks(v)):
            sp = sv.space
            r = self.n_rings[c]
            s1, s2, s3 = sp.starts
            e1, e2, e3 = (e + 1 for e in sp.ends)
            n1r, n2 = self.tp_shapes[c]
            block = X[self.tp_offsets[c] : self.tp_offsets[c] + n1r * n2].reshape(n1r, n2, self.n3)
            local = sv._data[_interior_slices(sv)]
            local[...] = 0.0
            lo = max(s1, r)
            if lo < e1:
                local[lo - s1 :] = block[lo - r : e1 - r, s2:e2, s3:e3]
        if self.polar:
            for c in range(self.n_comps):
                v.pol[c][:] = X[self.offsets[c] : self.offsets[c] + self.n_polar[c]]
        v.update_ghost_regions()
        return v


class ToroidalFourierSolver(LinearOperator):
    r"""Inverse of a linear operator :math:`A` (square, on Stencil-, Block- or PolarVectors) that is invariant under
    shifts in :math:`\eta_3`, by a real FFT along :math:`\eta_3` and one dense poloidal solve per toroidal mode number,
    see :mod:`struphy.feec.toroidal_fourier`. Use it as a preconditioner (``A`` then needs only to be close to shift
    invariant) or as a direct solver.

    Parameters
    ----------
    A : LinearOperator
        The operator to invert (collective ``dot``; it is applied once per poloidal unknown in the setup).

    check : bool
        Apply ``A`` to the preconditioned random vector and store the relative residual
        :math:`\|A P r - r\| / \|r\|` in :attr:`residual` (logged), a measure of the shift invariance of ``A``.

    seed : int
        Seed of the random vector of the check.
    """

    def __init__(self, A: LinearOperator, check: bool = True, seed: int = 1234):
        assert A.domain == A.codomain, "A must be square."
        self._space = A.domain
        self._layout = L = _Layout(self._space)
        n_modes = L.n3 // 2 + 1
        rank = 0 if L.comm is None else L.comm.Get_rank()
        size = 1 if L.comm is None else L.comm.Get_size()
        self._my_modes = [n for n in range(n_modes) if n % size == rank]
        self._n_modes = n_modes

        # probing: the column p of the Fourier blocks of A for all toroidal mode numbers
        mats = {n: np.zeros((L.n_pol, L.n_pol), dtype=complex) for n in self._my_modes}
        constrained = np.zeros(L.n_pol, dtype=bool)
        e = self._space.zeros()
        Ae = self._space.zeros()
        X = np.zeros((L.n_pol, L.n3))
        for p in range(L.n_pol):
            X[p, 0] = 1.0
            L.scatter(X, e)
            X[p, 0] = 0.0
            A.dot(e, out=Ae)
            Y = np.fft.rfft(L.gather(Ae), axis=1)
            if not np.any(Y):
                constrained[p] = True
            for n in self._my_modes:
                mats[n][:, p] = Y[:, n]

        # constrained unknowns (e.g. essential boundary conditions): identity
        scale = max((np.abs(M).max() for M in mats.values()), default=1.0)
        self._factors = {}
        self._cholesky = True
        for n, M in mats.items():
            idx = np.flatnonzero(constrained)
            M[idx, :] = 0.0
            M[:, idx] = 0.0
            M[idx, idx] = scale
            try:
                self._factors[n] = ("cho", sla.cho_factor(M, lower=False, check_finite=False))
            except np.linalg.LinAlgError:
                self._cholesky = False
                self._factors[n] = ("lu", sla.lu_factor(M, check_finite=False))
        if L.comm is not None:
            self._cholesky = L.comm.allreduce(self._cholesky, op=_mpi_land())
        self._constrained = constrained
        self._n_constrained = int(constrained.sum())

        self._residual = None
        if check:
            rng = np.random.default_rng(seed)
            R = rng.standard_normal((L.n_pol, L.n3))
            R[constrained] = 0.0
            r = L.scatter(R, self._space.zeros())
            Pr = self.dot(r)
            APr = A.dot(Pr)
            APr -= r
            self._residual = float(np.sqrt(APr.inner(APr) / r.inner(r)))
        logger.info(
            f"ToroidalFourierSolver: {L.n_pol} poloidal unknowns ({self._n_constrained} constrained), "
            f"{L.n3} toroidal coefficients, {n_modes} toroidal mode numbers, "
            f"{'Cholesky' if self._cholesky else 'LU'} factorizations"
            + ("" if self._residual is None else f", relative residual on a random vector {self._residual:.2e}")
        )

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
    def residual(self) -> float | None:
        r"""Relative residual :math:`\|A P r - r\| / \|r\|` on a random vector (None without ``check``): round-off
        for a shift invariant operator."""
        return self._residual

    @property
    def n_pol(self) -> int:
        """Number of unknowns in a poloidal plane (summed over the components)."""
        return self._layout.n_pol

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate: bool = False):
        """The inverse of a symmetric operator is symmetric."""
        return self

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        assert v.space == self._space
        L = self._layout
        B = np.fft.rfft(L.gather(v), axis=1)
        Xhat = np.zeros_like(B)
        for n in self._my_modes:
            kind, fac = self._factors[n]
            b = B[:, n].copy()
            b[self._constrained] = 0.0
            if kind == "cho":
                Xhat[:, n] = sla.cho_solve(fac, b, check_finite=False)
            else:
                Xhat[:, n] = sla.lu_solve(fac, b, check_finite=False)
        if L.comm is not None:
            Xhat = L.comm.allreduce(Xhat)
        X = np.fft.irfft(Xhat, n=L.n3, axis=1)
        if out is None:
            out = self._space.zeros()
        else:
            assert out.space == self._space
        return L.scatter(X, out)

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        return self.dot(rhs, out=out)


def _mpi_land():
    from maybempi import MPI

    return MPI.LAND
