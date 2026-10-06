r"""Geometric multigrid V-cycle as a preconditioner, and a multigrid-preconditioned CG solver.

Given a symmetric positive (semi-)definite operator :math:`A` on one space of a :class:`Derham`,
for example the Poisson operator :math:`\sigma \mathbb M^0 + \mathbb G^\top \mathbb M^1 \mathbb G`:

1. A hierarchy of coarser Derhams is built (:class:`MultiGridHierarchy`).
2. :math:`A` is re-discretized on every level from its expression tree (:class:`OperatorCoarsener`).
3. One application of the preconditioner is a V-cycle with zero initial guess::

       x_l = 0
       smooth(A_l, b_l, x_l)                      # n_pre times
       x_l += P_l V_cycle(l+1, R_l (b_l - A_l x_l))
       smooth(A_l, b_l, x_l)                      # n_post times

   with an exact solve on the coarsest level. With the same linear, :math:`A`-symmetric smoother before and
   after the coarse-grid correction, the V-cycle is a symmetric positive definite preconditioner for CG.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as sla
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator, LinearOperator, Vector
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from scope_profiler import ProfileManager

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.io.options import OptionsBase
from struphy.linear_algebra.multigrid.coarsen import OperatorCoarsener
from struphy.linear_algebra.multigrid.hierarchy import MultiGridHierarchy
from struphy.linear_algebra.multigrid.smoothers import (
    ChebyshevSmoother,
    DiagonalComputer,
    JacobiSmoother,
    KrylovSmoother,
    Smoother,
    _owned_slice,
    _stencil_blocks,
    inverse_diagonal,
)
from struphy.linear_algebra.multigrid.transfer import SplineProlongation
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")

OptsSmoother = Literal["chebyshev", "jacobi", "cg"]
OptsSmootherPrecond = Literal["mass", "jacobi", "identity"]
OptsCoarseSolver = Literal["direct", "cg"]
OptsNullspace = Literal["constants"]


@dataclass
class MultiGridOptions(OptionsBase):
    r"""Options of :class:`MultiGridPreconditioner`.

    Parameters
    ----------
    smoother : str
        "chebyshev" (default), "jacobi" (damped) or "cg" (fixed number of PCG steps, non-linear).

    smoother_precond : str
        Preconditioner inside the Chebyshev and CG smoothers: "mass" (Kronecker-product approximation of
        the inverse mass matrix of the space, default), "jacobi" (inverse diagonal of the operator) or "identity".

    smoother_degree : int
        Polynomial degree of the Chebyshev smoother, number of sweeps of the Jacobi smoother,
        or number of iterations of the CG smoother.

    n_pre, n_post : int
        Number of smoother applications before and after the coarse-grid correction.

    jacobi_omega : float
        Damping factor of the Jacobi smoother.

    chebyshev_bounds : tuple[float, float]
        Smoothing interval of the Chebyshev smoother relative to the estimated largest eigenvalue.

    eig_iter : int
        Number of Lanczos steps for the eigenvalue estimate of the Chebyshev smoother.

    max_levels : int | None
        Maximal number of levels (None: coarsen as long as possible).

    min_cells : int
        Minimal number of elements per coarsened direction on the coarsest level.

    coarse_solver : str
        "direct" (LU factorization of the assembled coarsest operator, replicated on all processes)
        or "cg" (PCG with the smoother preconditioner, to tolerance ``coarse_tol``).

    coarse_tol : float
        Relative tolerance of the "cg" coarse solver.

    nullspace : str | None
        "constants" if the operator is singular with the constant functions in its kernel (e.g. the
        Poisson operator on 0-forms with periodic or Neumann boundary conditions).

    matrix_free_mass : bool
        Whether re-discretized mass matrices on coarse levels are matrix-free.
    """

    smoother: OptsSmoother = "chebyshev"
    smoother_precond: OptsSmootherPrecond = "mass"
    smoother_degree: int = 3
    n_pre: int = 1
    n_post: int = 1
    jacobi_omega: float = 2.0 / 3.0
    chebyshev_bounds: tuple[float, float] = (0.1, 1.1)
    eig_iter: int = 15
    max_levels: int | None = None
    min_cells: int = 2
    coarse_solver: OptsCoarseSolver = "direct"
    coarse_tol: float = 1e-10
    nullspace: OptsNullspace | None = None
    matrix_free_mass: bool = False

    def __post_init__(self):
        check_option(self.smoother, OptsSmoother)
        check_option(self.smoother_precond, OptsSmootherPrecond)
        check_option(self.coarse_solver, OptsCoarseSolver)
        if self.nullspace is not None:
            check_option(self.nullspace, OptsNullspace)
        assert self.smoother_degree >= 1
        assert self.n_pre >= 0 and self.n_post >= 0 and self.n_pre + self.n_post >= 1


class MultiGridPreconditioner(LinearOperator):
    r"""One geometric multigrid V-cycle as an approximate inverse of ``A`` (see module docstring).

    Parameters
    ----------
    A : LinearOperator
        Symmetric positive (semi-)definite operator on ``derham.coeff_spaces[form]``, built from struphy
        operators (Derham derivatives, boundary operators, :class:`WeightedMassOperator`, ...) with
        ``+``, ``-``, ``*`` and ``@``.

    derham : Derham
        The finest level.

    domain : Domain
        Mapping, used to re-discretize mass matrices on the coarse levels.

    options : MultiGridOptions | None
        Options (default: ``MultiGridOptions()``).

    mass_ops : WeightedMassOperators | None
        Mass operators of the finest level (only used by the "mass" smoother preconditioner).
    """

    def __init__(
        self,
        A: LinearOperator,
        derham: Derham,
        domain: Domain,
        options: MultiGridOptions | None = None,
        *,
        mass_ops: WeightedMassOperators | None = None,
    ):
        self._options = MultiGridOptions() if options is None else options
        opts = self._options
        self._derham = derham
        self._domain_map = domain

        self._form = _find_form(derham, A.domain)
        assert A.codomain is A.domain, "MultiGridPreconditioner requires a square operator."
        if opts.nullspace == "constants":
            assert self._form == "0", "nullspace='constants' is only implemented for 0-forms."

        self._hierarchy = MultiGridHierarchy(derham, max_levels=opts.max_levels, min_cells=opts.min_cells)
        L = self._hierarchy.n_levels
        logger.info(f"Multigrid levels: {[d.num_elements for d in self._hierarchy.derhams]}")

        self._P = [SplineProlongation(self._hierarchy[l + 1], self._hierarchy[l], self._form) for l in range(L - 1)]
        self._R = [P.T for P in self._P]
        self._coarseners = [
            OperatorCoarsener(self._hierarchy[l], self._hierarchy[l + 1], domain, matrix_free=opts.matrix_free_mass)
            for l in range(L - 1)
        ]
        self._diag = [DiagonalComputer(d.degree) for d in self._hierarchy.derhams]

        if opts.smoother_precond == "mass":
            fine_mass = WeightedMassOperators(derham, domain) if mass_ops is None else mass_ops
            mass = [fine_mass] + [c.mass_ops for c in self._coarseners]
            self._mass_pc = [MassMatrixPreconditioner(getattr(m, "M" + self._form)) for m in mass]

        # work vectors per level
        self._b = [d.coeff_spaces[self._form].zeros() for d in self._hierarchy.derhams]
        self._x = [d.coeff_spaces[self._form].zeros() for d in self._hierarchy.derhams]
        self._r = [d.coeff_spaces[self._form].zeros() for d in self._hierarchy.derhams]
        self._e = [d.coeff_spaces[self._form].zeros() for d in self._hierarchy.derhams]

        self._A: list[LinearOperator] = []
        self.update(A)

    # ------------------------------------------------------------------
    @property
    def domain(self):
        return self._A[0].domain

    @property
    def codomain(self):
        return self._A[0].codomain

    @property
    def dtype(self):
        return self._A[0].dtype

    @property
    def options(self) -> MultiGridOptions:
        return self._options

    @property
    def hierarchy(self) -> MultiGridHierarchy:
        return self._hierarchy

    @property
    def operators(self) -> list[LinearOperator]:
        """System operator on each level, ``operators[0]`` is the given one."""
        return self._A

    @property
    def smoothers(self) -> list[Smoother]:
        return self._smoothers

    def transpose(self, conjugate: bool = False) -> "MultiGridPreconditioner":
        assert all(s.is_symmetric for s in self._smoothers), "Only a symmetric V-cycle can be transposed."
        assert self._options.n_pre == self._options.n_post
        return self

    # ------------------------------------------------------------------
    def update(self, A: LinearOperator) -> None:
        """Set a new fine-level operator (e.g. with changed scalars) and update all levels.

        Mass matrices and derivative operators of coarse levels are re-used if ``A`` is built from the same objects.
        """
        assert A.domain is self._derham.coeff_spaces[self._form]
        self._A = [A]
        for c in self._coarseners:
            self._A.append(c(self._A[-1]))
        self._smoothers = [self._make_smoother(l) for l in range(self._hierarchy.n_levels - 1)]
        self._setup_coarse_solver()

    def _smoother_precond(self, l: int) -> LinearOperator:
        opts = self._options
        if opts.smoother_precond == "mass":
            return self._mass_pc[l]
        if opts.smoother_precond == "jacobi":
            return inverse_diagonal(self._diag[l](self._A[l]))
        return IdentityOperator(self._A[l].domain)

    def _make_smoother(self, l: int) -> Smoother:
        opts = self._options
        A = self._A[l]
        if opts.smoother == "chebyshev":
            return ChebyshevSmoother(
                A,
                self._smoother_precond(l),
                degree=opts.smoother_degree,
                bounds=opts.chebyshev_bounds,
                eig_iter=opts.eig_iter,
            )
        if opts.smoother == "jacobi":
            D_inv = inverse_diagonal(self._diag[l](A))
            return JacobiSmoother(A, D_inv, omega=opts.jacobi_omega, sweeps=opts.smoother_degree)
        return KrylovSmoother(A, self._smoother_precond(l), iterations=opts.smoother_degree)

    def _setup_coarse_solver(self) -> None:
        opts = self._options
        A = self._A[-1]
        if opts.coarse_solver == "cg":
            pc = self._smoother_precond(len(self._A) - 1) if opts.smoother_precond != "identity" else None
            self._coarse_cg = inverse(A, "pcg", pc=pc, tol=1e-300, maxiter=1000, recycle=False)
            return

        Ad = _assemble_dense(A)
        if opts.nullspace == "constants":
            # A + s 1 1^T is regular; for b orthogonal to 1 its solution is the zero-mean solution of A x = b
            n = Ad.shape[0]
            Ad = Ad + np.mean(np.abs(np.diag(Ad))) / n * np.ones((n, n))
        else:
            # rows/cols of Dirichlet dofs are zero: put ones on the diagonal
            zero = np.flatnonzero(np.all(Ad == 0.0, axis=1))
            Ad[zero, zero] = 1.0
        self._coarse_lu = sla.lu_factor(Ad)

    # ------------------------------------------------------------------
    def dot(self, b: Vector, out: Vector | None = None) -> Vector:
        """Apply one V-cycle (zero initial guess) to ``b``."""
        assert b.space is self.domain
        if out is None:
            out = self.domain.zeros()
        b.copy(out=self._b[0])
        if self._options.nullspace == "constants":
            _remove_mean(self._b[0])
        with ProfileManager.profile_region("multigrid V-cycle", functions=[self._vcycle]):
            self._vcycle(0)
        self._x[0].copy(out=out)
        if self._options.nullspace == "constants":
            _remove_mean(out)
        return out

    def _vcycle(self, l: int) -> None:
        b, x = self._b[l], self._x[l]
        if l == len(self._A) - 1:
            self._coarse_solve(b, x)
            return

        x *= 0.0
        S = self._smoothers[l]
        with ProfileManager.profile_region(f"pre-smoother level {l}", functions=[S.smooth]):
            for _ in range(self._options.n_pre):
                S.smooth(b, x)

        r = S.residual(b, x, self._r[l])
        self._R[l].dot(r, out=self._b[l + 1])
        with ProfileManager.profile_region(f"coarse-grid correction level {l}", functions=[self._vcycle]):
            self._vcycle(l + 1)
        self._P[l].dot(self._x[l + 1], out=self._e[l])
        x += self._e[l]

        with ProfileManager.profile_region(f"post-smoother level {l}", functions=[S.smooth]):
            for _ in range(self._options.n_post):
                S.smooth(b, x)

    def _coarse_solve(self, b: Vector, x: Vector) -> None:
        if self._options.coarse_solver == "cg":
            nb = np.sqrt(b.inner(b))
            if nb == 0.0:
                x *= 0.0
                return
            self._coarse_cg._options["tol"] = self._options.coarse_tol * nb
            self._coarse_cg.dot(b, out=x)
            return

        bg = _gather(b)
        if self._options.nullspace == "constants":
            bg -= bg.mean()
        _scatter(sla.lu_solve(self._coarse_lu, bg), x)


class MultiGridSolver(LinearOperator):
    r"""Conjugate gradient method preconditioned with :class:`MultiGridPreconditioner`.

    Parameters
    ----------
    A, derham, domain, options, mass_ops :
        See :class:`MultiGridPreconditioner`.

    tol : float
        Relative tolerance, the iteration stops when :math:`\|b - A x\|_2 \leq \mathrm{tol}\, \|b\|_2`.

    maxiter : int
        Maximal number of CG iterations.

    verbose : bool
        Print the residual in every iteration.
    """

    def __init__(
        self,
        A: LinearOperator,
        derham: Derham,
        domain: Domain,
        options: MultiGridOptions | None = None,
        *,
        mass_ops: WeightedMassOperators | None = None,
        tol: float = 1e-8,
        maxiter: int = 100,
        verbose: bool = False,
    ):
        self._pc = MultiGridPreconditioner(A, derham, domain, options, mass_ops=mass_ops)
        self._tol = tol
        self._solver = inverse(A, "pcg", pc=self._pc, tol=tol, maxiter=maxiter, verbose=verbose, recycle=False)

    @property
    def domain(self):
        return self._pc.domain

    @property
    def codomain(self):
        return self._pc.codomain

    @property
    def dtype(self):
        return self._pc.dtype

    @property
    def preconditioner(self) -> MultiGridPreconditioner:
        return self._pc

    @property
    def info(self) -> dict:
        """Information of the last solve: ``niter``, ``success``, ``res_norm``."""
        return self._solver._info

    def update(self, A: LinearOperator) -> None:
        """Set a new operator (see :meth:`MultiGridPreconditioner.update`)."""
        self._pc.update(A)
        self._solver.linop = A

    def transpose(self, conjugate: bool = False) -> "MultiGridSolver":
        return self

    def dot(self, b: Vector, out: Vector | None = None, x0: Vector | None = None) -> Vector:
        """Solve ``A x = b`` (initial guess ``x0``, zero by default)."""
        nb = np.sqrt(b.inner(b))
        self._solver._options["tol"] = self._tol * nb if nb > 0.0 else self._tol
        self._solver._options["x0"] = x0 if x0 is not None else self.domain.zeros()
        return self._solver.dot(b, out=out)


# ----------------------------------------------------------------------------------------------------
def _find_form(derham: Derham, V) -> str:
    for form in ("0", "1", "2", "3", "v"):
        if derham.coeff_spaces[form] is V:
            return form
    raise ValueError("The operator does not act on a coefficient space of the given Derham.")


def _comm(v: Vector):
    blk = _stencil_blocks(v)[0]
    return blk.space.cart.comm if blk.space.parallel else None


def _gather(v: Vector) -> np.ndarray:
    """Global coefficient array of ``v`` (same on all processes)."""
    a = v.toarray()
    comm = _comm(v)
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, a, op=MPI.SUM)
    return a


def _scatter(a: np.ndarray, v: Vector) -> None:
    """Write the owned part of the global array ``a`` into ``v``."""
    offset = 0
    for blk in _stencil_blocks(v):
        V = blk.space
        n = int(np.prod(V.npts))
        glob = a[offset : offset + n].reshape(tuple(int(m) for m in V.npts))
        blk._data[_owned_slice(blk)] = glob[tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))]
        blk.ghost_regions_in_sync = False
        offset += n


def _assemble_dense(A: LinearOperator) -> np.ndarray:
    """Global dense matrix of ``A`` (same on all processes), by applying it to all unit vectors."""
    e = A.domain.zeros()
    blocks = _stencil_blocks(e)
    sizes = [int(np.prod(b.space.npts)) for b in blocks]
    N = sum(sizes)
    out = np.zeros((N, N))
    y = A.codomain.zeros()
    col = 0
    for blk, n in zip(blocks, sizes):
        V = blk.space
        for flat in range(n):
            gidx = np.unravel_index(flat, tuple(int(m) for m in V.npts))
            for b in blocks:
                b._data[...] = 0.0
                b.ghost_regions_in_sync = False
            if all(s <= i <= e_ for i, s, e_ in zip(gidx, V.starts, V.ends)):
                loc = tuple(int(i - s + p * m) for i, s, p, m in zip(gidx, V.starts, V.pads, V.shifts))
                blk._data[loc] = 1.0
            A.dot(e, out=y)
            out[:, col] = _gather(y)
            col += 1
    return out


def _remove_mean(v: Vector) -> None:
    """Subtract the mean of all coefficients (the projection orthogonal to the constant vector)."""
    comm = _comm(v)
    s = sum(float(np.sum(b._data[_owned_slice(b)])) for b in _stencil_blocks(v))
    N = sum(int(np.prod(b.space.npts)) for b in _stencil_blocks(v))
    if comm is not None and comm.size > 1:
        s = comm.allreduce(s, op=MPI.SUM)
    mean = s / N
    for b in _stencil_blocks(v):
        b._data[_owned_slice(b)] -= mean
        b.ghost_regions_in_sync = False
