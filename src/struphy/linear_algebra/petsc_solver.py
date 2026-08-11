import logging

import cunumpy as xp
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.linalg.basic import (
    ComposedLinearOperator,
    IdentityOperator,
    InverseLinearOperator,
    LinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    Vector,
)
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.stencil import StencilMatrix
from feectools.linalg.topetsc import get_npts_local, mat_topetsc, vec_topetsc
from feectools.linalg.utilities import petsc_to_psydac

logger = logging.getLogger("struphy")


def _directional_derivative_to_stencil_matrix(op):
    """Build a :class:`~feectools.linalg.stencil.StencilMatrix` equivalent to a
    (matrix-free) :class:`~feectools.feec.derivatives.DirectionalDerivativeOperator`, so it can
    be handed to :func:`feectools.linalg.topetsc.mat_topetsc`.

    ``DirectionalDerivativeOperator.dot`` computes, along its differentiation axis
    ``diffdir`` (identity along every other axis):

    - ``out = v[..., k+1, ...] - v[..., k, ...]``   if not negative, not transposed
    - ``out = v[..., k, ...] - v[..., k+1, ...]``   if negative, not transposed
    - ``out = v[..., k-1, ...] - v[..., k, ...]``   if not negative, transposed
    - ``out = v[..., k, ...] - v[..., k-1, ...]``   if negative, transposed

    i.e. a plain two-point (identity, shift-by-one) stencil.

    Note
    ----
    Only verified for a *periodic* differentiation axis. For a non-periodic axis under a
    parallel (MPI-comm-attached) space -- which is how struphy always builds its Derham
    complex, even with a single rank -- this construction (and feectools' own
    ``DirectionalDerivativeOperator.tokronstencil().tostencil()``) was found to disagree with
    the operator's actual ``.dot()`` at the two boundary planes along that axis. The root
    cause was not identified; rather than risk silently wrong results, this case raises
    ``NotImplementedError``.
    """
    assert isinstance(op, DirectionalDerivativeOperator)

    V = op.domain
    W = op.codomain
    ndim = V.ndim
    diffdir = op._diffdir
    negative = op._negative
    transposed = op._transposed

    if not V.periods[diffdir]:
        raise NotImplementedError(
            "PETScSolver cannot (yet) assemble a DirectionalDerivativeOperator along a "
            f"non-periodic axis (diffdir={diffdir}, periods={V.periods}) of a parallel "
            "(MPI-comm-attached) space: this was found to disagree with the operator's actual "
            "action at the domain boundary, for a reason not yet root-caused. Only fully "
            "periodic operators (e.g. derham.grad on a fully periodic domain) are supported."
        )

    M = StencilMatrix(V, W)

    def off(o):
        return slice(o, o + 1)

    rows = tuple(slice(None) for _ in range(ndim))
    identity_key = tuple(off(0) for _ in range(ndim))

    shift = -1 if transposed else 1
    shifted_key = tuple(off(shift) if d == diffdir else off(0) for d in range(ndim))

    if negative:
        M[rows + identity_key] = 1.0
        M[rows + shifted_key] = -1.0
    else:
        M[rows + identity_key] = -1.0
        M[rows + shifted_key] = 1.0

    M.remove_spurious_entries()
    return M


def _materialize_block(block):
    """Turn a ``BlockLinearOperator`` block entry into a concrete matrix (``StencilMatrix`` or
    ``None``) that :func:`feectools.linalg.topetsc.mat_topetsc` can handle directly.

    Blocks of a topological operator such as ``derham.curl`` are not always plain
    ``StencilMatrix``: sign conventions are sometimes expressed via ``ScaledLinearOperator``
    wrapping a ``StencilMatrix``/``DirectionalDerivativeOperator`` rather than baking the sign
    into the matrix data (observed e.g. for ``derham.curl.T``, whose transposed blocks land on
    this path). ``mat_topetsc`` calls ``.update_ghost_regions()`` on every block, which only
    concrete matrix types implement -- so any such wrapper must be resolved to a concrete matrix
    first. Recurses through nested ``ScaledLinearOperator``s.
    """
    if block is None:
        return None
    if isinstance(block, DirectionalDerivativeOperator):
        return _directional_derivative_to_stencil_matrix(block)
    if isinstance(block, ScaledLinearOperator):
        inner = _materialize_block(block.operator)
        if inner is None:
            return None
        scaled = inner.copy()
        scaled *= block.scalar
        return scaled
    return block


def _assemble_leaf_operator(A):
    """Return an operator equivalent to `A` that is directly convertible via
    :func:`feectools.linalg.topetsc.mat_topetsc` (i.e. a ``StencilMatrix`` or a
    ``BlockLinearOperator`` whose blocks are all ``StencilMatrix``), replacing any
    ``DirectionalDerivativeOperator``/``ScaledLinearOperator`` (block or bare) by its assembled
    equivalent -- see :func:`_materialize_block`.
    """
    if isinstance(A, (DirectionalDerivativeOperator, ScaledLinearOperator)):
        return _materialize_block(A)

    if isinstance(A, BlockLinearOperator):
        out = BlockLinearOperator(A.domain, A.codomain)
        for i, j in A.nonzero_block_indices:
            out[i, j] = _materialize_block(A[i, j])
        return out

    return A


def _comm_of(space):
    """MPI communicator of a StencilVectorSpace/BlockVectorSpace, matching mat_topetsc's convention."""
    if isinstance(space, BlockVectorSpace):
        return space.spaces[0].cart.global_comm
    return space.cart.global_comm


def _identity_petsc_mat(space):
    """Build a PETSc.Mat representing the identity operator on `space`."""
    from petsc4py import PETSc

    comm = _comm_of(space)
    localsize = int(xp.sum(xp.prod(get_npts_local(space), axis=1)))
    globalsize = space.dimension

    gmat = PETSc.Mat().create(comm=comm)
    gmat.setSizes(size=((localsize, globalsize), (localsize, globalsize)))
    gmat.setType("mpiaij" if comm else "seqaij")
    gmat.setUp()

    ones = space.zeros()
    ones._data[:] = 1.0
    gmat.setDiagonal(vec_topetsc(ones))
    gmat.assemble()

    return gmat


def _assemble_petsc_matrix(A):
    """Recursively assemble a ``PETSc.Mat`` for a (possibly composite) feectools
    ``LinearOperator``, by converting every assembled leaf via
    :func:`feectools.linalg.topetsc.mat_topetsc` and combining the pieces with PETSc's own
    matrix algebra (``matMult`` for composition, ``axpy`` for sums, ``scale`` for scalar
    multiples). This lets algebraic preconditioners (jacobi, gamg, ...) work on operators such
    as ``grad.T @ M @ grad`` that are not themselves a ``StencilMatrix``/``BlockLinearOperator``.

    Parameters
    ----------
    A : feectools.linalg.basic.LinearOperator
        Operator to assemble. Supported: ``StencilMatrix``, ``BlockLinearOperator``, any operator
        exposing an assembled ``.matrix`` (e.g. ``WeightedMassOperator``), ``IdentityOperator``,
        ``ScaledLinearOperator``, ``SumLinearOperator`` and ``ComposedLinearOperator`` built out of
        the above (as produced e.g. by ``derham.grad.T @ mass_ops.M1 @ derham.grad``).

    Returns
    -------
    gmat : PETSc.Mat
    """
    if isinstance(A, (StencilMatrix, BlockLinearOperator, DirectionalDerivativeOperator)):
        return mat_topetsc(_assemble_leaf_operator(A))

    matrix = getattr(A, "matrix", None)
    if isinstance(matrix, (StencilMatrix, BlockLinearOperator, DirectionalDerivativeOperator)):
        return mat_topetsc(_assemble_leaf_operator(matrix))

    if isinstance(A, IdentityOperator):
        return _identity_petsc_mat(A.domain)

    if isinstance(A, ScaledLinearOperator):
        gmat = _assemble_petsc_matrix(A.operator)
        gmat.scale(A.scalar)
        return gmat

    if isinstance(A, ComposedLinearOperator):
        from petsc4py import PETSc

        mats = [_assemble_petsc_matrix(m) for m in A.multiplicants]
        gmat = mats[0]
        for m in mats[1:]:
            gmat = gmat.matMult(m)
        return gmat

    if isinstance(A, SumLinearOperator):
        from petsc4py import PETSc

        mats = [_assemble_petsc_matrix(a) for a in A.addends]
        gmat = mats[0].copy()
        for m in mats[1:]:
            gmat.axpy(1.0, m, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        return gmat

    raise NotImplementedError(
        f"PETScSolver cannot assemble a PETSc matrix for operator of type {type(A)}. "
        "Supported: StencilMatrix, BlockLinearOperator, operators exposing an assembled "
        "'.matrix', IdentityOperator, and Scaled/Sum/Composed combinations thereof."
    )


class PETScSolver(InverseLinearOperator):
    """(Approximate) inverse of a feectools ``LinearOperator``, computed via a PETSc ``KSP``
    Krylov solver.

    ``A`` is assembled into a ``PETSc.Mat`` (see :func:`_assemble_petsc_matrix` -- this also
    handles composite operators such as ``grad.T @ M @ grad``, not just plain
    ``StencilMatrix``/``BlockLinearOperator``) and the right-hand side is converted to a
    ``PETSc.Vec`` via :func:`feectools.linalg.topetsc.vec_topetsc`; the solve itself is delegated
    to ``petsc4py.PETSc.KSP``. Requires the optional ``petsc4py`` dependency
    (``pip install struphy[petsc]``).

    Parameters
    ----------
    A : feectools.linalg.basic.LinearOperator
        Left-hand-side matrix of the linear system, see :func:`_assemble_petsc_matrix` for the
        supported operator types.

    x0 : feectools.linalg.basic.Vector, default=None
        Kept for interface compatibility with the other
        :class:`~feectools.linalg.basic.InverseLinearOperator` subclasses; unused by PETSc's KSP.

    tol : float, default=1e-6
        Relative tolerance, passed to ``KSP.setTolerances(rtol=tol)``. Note this differs from
        feectools' own solvers, whose ``tol`` is an *absolute* tolerance on the residual norm --
        for a poorly-scaled system (e.g. a right-hand side far from order 1) the two are not
        directly comparable; see git history for a reverted attempt to unify them via
        ``atol``, which caused severe slowdowns/inaccuracy for such systems.

    maxiter : int, default=1000
        Maximum number of KSP iterations.

    verbose : bool, default=False
        If True, log convergence information after each solve.

    recycle : bool, default=False
        Kept for interface compatibility; unused by PETSc's KSP.

    ksp_type : str, default="cg"
        PETSc Krylov solver type, see ``petsc4py.PETSc.KSP.Type``.

    pc_type : str, default="none"
        PETSc preconditioner type, see ``petsc4py.PETSc.PC.Type``. E.g. ``"gamg"`` (algebraic
        multigrid) for large, ill-conditioned elliptic systems.
    """

    def __init__(
        self,
        A,
        *,
        x0=None,
        tol=1e-6,
        maxiter=1000,
        verbose=False,
        recycle=False,
        ksp_type="cg",
        pc_type="none",
    ):
        assert isinstance(A, LinearOperator), f"PETScSolver requires a LinearOperator, got {type(A)}."

        self._options = {
            "x0": x0,
            "tol": tol,
            "maxiter": maxiter,
            "verbose": verbose,
            "recycle": recycle,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        }

        super().__init__(A, **self._options)

        self._info = None
        self._ksp = None
        # operator for which self._ksp's PETSc.Mat was last built, used to avoid
        # re-assembling the matrix on every solve() call when `linop` is unchanged
        self._ksp_linop = None

    def _get_ksp(self):
        from petsc4py import PETSc

        A = self._A
        if self._ksp is None or self._ksp_linop is not A:
            gmat = _assemble_petsc_matrix(A)

            if self._ksp is None:
                self._ksp = PETSc.KSP().create(comm=gmat.getComm())

            self._ksp.setType(self._options["ksp_type"])
            self._ksp.getPC().setType(self._options["pc_type"])
            self._ksp.setTolerances(rtol=self._options["tol"], max_it=self._options["maxiter"])
            self._ksp.setOperators(gmat)
            self._ksp.setFromOptions()

            self._ksp_linop = A

        return self._ksp

    def solve(self, b, out=None):
        """Solve ``A x = b`` using a PETSc KSP Krylov solver.

        Parameters
        ----------
        b : feectools.linalg.basic.Vector
            Right-hand-side vector of the linear system.

        out : feectools.linalg.basic.Vector | None
            The output vector, or None (optional).

        Returns
        -------
        x : feectools.linalg.basic.Vector
            Numerical solution of the linear system. Convergence info is available
            via :meth:`get_info`.
        """
        assert isinstance(b, Vector)
        assert b.space is self._domain

        ksp = self._get_ksp()

        gvec_b = vec_topetsc(b)
        gvec_x = gvec_b.duplicate()

        ksp.solve(gvec_b, gvec_x)

        out = petsc_to_psydac(gvec_x, self._codomain, out=out)

        self._info = {
            "niter": ksp.getIterationNumber(),
            "success": ksp.getConvergedReason() > 0,
            "res_norm": ksp.getResidualNorm(),
        }

        if self._options["verbose"]:
            logger.info(f"PETSc KSP solver info: {self._info}")

        gvec_b.destroy()
        gvec_x.destroy()

        return out

    def dot(self, b, out=None):
        return self.solve(b, out=out)
