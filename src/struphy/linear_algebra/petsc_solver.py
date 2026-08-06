import logging

from feectools.linalg.basic import InverseLinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator
from feectools.linalg.stencil import StencilMatrix
from feectools.linalg.topetsc import mat_topetsc, vec_topetsc
from feectools.linalg.utilities import petsc_to_psydac

logger = logging.getLogger("struphy")


class PETScSolver(InverseLinearOperator):
    """(Approximate) inverse of an assembled operator, computed via a PETSc ``KSP`` Krylov solver.

    ``A`` is converted to a ``PETSc.Mat`` via :func:`feectools.linalg.topetsc.mat_topetsc`
    and the right-hand side is converted to a ``PETSc.Vec`` via
    :func:`feectools.linalg.topetsc.vec_topetsc`; the solve itself is delegated to
    ``petsc4py.PETSc.KSP``. Requires the optional ``petsc4py`` dependency
    (``pip install struphy[petsc]``).

    Parameters
    ----------
    A : feectools.linalg.stencil.StencilMatrix | feectools.linalg.block.BlockLinearOperator
        Left-hand-side matrix of the linear system. Only assembled operators can be
        converted to a ``PETSc.Mat``, see :func:`feectools.linalg.topetsc.mat_topetsc`.

    x0 : feectools.linalg.basic.Vector, default=None
        Kept for interface compatibility with the other
        :class:`~feectools.linalg.basic.InverseLinearOperator` subclasses; unused by PETSc's KSP.

    tol : float, default=1e-6
        Relative tolerance, passed to ``KSP.setTolerances(rtol=tol)``.

    maxiter : int, default=1000
        Maximum number of KSP iterations.

    verbose : bool, default=False
        If True, log convergence information after each solve.

    recycle : bool, default=False
        Kept for interface compatibility; unused by PETSc's KSP.

    ksp_type : str, default="cg"
        PETSc Krylov solver type, see ``petsc4py.PETSc.KSP.Type``.

    pc_type : str, default="none"
        PETSc preconditioner type, see ``petsc4py.PETSc.PC.Type``.
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
        assert isinstance(A, (StencilMatrix, BlockLinearOperator)), (
            f"PETScSolver only supports assembled operators (StencilMatrix or BlockLinearOperator), got {type(A)}."
        )

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
            gmat = mat_topetsc(A)

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
