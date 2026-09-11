import logging
from dataclasses import dataclass

from struphy.io.options import LiteralOptions

logger = logging.getLogger("struphy")

# kwargs accepted by struphy.linear_algebra.petsc_solver.PETScSolver.__init__
_PETSC_SOLVER_KWARGS = ("x0", "tol", "maxiter", "verbose", "recycle", "ksp_type", "pc_type", "near_null_space")


def inverse(A, solver: str, **kwargs):
    """Create an (approximate) inverse of ``A``.

    Thin wrapper around :func:`feectools.linalg.solvers.inverse` that additionally
    supports ``solver="petsc"``, dispatching to
    :class:`~struphy.linear_algebra.petsc_solver.PETScSolver`. For all other solver
    names this simply delegates to the feectools implementation.

    Parameters
    ----------
    A : feectools.linalg.basic.LinearOperator
        Left-hand-side matrix of the linear system. For ``solver="petsc"``, see
        :func:`struphy.linear_algebra.petsc_solver._assemble_petsc_matrix` for the
        supported operator types -- this includes plain assembled matrices as well as
        composite operators such as ``grad.T @ M @ grad``.

    solver : str
        Preferred iterative solver, one of feectools' options ('cg', 'pcg',
        'bicg', 'bicgstab', 'pbicgstab', 'minres', 'lsmr', 'gmres') or 'petsc'.

    Returns
    -------
    obj : feectools.linalg.basic.InverseLinearOperator
        A linear operator acting as the (approximate) inverse of A.
    """
    if solver == "petsc":
        from struphy.linear_algebra.petsc_solver import PETScSolver

        if kwargs.get("pc") is not None:
            logger.debug("PETScSolver ignores the feectools 'pc' preconditioner; use 'pc_type' instead.")

        petsc_kwargs = {k: v for k, v in kwargs.items() if k in _PETSC_SOLVER_KWARGS}
        petsc_kwargs.setdefault("ksp_type", "cg")
        petsc_kwargs.setdefault("pc_type", "jacobi")

        return PETScSolver(A, **petsc_kwargs)

    # pc_type/ksp_type/near_null_space are petsc-only (see _PETSC_SOLVER_KWARGS above);
    # feectools' InverseLinearOperator subclasses forward unknown kwargs straight to their
    # constructor and would raise on them, so they never reach this branch.
    kwargs.pop("pc_type", None)
    kwargs.pop("ksp_type", None)
    kwargs.pop("near_null_space", None)

    from feectools.linalg.solvers import inverse as feectools_inverse

    return feectools_inverse(A, solver, **kwargs)


@dataclass
class SolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 3000
    info: bool = False
    recycle: bool = True
    pc_type: LiteralOptions.OptsPETScPrecond = "jacobi"
    """Preconditioner for ``solver="petsc"`` only (ignored otherwise): PETSc's ``PCType``
    name, e.g. ``"jacobi"`` (cheap, diagonal) or ``"gamg"`` (algebraic multigrid -- far
    stronger for large, ill-conditioned systems, but with more setup overhead per matrix
    assembly)."""

    def __post_init__(self):
        self.verbose = False
        if logger.level <= logging.DEBUG:
            self.verbose = True


@dataclass
class DiscreteGradientSolverParameters:
    """Parameters for discrete gradient solvers."""

    relaxation_factor: float = 0.5
    tol: float = 1e-12
    maxiter: int = 20
    info: bool = False

    def __post_init__(self):
        self.verbose = False
        if logger.level <= logging.DEBUG:
            self.verbose = True


@dataclass
class NonlinearSolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 100
    info: bool = False
    type: LiteralOptions.OptsNonlinearSolver = "Picard"
    linearize: bool = False

    def __post_init__(self):
        self.verbose = False
        if logger.level <= logging.DEBUG:
            self.verbose = True
