from dataclasses import dataclass

from struphy.io.options import OptsNonlinearSolver


@dataclass
class SolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 3000
    info: bool = False
    verbose: bool = False
    recycle: bool = True


@dataclass
class NonlinearSolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 100
    info: bool = False
    verbose: bool = False
    type: OptsNonlinearSolver = "Picard"
    linearize: bool = False
