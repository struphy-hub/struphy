import logging
from dataclasses import dataclass

from struphy.io.options import LiteralOptions

logger = logging.getLogger("struphy")


@dataclass
class SolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 3000
    info: bool = False
    recycle: bool = True

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
