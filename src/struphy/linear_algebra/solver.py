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
    """Parameters for discrete gradient solvers.

    Parameters
    ----------
    relaxation_factor : float
        Relaxation of the fixed-point iteration (1.0: no relaxation).
    tol : float
        Guard on the iterates: the largest change of a marker position between two iterations.
        Independent of the number of markers.
    tol_energy : float
        Tolerance on the absolute energy difference over the step, which decides when the iteration stops, together
        with the guard ``tol``. It is limited by the accuracy of the inner linear solves.
    maxiter : int
        Maximum number of iterations.
    info : bool
        Whether to log the convergence of each iteration.
    """

    relaxation_factor: float = 0.5
    tol: float = 1e-7
    tol_energy: float = 1e-13
    maxiter: int = 100
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
