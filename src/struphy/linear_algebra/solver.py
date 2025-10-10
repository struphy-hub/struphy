from dataclasses import dataclass


@dataclass
class SolverParameters:
    """Parameters for psydac solvers."""

    tol: float = 1e-8
    maxiter: int = 3000
    info: bool = False
    verbose: bool = False
    recycle: bool = True

@dataclass
class DiscreteGradientSolverParameters:
    """Parameters for discrete gradient solvers."""

    relaxation_factor : float = 0.5
    tol: float = 1e-12
    maxiter: int = 20
    verbose: bool = False
    info: bool = False