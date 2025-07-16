from typing import Callable

from struphy.fields_background.base import FluidEquilibrium
from struphy.io.options import FieldsBackground
from struphy.kinetic_background.base import KineticBackground


class InitialCondition:
    """Callable initial condition as sum of background + perturbation."""
    def __init__(self, 
                 background: list = None,
                 perturbation: list = None,
                 equil: FluidEquilibrium = None,):
        
        for b in background:
            assert isinstance(b, (FieldsBackground, KineticBackground))
        self._background = background
        
        for p in perturbation:
            assert isinstance(p, tuple)
            assert len(p) == 2
            assert isinstance(p[0], Callable)
            assert isinstance(p[1], tuple)
        self._perturbation = perturbation
        
        self._equil = equil
        
    def __call__(eta1, eta2, eta3, *v):
        return 1
        