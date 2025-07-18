from abc import ABCMeta, abstractmethod
from mpi4py import MPI

from struphy.initial.base import InitialCondition


class Variable(metaclass=ABCMeta):
    """Single variable (unknown) of a Species."""
    
    @property
    def backgrounds(self):
        return self._backgrounds
    
    @property
    def perturbations(self):
        return self._perturbations

    def add_background(self, background = None, verbose=False,):
        # assert isinstance(...)
        if not hasattr(self, "_backgrounds"):
            self._backgrounds = []
        self._backgrounds += [background]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Variable '{self.__class__.__name__}': added background '{background.__class__.__name__}' with:")
            for k, v in background.__dict__.items():
                print(f'  {k}: {v}')

    def add_perturbation(
        self,
        perturbation = None,
        given_in_basis: tuple = None,
        verbose=False,
    ):
        # assert isinstance(...)
        if not hasattr(self, "_perturbations"):
            self._perturbations = []
        self._perturbations += [(perturbation, given_in_basis)]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Variable '{self.__class__.__name__}': added perturbation '{perturbation.__class__.__name__}',{given_in_basis = }, with:")
            for k, v in perturbation.__dict__.items():
                print(f'  {k}: {v}')

    def define_initial_condition(self):
        self._initial_condition = InitialCondition(
            background=self.backgrounds,
            perturbation=self.perturbations,
        )

    
class FEECVariable(Variable):
    def __init__(self, space: str = "H1"):
        assert space in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")
        self._space = space
        
    @property
    def space(self):
        return self._space
    
    
class PICVariable(Variable):
    pass
    
    
class SPHVariable(Variable):
    pass