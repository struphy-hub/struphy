from typing import Callable

from struphy.fields_background.base import FluidEquilibrium
from struphy.kinetic_background.base import KineticBackground


class Variables:
    """Class for all Variable objects of a model."""
    def __init__(self, **species_dct):
        
        assert "em_fields" in species_dct
        assert "fluid" in species_dct
        assert "kinetic" in species_dct
        
        for name, space in species_dct["em_fields"].items():
            setattr(self, name, Variable(name, space))
            
        for name, space in species_dct["fluid"].items():
            setattr(self, name, Variable(name, space))
            
        for name, space in species_dct["kinetic"].items():
            setattr(self, name, Variable(name, space))
            
    @property
    def all(self):
        return self.__dict__
    
    def add_background(self, variable: str, background,):
        assert variable in self.all, f'Variable {variable} is not part of model variables {self.all = }' 
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_background(background)
        
    def add_perturbation(self, variable: str, perturbation: Callable, given_in_basis: tuple = None,):
        assert variable in self.all, f'Variable {variable} is not part of model variables {self.all = }' 
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_perturbation(perturbation=perturbation, 
                             given_in_basis=given_in_basis,)


class Variable:
    """Single variable (unknown) of a StruphyModel.
    """
    def __init__(self, name: str, space: str, save_data: bool = True):
        
        self._name = name
        self._space = space
        self._save_data = save_data
        
        self._background = []
        self._perturbation = []
        
    ## attributes
        
    @property
    def name(self):
        return self._name
    
    @property
    def space(self):
        return self._space
    
    @property
    def save_data(self):
        return self._save_data
    
    @property
    def background(self):
        return self._background
    
    # @background.setter
    # def background(self, new):
    #     assert isinstance(new, FluidEquilibrium | KineticBackground)
    #     self._background = new
    
    @property
    def perturbation(self):
        return self._perturbation
    
    # @perturbation.setter
    # def perturbation(self, new):
    #     assert isinstance(new, Perturbation)
    #     self._perturbation = new
    
    @property
    def initial_condition(self): 
        if not hasattr(self, "_initial_condition"):
            self.set_initial_condition()
        return self._initial_condition
    
    ## methods
    
    def add_background(self, background):
        # assert isinstance(...)
        self._background += [background]
        
    def add_perturbation(self, 
                         perturbation: Callable = None, 
                         given_in_basis: tuple = None,):
        # assert isinstance(...)
        self._perturbation += [(perturbation, given_in_basis)]
    
    def set_initial_condition(self):
        self._initial_condition = InitialCondition(self.background, 
                                                    self.perturbation,)
    
    def eval_initial_condition(self, eta1, eta2, eta3, *v):
        """Callable initial condition as sum of background + perturbation."""
        return self.initial_condition(eta1, eta2, eta3, *v)
    
    
class InitialCondition:
    """Callable initial condition as sum of background + perturbation."""
    def __init__(self, 
                 background: FluidEquilibrium | KineticBackground = None,
                 perturbation : Callable = None,):
        
        self._background = background
        self._perturbation = perturbation

