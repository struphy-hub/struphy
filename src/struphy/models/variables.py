from abc import ABCMeta, abstractmethod
from mpi4py import MPI

from struphy.initial.base import InitialCondition
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.io.options import FieldsBackground
from struphy.initial.perturbations import Perturbation
from struphy.geometry.base import Domain
from struphy.fields_background.base import FluidEquilibrium


class Variable(metaclass=ABCMeta):
    """Single variable (unknown) of a Species."""
    
    @abstractmethod
    def allocate(self):
        """Alocate object and memory for variable."""
    
    @property
    def backgrounds(self):
        if not hasattr(self, "_backgrounds"):
            self._backgrounds = None
        return self._backgrounds
    
    @property
    def perturbations(self):
        if not hasattr(self, "_perturbations"):
            self._perturbations = None
        return self._perturbations
    
    @property
    def save_data(self):
        """Store variable data during simulation (default=True)."""
        if not hasattr(self, "_save_data"):
            self._save_data = True
        return self._save_data
    
    @save_data.setter
    def save_data(self, new):
        assert isinstance(new, bool)
        self._save_data = new
    
    @property
    def species(self):
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    def add_background(self, background, verbose=True):
        """Type inference of added background done in sub class."""
        if not hasattr(self, "_backgrounds") or self.backgrounds is None:
            self._backgrounds = background
        else:
            if not isinstance(self.backgrounds, list):
                self._backgrounds = [self.backgrounds]
            self._backgrounds += [background]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nVariable '{self.__name__}' of species '{self.species}' - added background '{background.__class__.__name__}' with:")
            for k, v in background.__dict__.items():
                print(f'  {k}: {v}')

    def add_perturbation(self, perturbation: Perturbation, verbose=True):
        if not hasattr(self, "_perturbations") or self.perturbations is None:
            self._perturbations = perturbation
        else:
            if not isinstance(self.perturbations, list):
                self._perturbations = [self.perturbations]
            self._perturbations += [perturbation]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nVariable '{self.__name__}' of species '{self.species}' - added perturbation '{perturbation.__class__.__name__}' with:")
            for k, v in perturbation.__dict__.items():
                print(f'  {k}: {v}')

    def define_initial_condition(self):
        self._initial_condition = InitialCondition(
            background=self.backgrounds,
            perturbation=self.perturbations,
        )

    
class FEECVariable(Variable):
    def __init__(self, name: str = "a_feec_var", space: str = "H1"):
        assert space in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")
        self._name = name
        self._space = space
        
    @property
    def __name__(self):
        return self._name
        
    @property
    def space(self):
        return self._space
    
    @property
    def spline(self) -> SplineFunction:
        return self._spline
    
    def add_background(self, background: FieldsBackground, verbose=True):
        super().add_background(background, verbose=verbose)
    
    def allocate(self, derham: Derham, domain: Domain = None, equil: FluidEquilibrium = None,):
        self._spline = derham.create_spline_function(
                        name=self.__name__,
                        space_id=self.space,
                        backgrounds=self.backgrounds,
                        perturbations=self.perturbations,
                        domain=domain,
                        equil=equil,
                    )
    
    
class PICVariable(Variable):
    pass
    
    
class SPHVariable(Variable):
    pass