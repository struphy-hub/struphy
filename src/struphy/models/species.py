from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable
import numpy as np
from mpi4py import MPI

from struphy.fields_background.base import FluidEquilibrium
from struphy.initial.base import InitialCondition
from struphy.kinetic_background.base import KineticBackground
from struphy.io.options import Units
from struphy.physics.physics import ConstantsOfNature
from struphy.models.variables import Variable


class Species(metaclass=ABCMeta):
    """Single species of a StruphyModel."""
    
    # set species attribute for each variable
    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Variable):
                v._species = self.__class__.__name__
    
    @property
    def variables(self):
        if not hasattr(self, "_variables"):
            self._variables = {}
            for k, v in self.__dict__.items():
                if isinstance(v, Variable):
                    self._variables[k] = v 
        return self._variables
    
    @property
    def charge_number(self) -> int:
        """Charge number in units of elementary charge."""
        return self._charge_number
    
    @property
    def mass_number(self) -> int:
        """Mass number in units of proton mass."""
        return self._mass_number

    def set_phys_params(self, charge_number: int = 1, mass_number: int = 1):
        """Set charge- and mass number."""
        self._charge_number = charge_number
        self._mass_number = mass_number
    
    @property
    def equation_params(self) -> dict:
        if not hasattr(self, "_equation_params"):
            self.setup_equation_params()
        return self._equation_params
    
    def setup_equation_params(self, units: Units = None, verbose=False):
        """Set the following equation parameters:
        
        * alpha = plasma-frequenca / cyclotron frequency
        * epsilon = 1 / (cyclotron frequency * time unit)
        * kappa = plasma frequency * time unit
        """
        Z = self.charge_number
        A = self.mass_number
        
        if units is None:
            units = Units()

        con = ConstantsOfNature()
        
        # compute equation parameters
        self._equation_params = {}
        om_p = np.sqrt(units.n * (Z * con.e) ** 2 / (con.eps0 * A * con.mH))
        om_c = Z * con.e * units.B / (A * con.mH)
        self._equation_params["alpha"] = om_p / om_c
        self._equation_params["epsilon"] = 1.0 / (om_c * units.t)
        self._equation_params["kappa"] = om_p * units.t

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f'Set normalization parameters for species {self.__class__.__name__}:')
            for key, val in self.equation_params.items():
                print((key + ":").ljust(25), "{:4.3e}".format(val))


class FieldSpecies(Species):
    """Species without mass and charge (so-called 'fields')."""


class FluidSpecies(Species):
    """Single fluid species in 3d configuration space."""
    pass


class KineticSpecies(Species):
    """Single kinetic species in 3d + vdim phase space."""
    pass


class DiagnosticSpecies(Species):
    """Diagnostic species (fields) without mass and charge."""
    pass

    
