from copy import deepcopy
from typing import Callable
import numpy as np
from mpi4py import MPI

from struphy.fields_background.base import FluidEquilibrium
from struphy.initial.base import InitialCondition
from struphy.kinetic_background.base import KineticBackground
from struphy.io.options import Units
from struphy.physics.physics import ConstantsOfNature


class Species:
    """Handles the three species types in StruphyModel: em_fields, fluid, kinetic."""

    def __init__(
        self,
        fluid: tuple = None,
        kinetic: tuple = None,
        em_fields: bool = True,
    ):
        # fluid species
        if fluid is None:
            self._fluid = None
        else:
            self._fluid = MultiSpecies()
            for f in fluid:
                self._fluid.add_species(name=f)

        # kinetic species
        if kinetic is None:
            self._kinetic = None
        else:
            self._kinetic = MultiSpecies()
            for k in kinetic:
                self._kinetic.add_species(name=k)

        # electromagnetic fields
        if em_fields:
            self._em_fields = SubSpecies(name="em_fields")

    @property
    def fluid(self):
        return self._fluid

    @property
    def kinetic(self):
        return self._kinetic

    @property
    def em_fields(self):
        return self._em_fields


class MultiSpecies:
    """Handles multiple fluid or kinetic species."""

    def __init__(
        self,
    ):
        pass

    @property
    def all(self):
        return self.__dict__

    def add_species(self, name: str = "mhd"):
        setattr(self, name, SubSpecies(name))


class SubSpecies:
    """Handles the three species types in StruphyModel: em_fields, fluid, kinetic."""

    def __init__(self, name: str = "mhd"):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def all(self):
        out = deepcopy(self.__dict__)
        out.pop("_name")
        return out

    def add_variable(self, name: str = "velocity", space: str = "Hdiv"):
        setattr(self, name, Variable(name, space))

    def add_background(
        self,
        variable: str,
        background,
        verbose=False,
    ):
        assert variable in self.all, f"Variable {variable} is not part of model variables {self.all.keys()}"
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_background(background, verbose=verbose)

    def add_perturbation(
        self,
        variable: str,
        perturbation: Callable,
        given_in_basis: tuple = None,
        verbose=False,
    ):
        assert variable in self.all, f"Variable {variable} is not part of model variables {self.all.keys()}"
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_perturbation(
            perturbation=perturbation,
            given_in_basis=given_in_basis,
            verbose=verbose,
        )

    @property
    def Z(self) -> int:
        """Charge number in units of elementary charge."""
        return self._Z
    
    @property
    def A(self) -> int:
        """Mass number in units of proton mass."""
        return self._A

    def set_phys_params(self, Z: int = 1, A: int = 1):
        """Set charge- and mass number."""
        self._Z = Z
        self._A = A
    
    @property
    def equation_params(self) -> dict:
        if not hasattr(self, "_equation_params"):
            self.set_equation_params()
        return self._equation_params
    
    def set_equation_params(self, units: Units = None, verbose=False):
        Z = self.Z
        A = self.A
        
        if units is None:
            units = Units()

        con = ConstantsOfNature()
        
        # compute equation parameters
        self._equation_params = {}
        om_p = np.sqrt(units.n * (Z * con.e) ** 2 / (con.eps0 * A * con.mH))
        om_c = Z * con.e * units.B / (A * con.mH)
        self._equation_params["alpha"] = om_p / om_c
        self._equation_params["epsilon"] = 1.0 / (om_c * units["t"])
        self._equation_params["kappa"] = om_p * units["t"]

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f'Set normalization parameters for speceis {self.name}:')
            for key, val in self.equation_params.items():
                print((key + ":").ljust(25), "{:4.3e}".format(val))


class Variable:
    """Single variable (unknown) of a StruphyModel."""

    def __init__(self, name: str, space: str, save_data: bool = True):
        self._name = name
        self._space = space
        self._save_data = save_data

        self._background = []
        self._perturbation = []
        
        self._has_particles = False
        if "Particles" in space:
            self._has_particles = True
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

    @property
    def perturbation(self):
        return self._perturbation
    
    @property
    def has_particles(self):
        return self._has_particles

    @property
    def initial_condition(self):
        if not hasattr(self, "_initial_condition"):
            self.set_initial_condition()
        return self._initial_condition

    ## methods

    def add_background(self, background, verbose=False,):
        # assert isinstance(...)
        self._background += [background]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Variable '{self.name}': added background '{background.__class__.__name__}' with:")
            for k, v in background.__dict__.items():
                print(f'  {k}: {v}')

    def add_perturbation(
        self,
        perturbation: Callable = None,
        given_in_basis: tuple = None,
        verbose=False,
    ):
        # assert isinstance(...)
        self._perturbation += [(perturbation, given_in_basis)]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Variable '{self.name}': added perturbation '{perturbation.__class__.__name__}',{given_in_basis = }, with:")
            for k, v in perturbation.__dict__.items():
                print(f'  {k}: {v}')

    def set_initial_condition(self):
        self._initial_condition = InitialCondition(
            background=self.background,
            perturbation=self.perturbation,
        )
