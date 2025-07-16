from copy import deepcopy
from typing import Callable

from struphy.fields_background.base import FluidEquilibrium
from struphy.initial.base import InitialCondition
from struphy.kinetic_background.base import KineticBackground


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
    ):
        assert variable in self.all, f"Variable {variable} is not part of model variables {self.all.keys()}"
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_background(background)

    def add_perturbation(
        self,
        variable: str,
        perturbation: Callable,
        given_in_basis: tuple = None,
    ):
        assert variable in self.all, f"Variable {variable} is not part of model variables {self.all.keys()}"
        var = getattr(self, variable)
        assert isinstance(var, Variable)
        var.add_perturbation(
            perturbation=perturbation,
            given_in_basis=given_in_basis,
        )

    def set_options(self, propagator):
        pass


class Variable:
    """Single variable (unknown) of a StruphyModel."""

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

    @property
    def perturbation(self):
        return self._perturbation

    @property
    def initial_condition(self):
        if not hasattr(self, "_initial_condition"):
            self.set_initial_condition()
        return self._initial_condition

    ## methods

    def add_background(self, background):
        # assert isinstance(...)
        self._background += [background]

    def add_perturbation(
        self,
        perturbation: Callable = None,
        given_in_basis: tuple = None,
    ):
        # assert isinstance(...)
        self._perturbation += [(perturbation, given_in_basis)]

    def set_initial_condition(self):
        self._initial_condition = InitialCondition(
            background=self.background,
            perturbation=self.perturbation,
        )
