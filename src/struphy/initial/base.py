from typing import Callable
from abc import ABCMeta, abstractmethod

from struphy.fields_background.base import FluidEquilibrium
from struphy.io.options import FieldsBackground
from struphy.kinetic_background.base import KineticBackground
from struphy.io.options import GivenInBasis, check_option


class Perturbation(metaclass=ABCMeta):
    """Base class for perturbations that can be chosen as initial conditions."""
    
    @abstractmethod
    def __call__(self, eta1, eta2, eta3, flat_eval=False):
        pass
    
    def prepare_eval_pts(self):
        # TODO: we could prepare the arguments via a method in this base class (flat_eval, sparse meshgrid, etc.).
        pass
    
    @property
    def given_in_basis(self) -> str:
        r"""In which basis the perturbation is represented, must be set in child class (use the setter below).
        
        Either
            * '0', '1', '2' or  '3' for a p-form basis
            * 'v' for a vector-field basis
            * 'physical' when defined on the physical (mapped) domain
            * 'physical_at_eta' when given the physical components evaluated on the logical domain, u(F(eta))
            * 'norm' when given in the normalized co-variant basis (:math:`\delta_i / |\delta_i|`)
        """
        return self._given_in_basis

    @given_in_basis.setter
    def given_in_basis(self, new: str):
        check_option(new, GivenInBasis)
        self._given_in_basis = new
        
    @property
    def comp(self) -> int:
        """Which component of vector is perturbed (=0 for scalar-valued functions).
        Can be set in child class (use the setter below)."""
        if not hasattr(self, "_comp"):
            self._comp = 0
        return self._comp
    
    @comp.setter
    def comp(self, new: int):
        assert new in (0, 1, 2)
        self._comp = new
        


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
        