from __future__ import annotations
from mpi4py import MPI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from struphy.fields_background.base import FluidEquilibrium
    from struphy.geometry.base import Domain
    from struphy.io.options import DerhamOptions, Time, Units
    from struphy.topology import grids
    from struphy.models.base import StruphyModel


class StruphyParameters:
    """Wrapper around simulation parameters."""

    def __init__(
        self,
        model: StruphyModel = None,
        units: Units = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        time: Time = None,
        grid: grids.TensorProductGrid = None,
        derham: DerhamOptions = None,
        verbose: bool = False
    ):
        self._model = model
        self._units = units
        self._domain = domain
        self._equil = equil
        self._time = time
        self._grid = grid
        self._derham = derham
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n{self.model = }")
            print(f"{self.units = }")
            print(f"{self.domain = }")
            print(f"{self.equil = }")
            print(f"{self.time = }")
            print(f"{self.grid = }")
            print(f"{self.derham = }\n")

    @property
    def model(self):
        return self._model

    @property
    def units(self):
        return self._units
    
    @property
    def domain(self):
        return self._domain

    @property
    def equil(self):
        return self._equil
    
    @property
    def time(self):
        return self._time
    
    @property
    def grid(self):
        return self._grid

    @property
    def derham(self):
        return self._derham
