from struphy.fields_background.base import FluidEquilibrium
from struphy.geometry.base import Domain
from struphy.topology import grids
from struphy.io.options import Units, Time, DerhamOptions


class StruphyParameters:
    """Wrapper around Struphy simulation parameters."""

    def __init__(
        self,
        model: str = None,
        domain: Domain = None,
        grid: grids.TensorProductGrid = None,
        equil: FluidEquilibrium = None,
        units: Units = None,
        time: Time = None,
        derham: DerhamOptions = None,
        em_fields=None,
        fluid=None,
        kinetic=None,
        diagnostic_fields=None,
    ):

        self._model = model
        self._domain = domain
        self._grid = grid
        self._equil = equil
        self._units = units
        self._time = time
        self._derham = derham
        self._em_fields = em_fields
        self._fluid = fluid
        self._kinetic = kinetic
        self._diagnostic_fields = diagnostic_fields

    @property
    def model(self):
        return self._model

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self._grid

    @property
    def equil(self):
        return self._equil

    @property
    def units(self):
        return self._units

    @property
    def time(self):
        return self._time

    @property
    def derham(self):
        return self._derham

    @property
    def em_fields(self):
        return self._em_fields

    @property
    def fluid(self):
        return self._fluid

    @property
    def kinetic(self):
        return self._kinetic

    @property
    def diagnostic_fields(self):
        return self._diagnostic_fields
