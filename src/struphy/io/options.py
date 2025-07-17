from dataclasses import dataclass
from typing import Literal, get_args
import numpy as np

# needed for import in StruphyParameters
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.topology import grids
from struphy.physics.physics import ConstantsOfNature

## generic options

SplitAlgos = Literal["LieTrotter", "Strang"]


class Units:
    """
    Base units are passed to __init__, other units derive from these.

    Parameters
    ----------
    x : float
        Unit of length in meters.

    B : float
        Unit of magnetic field in Tesla.

    n : float
        Unit of particle number density in 1e20/m^3.

    kBT : float, optional
        Unit of internal energy in keV. 
        Only in effect if the velocity scale is set to 'thermal'.
    """

    def __init__(self, 
        x: float = 1.0,
        B: float = 1.0,
        n: float = 1.0,
        kBT: float = None,):
        
        self._x = x
        self._B = B
        self._n = n * 1e20
        self._kBT = kBT
        
    @property
    def x(self):
        return self._x
    
    @property
    def B(self):
        return self._B
    
    @property
    def n(self):
        """Unit of particle number density in 1/m^3."""
        return self._n
    
    @property
    def kBT(self):
        return self._kBT
    
    @property
    def v(self):
        """Unit of velocity in m/s."""
        return self._v
    
    @property
    def t(self):
        """Unit of time in s."""
        return self._t
    
    @property
    def p(self):
        """Unit of pressure in Pa, equal to B^2/mu0 if velocity_scale='alfvén'."""
        return self._p
    
    @property
    def rho(self):
        """Unit of mass density in kg/m^3."""
        return self._rho
    
    @property
    def j(self):
        """Unit of current density in A/m^2."""
        return self._j
    
    def derive_units(self, velocity_scale: str = "light", A_bulk: int = None, Z_bulk: int = None,
                     verbose=False):
        """Derive the remaining units from the base units, velocity scale and bulk species' A and Z."""

        from mpi4py import MPI

        con = ConstantsOfNature()

        # velocity (m/s)
        if velocity_scale is None:
            self._v = 1.0

        elif velocity_scale == "light":
            self._v = con.c

        elif velocity_scale == "alfvén":
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "alfvén".'
            self._v = self.B / np.sqrt(self.n * A_bulk * con.mH * con.mu0)

        elif velocity_scale == "cyclotron":
            assert Z_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
            self._v = Z_bulk * con.e * self.B / (A_bulk * con.mH) * self.x

        elif velocity_scale == "thermal":
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "thermal".'
            assert self.kBT is not None
            self._v = np.sqrt(self.kBT * 1000 * con.e / (con.mH * A_bulk))

        # time (s)
        self._t = self.x / self.v
        
        # return if no bulk is present
        if A_bulk is None:
            self._p = None
            self._rho = None
            self._j = None
        else:
            # pressure (Pa), equal to B^2/mu0 if velocity_scale='alfvén'
            self._p = A_bulk * con.mH * self.n * self.v ** 2

            # mass density (kg/m^3)
            self._rho = A_bulk * con.mH * self.n

            # current density (A/m^2)
            self._j = con.e * self.n * self.v
        
        # print to screen
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            units_used = (" m", " T", " m⁻³", "keV", " m/s", " s", " bar", " kg/m³", " A/m²",)
            print("\nUNITS:")
            for (k, v), u in zip(self.__dict__.items(), units_used):
                if v is None:
                    print(f"Unit of {k[1:]} not specified.")
                else:
                    print(
                        f"Unit of {k[1:]}:".ljust(25),
                        "{:4.3e}".format(v) + u,
                    )


@dataclass
class Time:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    dt: float = 1.0
    Tend: float = 1.0
    split_algo: SplitAlgos = "LieTrotter"

    def __post_init__(self):
        options = get_args(SplitAlgos)
        assert self.split_algo in options, f"'{self.split_algo}' is not in {options}"


## field options

PolarRegularity = Literal[-1, 1]
BackgroundOpts = Literal["LogicalConst", "FluidEquilibrium"]


@dataclass
class DerhamOptions:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    polar_ck: int = -1
    local_projectors: bool = False

    def __post_init__(self):
        options = get_args(PolarRegularity)
        assert self.polar_ck in options, f"'{self.polar_ck}' is not in {options}"


@dataclass
class FieldsBackground:
    """...

    Parameters
    ----------
    x : float
        Unit of length in m.
    """

    kind: str = "LogicalConst"
    values: tuple = (1.5,)
    variable: str = None

    def __post_init__(self):
        options = get_args(BackgroundOpts)
        assert self.kind in options, f"'{self.kind}' is not in {options}"


## kinetic options
