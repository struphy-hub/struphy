from dataclasses import dataclass
from typing import Literal, get_args
import numpy as np
import os

from struphy.physics.physics import ConstantsOfNature


## Literal options

# time
SplitAlgos = Literal["LieTrotter", "Strang"]

# derham
PolarRegularity = Literal[-1, 1]
OptsFEECSpace = Literal["H1", "Hcurl", "Hdiv", "L2", "H1vec"]
OptsVecSpace = Literal["Hcurl", "Hdiv", "H1vec"]

# fields background
BackgroundTypes = Literal["LogicalConst", "FluidEquilibrium"]

# perturbations
NoiseDirections = Literal["e1", "e2", "e3", "e1e2", "e1e3", "e2e3", "e1e2e3"]
GivenInBasis = Literal['0', '1', '2', '3', 'v', 'physical', 'physical_at_eta', 'norm', None]

# solvers
OptsSymmSolver = Literal["pcg", "cg"]
OptsGenSolver = Literal["pbicgstab", "bicgstab"]
OptsMassPrecond = Literal["MassMatrixPreconditioner", None]

# markers
OptsPICSpace = Literal["Particles6D", "DeltaFParticles6D", "Particles5D", "Particles3D"]
OptsMarkerBC = Literal["periodic", "reflect"]
OptsRecontructBC = Literal["periodic", "mirror", "fixed"]
OptsLoading = Literal["pseudo_random", 'sobol_standard', 'sobol_antithetic', 'external', 'restart', "tesselation"]
OptsSpatialLoading = Literal["uniform", "disc"]


## Option classes

@dataclass
class Time:
    """Time stepping options.

    Parameters
    ----------
    dt : float
        Time step.
        
    Tend : float
        End time.
        
    split_algo : SplitAlgos
        Splitting algorithm (the order of the propagators is defined in the model).
    """

    dt: float = 0.01
    Tend: float = 0.03
    split_algo: SplitAlgos = "LieTrotter"

    def __post_init__(self):
        check_option(self.split_algo, SplitAlgos)


@dataclass
class BaseUnits:
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
    x: float = 1.0
    B: float = 1.0
    n: float = 1.0
    kBT: float = None


class Units:
    """
    Colllects base units and derives other units from these.
    """

    def __init__(self, base: BaseUnits = None):
        if base is None:
            base = BaseUnits()
        
        self._x = base.x
        self._B = base.B
        self._n = base.n * 1e20
        self._kBT = base.kBT
        
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
            print("")
            for (k, v), u in zip(self.__dict__.items(), units_used):
                if v is None:
                    print(f"Unit of {k[1:]} not specified.")
                else:
                    print(
                        f"Unit of {k[1:]}:".ljust(25),
                        "{:4.3e}".format(v) + u,
                    )


@dataclass
class DerhamOptions:
    """Options for the Derham spaces.

    Parameters
    ----------
    p : tuple[int]
        Spline degree in each direction.

    spl_kind : tuple[bool]
        Kind of spline in each direction (True=periodic, False=clamped).

    dirichlet_bc : tuple[tuple[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).
        
    nquads : tuple[int]
        Number of Gauss-Legendre quadrature points in each direction (default = p, leads to exact integration of degree 2p-1 polynomials).

    nq_pr : tuple[int]
        Number of Gauss-Legendre quadrature points in each direction for geometric projectors (default = p+1, leads to exact integration of degree 2p+1 polynomials).
    
    polar_ck : PolarRegularity
        Smoothness at a polar singularity at eta_1=0 (default -1 : standard tensor product splines, OR 1 : C1 polar splines)

    local_projectors : bool
        Whether to build the local commuting projectors based on quasi-inter-/histopolation.
    """
    p: tuple = (1, 1, 1)
    spl_kind: tuple = (True, True, True)
    dirichlet_bc: tuple = ((False, False), (False, False), (False, False))
    nquads: tuple = None
    nq_pr: tuple = None
    polar_ck: PolarRegularity = -1
    local_projectors: bool = False

    def __post_init__(self):
        check_option(self.polar_ck, PolarRegularity)


@dataclass
class FieldsBackground:
    """Options for backgrounds in configuration (=position) space.
 
    Parameters
    ----------
    type : BackgroundTypes
        Type of background.
        
    values : tuple[float]
        Values for LogicalConst on the unit cube. 
        Can be length 1 for scalar functions; must be length 3 for vector-valued functions.
        
    variable : str
        Name of the function in FluidEquilibrium that should be the background.
    """

    type: BackgroundTypes = "LogicalConst"
    values: tuple = (1.5, 0.7, 2.4)
    variable: str = None

    def __post_init__(self):
        check_option(self.type, BackgroundTypes)
        
        
@dataclass
class EnvironmentOptions:
    """Environment options for launching run on current architecture 
    (these options do not influence the simulation result). 

    Parameters
    ----------
    out_folders : str
        The directory where all sim_folders are stored. 
        
    sim_folder : str
        Folder in 'out_folders/' for the current simulation (default='sim_1').
        Will create the folder if it does not exist OR cleans the folder for new runs.
        
    restart : bool
        Whether to restart a run (default=False).

    max_runtime : int,
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).

    save_step : int
        When to save data output: every time step (save_step=1), every second time step (save_step=2), etc (default=1).

    sort_step: int, optional
        Sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)

    num_clones: int, optional
        Number of domain clones (default=1)
    """

    out_folders: str = os.getcwd()
    sim_folder: str = "sim_1"
    restart: bool = False
    max_runtime: int = 300
    save_step: int = 1
    sort_step: int = 0
    num_clones: int = 1
    
    def __post_init__(self):
        self.path_out: str = os.path.join(self.out_folders, self.sim_folder)
        
    def __repr__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
    
    
def check_option(opt, options):
    """Check if opt is contained in options; if opt is a list, checks for each element."""
    opts = get_args(options)
    if not isinstance(opt, list):
        opt = [opt]
    for o in opt:
        assert o in opts, f"Option '{o}' is not in {opts}." 

