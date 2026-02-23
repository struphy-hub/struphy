import os
from dataclasses import dataclass
from typing import Literal

from struphy.utils.utils import check_option


@dataclass
class LiteralOptions:
    """
    (String) options for parameters in launch files, including:

    * Time stepping
    * Derham finite element spaces
    * Background/equilibrium types
    * Model types
    * Initial conditions / perturbations
    * Linear, nonlinear, explicit (Butcher) and saddle-point solvers
    * Drawing of markers, type of PIC methods
    * Noise filtering in PIC
    * Smoothing kernels for SPH methods
    * Particle binning in phase space
    """

    # time
    SplitAlgos = Literal["LieTrotter", "Strang"]

    # derham
    PolarRegularity = Literal[-1, 1]
    OptsFEECSpace = Literal["H1", "Hcurl", "Hdiv", "L2", "H1vec"]
    OptsVecSpace = Literal["Hcurl", "Hdiv", "H1vec"]

    # fields background
    BackgroundTypes = Literal["LogicalConst", "FluidEquilibrium"]

    # models
    ModelTypes = Literal["Toy", "Kinetic", "Fluid", "Hybrid"]

    # perturbations
    NoiseDirections = Literal["e1", "e2", "e3", "e1e2", "e1e3", "e2e3", "e1e2e3"]
    GivenInBasis = Literal["0", "1", "2", "3", "v", "physical", "physical_at_eta", "norm", None]

    # solvers
    OptsSymmSolver = Literal["pcg", "cg"]
    OptsGenSolver = Literal["pbicgstab", "bicgstab", "GMRES"]
    OptsMassPrecond = Literal["MassMatrixPreconditioner", "MassMatrixDiagonalPreconditioner", None]
    OptsSaddlePointSolver = Literal["Uzawa", "GMRES"]
    OptsDirectSolver = Literal["SparseSolver", "ScipySparse", "InexactNPInverse", "DirectNPInverse"]
    OptsNonlinearSolver = Literal["Picard", "Newton"]
    OptsButcher = Literal["rk4", "forward_euler", "heun2", "rk2", "heun3", "3/8 rule"]

    # markers
    OptsPICSpace = Literal["Particles6D", "DeltaFParticles6D", "Particles5D", "Particles3D"]
    OptsMarkerBC = Literal["periodic", "reflect"]
    OptsRecontructBC = Literal["periodic", "mirror", "fixed", "noslip"]
    OptsLoading = Literal[
        "pseudo_random",
        "sobol_standard",
        "sobol_antithetic",
        "external",
        "restart",
        "tesselation",
    ]
    OptsSpatialLoading = Literal["uniform", "disc"]
    OptsMPIsort = Literal["each", "last", None]

    # filters
    OptsFilter = Literal["fourier_in_tor", "hybrid", "three_point", None]

    # sph
    OptsKernel = Literal[
        "trigonometric_1d",
        "gaussian_1d",
        "linear_1d",
        "trigonometric_2d",
        "gaussian_2d",
        "linear_2d",
        "trigonometric_3d",
        "gaussian_3d",
        "linear_isotropic_3d",
        "linear_3d",
    ]

    # Create new Literal to determine type of binning output
    BinningQuantity = Literal[
        "density",
        "current_1",
        "current_2",
        "current_3",
        "energy_tensor_11",
        "energy_tensor_22",
        "energy_tensor_33",
        "energy_tensor_12",
        "energy_tensor_13",
        "energy_tensor_23",
        "heat_flux_1",
        "heat_flux_2",
        "heat_flux_3",
    ]


@dataclass
class Time:
    """Set options for time stepping in parameter/launch files.

    Parameters
    ----------
    dt : float
        Time step.

    Tend : float
        End time.

    split_algo : LiteralOptions.SplitAlgos
        Splitting algorithm (the order of the propagators is defined in the model).
    """

    dt: float = 0.01
    Tend: float = 0.03
    split_algo: LiteralOptions.SplitAlgos = "LieTrotter"

    def __post_init__(self):
        check_option(self.split_algo, LiteralOptions.SplitAlgos)

    def __repr__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""


@dataclass
class BaseUnits:
    """Set base units in parameter/launch files from which other units are derived. See :ref:`normalization`.

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

    def __repr__(self):
        units = ["m", "T", "1e20/m^3", "keV"]
        for (k, v), unit in zip(self.__dict__.items(), units):
            print(f"{k}:".ljust(20), v, unit)
        return ""


@dataclass
class DerhamOptions:
    """Set options for the Derham spaces in parameter/launch files. See :ref:`geomFE`.

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
    polar_ck: LiteralOptions.PolarRegularity = -1
    local_projectors: bool = False

    def __post_init__(self):
        check_option(self.polar_ck, LiteralOptions.PolarRegularity)

    def __repr__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""


@dataclass
class FieldsBackground:
    """Set options for static fluid backgrounds/equilibria in parameter/launch files.

    Parameters
    ----------
    type : BackgroundTypes
        Type of background.

    values : tuple[float]
        Values for LogicalConst on the unit cube.
        Can be length 1 for scalar functions; must be length 3 for vector-valued functions.

    variable : str
        Name of the method in :class:`~struphy.fields_background.base.FluidEquilibrium` that should be the background.
    """

    type: LiteralOptions.BackgroundTypes = "LogicalConst"
    values: tuple = (1.5, 0.7, 2.4)
    variable: str = None

    def __post_init__(self):
        check_option(self.type, LiteralOptions.BackgroundTypes)

    def __repr__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""


@dataclass
class EnvironmentOptions:
    """Set environment options for launching run on current architecture
    (these options do not influence the simulation result).

    Parameters
    ----------
    out_folders : str
        Absolute path to directory for ``sim_folder``.

    sim_folder : str
        Folder in ``out_folders/`` for the current simulation (default= ``sim_1/`` ).
        Will create the folder if it does not exist OR cleans the folder for new runs.

    restart : bool
        Whether to restart a run (default=False).

    max_runtime : int
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).

    save_step : int
        When to save data output: every time step (save_step=1), every second time step (save_step=2), etc (default=1).

    sort_step: int, optional
        Sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)

    num_clones: int, optional
        Number of domain clones (default=1)

    profiling_activated: bool, optional
        Activate profiling with scope-profiler (default=False)

    profiling_trace: bool, optional
        Save time-trace of each profiling region (default=False)
    """

    out_folders: str = os.getcwd()
    sim_folder: str = "sim_1"
    restart: bool = False
    max_runtime: int = 300
    save_step: int = 1
    sort_step: int = 0
    num_clones: int = 1
    profiling_activated: bool = False
    profiling_trace: bool = False

    def __post_init__(self):
        self.path_out: str = os.path.join(self.out_folders, self.sim_folder)

    def __repr__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""
