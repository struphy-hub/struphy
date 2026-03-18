import os
from dataclasses import dataclass, fields
from typing import Any, Callable, Literal

from struphy.utils.utils import (
    __class_with_params_repr_no_defaults__,
    __dataclass_repr_no_defaults__,
    all_class_params_are_default,
    check_option,
)


class OptionsBase:
    def to_dict(self) -> dict:
        """Convert dataclass instance to dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(type(self)) if field.init}

    @classmethod
    def from_dict(cls, dct) -> "Any":
        """Create dataclass instance from dictionary."""
        valid_fields = {field.name for field in fields(cls) if field.init}
        return cls(**{key: value for key, value in dct.items() if key in valid_fields})


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
    OptsNonTrivialBoundaryCondition = Literal["free", "hom_dirichlet", "lifting"]

    # fields background
    BackgroundTypes = Literal["LogicalConst", "FluidEquilibrium"]

    # models
    ModelTypes = Literal["Toy", "Kinetic", "Fluid", "Hybrid"]

    # perturbations
    NoiseDirections = Literal["e1", "e2", "e3", "e1e2", "e1e3", "e2e3", "e1e2e3"]
    GivenInBasis = Literal["0", "1", "2", "3", "v", "physical", "physical_at_eta", "norm", None]

    # solvers
    OptsSymmSolver = Literal["pcg", "cg"]
    OptsGenSolver = Literal["pbicgstab", "bicgstab", "gmres"]
    OptsMassPrecond = Literal["MassMatrixPreconditioner", "MassMatrixDiagonalPreconditioner", None]
    OptsSaddlePointSolver = Literal["uzawa"]
    OptsDirectSolver = Literal["SparseSolver", "ScipySparse", "InexactNPInverse", "DirectNPInverse"]
    OptsNonlinearSolver = Literal["Picard", "Newton"]
    OptsButcher = Literal["rk4", "forward_euler", "heun2", "rk2", "heun3", "3/8 rule"]

    # markers
    OptsPICSpace = Literal["Particles6D", "DeltaFParticles6D", "Particles5D", "Particles3D"]
    OptsMarkerBC = Literal["periodic", "reflect"]
    OptsRecontructBC = Literal["periodic", "mirror", "fixed"]
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
class Time(OptionsBase):
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

    def __str__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


@dataclass
class BaseUnits(OptionsBase):
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

    def __str__(self):
        units = ["m", "T", "1e20/m^3", "keV"]
        for (k, v), unit in zip(self.__dict__.items(), units):
            print(f"{k}:".ljust(20), v, unit)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


NonTrivialBC = LiteralOptions.OptsNonTrivialBoundaryCondition


@dataclass
class DerhamOptions:
    """Set options for the 3D discrete de Rham spaces in parameter/launch files.

    This dataclass controls spline degrees, boundary conditions and quadrature
    settings used to build the FEEC de Rham sequence. See :ref:`geomFE`.

    Parameters
    ----------
    p : tuple[int, int, int]
        Spline degree in each logical direction ``(eta_1, eta_2, eta_3)``.

    bcs : tuple[None | tuple[NonTrivialBC, NonTrivialBC], None | tuple[NonTrivialBC, NonTrivialBC], None | tuple[NonTrivialBC, NonTrivialBC]] | None
        Boundary condition selector for each direction.
        Use ``None`` in a direction for periodic boundaries, or a tuple
        ``(left, right)`` with entries in ``{"free", "hom_dirichlet",
        "lifting"}`` for non-periodic boundaries. If ``bcs`` itself is
        ``None``, all directions are periodic.

    lifting_eta1 : tuple[float | Callable | None, float | Callable | None] | None
    lifting_eta2 : tuple[float | Callable | None, float | Callable | None] | None
    lifting_eta3 : tuple[float | Callable | None, float | Callable | None] | None
        Boundary lifting data on left/right faces in directions
        ``eta_1``, ``eta_2`` and ``eta_3``. Values are required only on sides
        where the corresponding entry in ``bcs`` is ``"lifting"``.
        Each side can be given as a constant, a callable, or ``None`` when
        unused.

    nquads : tuple[int, int, int] | None
        Number of Gauss-Legendre quadrature points per direction for cell
        integrals. If ``None``, backend defaults are used.

    nq_pr : tuple[int, int, int] | None
        Number of Gauss-Legendre quadrature points per direction for geometric
        projector/histopolation integrals. If ``None``, backend defaults are
        used.

    polar_ck : PolarRegularity
        Smoothness at a possible polar singularity at ``eta_1 = 0``.
        ``-1`` gives standard tensor-product splines, ``1`` gives C1 polar
        splines.

    local_projectors : bool
        Whether to build local commuting projectors based on
        quasi-inter-/histopolation.
    """

    p: tuple = (1, 1, 1)
    bcs: (
        tuple[
            None | tuple[NonTrivialBC, NonTrivialBC],
            None | tuple[NonTrivialBC, NonTrivialBC],
            None | tuple[NonTrivialBC, NonTrivialBC],
        ]
        | None
    ) = None
    lifting_eta1: tuple[float | Callable | None, float | Callable | None] | None = None
    lifting_eta2: tuple[float | Callable | None, float | Callable | None] | None = None
    lifting_eta3: tuple[float | Callable | None, float | Callable | None] | None = None
    nquads: tuple[int, int, int] | None = None
    nq_pr: tuple[int, int, int] | None = None
    polar_ck: LiteralOptions.PolarRegularity = -1
    local_projectors: bool = False

    def __post_init__(self):
        check_option(self.polar_ck, LiteralOptions.PolarRegularity)
        if self.bcs is not None:
            for bc in self.bcs:
                if bc is not None:
                    check_option(bc, LiteralOptions.OptsNonTrivialBoundaryCondition)

    def __str__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)

    def to_dict(self) -> dict:
        dct = {
            "p": self.p,
            "bcs": self.bcs,
            "lifting_eta1": self.lifting_eta1,
            "lifting_eta2": self.lifting_eta2,
            "lifting_eta3": self.lifting_eta3,
            "nquads": self.nquads,
            "nq_pr": self.nq_pr,
            "polar_ck": self.polar_ck,
            "local_projectors": self.local_projectors,
        }
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> "DerhamOptions":
        return cls(
            p=dct["p"],
            bcs=dct["bcs"],
            lifting_eta1=dct.get("lifting_eta1", None),
            lifting_eta2=dct.get("lifting_eta2", None),
            lifting_eta3=dct.get("lifting_eta3", None),
            nquads=dct["nquads"],
            nq_pr=dct["nq_pr"],
            polar_ck=dct["polar_ck"],
            local_projectors=dct["local_projectors"],
        )


@dataclass
class FieldsBackground(OptionsBase):
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

    def __str__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)

    def to_dict(self) -> dict:
        dct = {
            "type": self.type,
            "values": self.values,
            "variable": self.variable,
        }
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> "FieldsBackground":
        return cls(
            type=dct["type"],
            values=dct["values"],
            variable=dct["variable"],
        )


@dataclass
class EnvironmentOptions(OptionsBase):
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

    def __str__(self):
        for k, v in self.__dict__.items():
            print(f"{k}:".ljust(20), v)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)

    def to_dict(self) -> dict:
        dct = {
            "out_folders": self.out_folders,
            "sim_folder": self.sim_folder,
            "restart": self.restart,
            "max_runtime": self.max_runtime,
            "save_step": self.save_step,
            "sort_step": self.sort_step,
            "num_clones": self.num_clones,
            "profiling_activated": self.profiling_activated,
            "profiling_trace": self.profiling_trace,
        }
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> "EnvironmentOptions":
        return cls(
            out_folders=dct["out_folders"],
            sim_folder=dct["sim_folder"],
            restart=dct["restart"],
            max_runtime=dct["max_runtime"],
            save_step=dct["save_step"],
            sort_step=dct["sort_step"],
            num_clones=dct["num_clones"],
            profiling_activated=dct["profiling_activated"],
            profiling_trace=dct["profiling_trace"],
        )
