import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Callable, Literal

from struphy.utils.utils import (
    __class_with_params_repr_no_defaults__,
    __dataclass_repr_all_stacked__,
    __dataclass_repr_no_defaults__,
    all_class_params_are_default,
    check_option,
)

logger = logging.getLogger("struphy")


@dataclass
class OptionsBase:
    def to_dict(self) -> dict:
        """Convert dataclass instance to dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(type(self)) if field.init}

    @classmethod
    def from_dict(cls, dct) -> "Any":
        """Create dataclass instance from dictionary."""
        valid_fields = {field.name for field in fields(cls) if field.init}
        return cls(**{key: value for key, value in dct.items() if key in valid_fields})

    def __repr__(self):
        return __dataclass_repr_all_stacked__(self)


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
    OptsFEECSpace = Literal["H1", "Hcurl", "Hdiv", "L2", "H1vec"]
    OptsVecSpace = Literal["Hcurl", "Hdiv", "H1vec"]
    OptsNonTrivialBoundaryCondition = Literal["free", "dirichlet"]

    # fields backgrounds
    BackgroundTypes = Literal["LogicalConst", "FluidEquilibrium"]

    # kinetic backgrounds
    VelocityCoordinates = Literal["cartesian", "vpara_mu", "vpara_vperp", "vpara_energy"]
    OptsGaussianCoordinate = Literal["cartesian", "polar", "mu", "energy"]
    KineticDimensionsToPlot = Literal["e1", "e2", "e3", "v1", "v2", "v3"]

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
    OptsPICSpace = Literal["Particles6D", "DeltaFParticles6D", "Particles5D", "Particles5Dvperp", "Particles3D"]
    OptsMarkerBC = Literal["periodic", "reflect", "remove", "refill"]
    OptsRecontructBC = Literal["periodic", "mirror", "fixed", "noslip"]
    OptsRefillBC = Literal["inner", "outer"]
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


@dataclass(repr=False)
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

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


NonTrivialBC = LiteralOptions.OptsNonTrivialBoundaryCondition


@dataclass(repr=False)
class DerhamOptions(OptionsBase):
    """Set options for the 3D discrete de Rham spaces in parameter/launch files.

    This dataclass controls spline degrees, boundary conditions and quadrature
    settings used to build the FEEC de Rham sequence. See :ref:`geomFE`.

    Parameters
    ----------
    degree : tuple[int, int, int]
        Spline degree in each logical direction ``(eta_1, eta_2, eta_3)``.

    bcs : tuple[None | tuple[NonTrivialBC, NonTrivialBC], None | tuple[NonTrivialBC, NonTrivialBC], None | tuple[NonTrivialBC, NonTrivialBC]]
        Boundary condition selector for each direction.
        Use ``None`` in a direction for periodic boundaries, or a tuple
        ``(left, right)`` with entries in ``{"free", "dirichlet"}`` for non-periodic boundaries.

    nquads : tuple[int, int, int] | None
        Number of Gauss-Legendre quadrature points per direction for cell
        integrals. If ``None``, backend defaults are used.

    nquads_proj : tuple[int, int, int] | None
        Number of Gauss-Legendre quadrature points per direction for geometric
        projector/histopolation integrals. If ``None``, backend defaults are
        used.

    polar_splines : bool
        Smoothness at a possible polar singularity at ``eta_1 = 0``.
        ``False`` gives standard tensor-product splines, ``True`` gives C1 polar
        splines.

    local_projectors : bool
        Whether to build local commuting projectors based on
        quasi-inter-/histopolation.
    """

    degree: tuple[int, int, int] = (1, 1, 1)
    bcs: tuple[
        None | tuple[NonTrivialBC, NonTrivialBC],
        None | tuple[NonTrivialBC, NonTrivialBC],
        None | tuple[NonTrivialBC, NonTrivialBC],
    ] = (None, None, None)
    nquads: tuple[int, int, int] | None = None
    nquads_proj: tuple[int, int, int] | None = None
    polar_splines: bool = False
    local_projectors: bool = False

    def __post_init__(self):
        for bc in self.bcs:
            if bc is not None:
                assert isinstance(bc, tuple) and len(bc) == 2, (
                    "Boundary conditions must be given as a tuple (left, right) for each direction."
                )
                check_option(bc[0], LiteralOptions.OptsNonTrivialBoundaryCondition)
                check_option(bc[1], LiteralOptions.OptsNonTrivialBoundaryCondition)

    # def __str__(self):
    #     for k, v in self.__dict__.items():
    #         logger.info(f"{k + ':':<20}{v}")
    #     return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    def __repr_all_stacked__(self):
        return __dataclass_repr_all_stacked__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


@dataclass(repr=False)
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

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


@dataclass(repr=False)
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

    sim_label: str | None, optional
        Label for the simulation (default=None)

    restart : bool
        Whether to restart a run (default=False).

    max_runtime : int
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).

    save_step : int
        When to save data output: every time step (save_step=1), every second time step (save_step=2), etc (default=1).

    save_restart : bool
        Whether to write the restart checkpoint (full marker arrays and FEEC
        restart coefficients, written once at setup and once at the end of
        the run). Restart data can dominate the run time for large particle
        counts; set to ``False`` to skip it when restart capability isn't
        needed, e.g. for pure timing/benchmark runs (default=True).

    sort_step: int, optional
        Sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)

    num_clones: int, optional
        Number of domain clones (default=1)

    profiling_activated: bool, optional
        Activate profiling with scope-profiler (default=False)
    """

    out_folders: str = os.getcwd()
    sim_folder: str = "sim_1"
    sim_label: str | None = None
    restart: bool = False
    max_runtime: int = 300
    save_step: int = 1
    save_restart: bool = True
    sort_step: int = 0
    num_clones: int = 1
    profiling_activated: bool = False

    def __post_init__(self):
        self.path_out: str = os.path.join(self.out_folders, self.sim_folder)

    # def __str__(self):
    #     for k, v in self.__dict__.items():
    #         logger.info(f"{k + ':':<20}{v}")
    #     return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    def __repr_all_stacked__(self):
        return __dataclass_repr_all_stacked__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)


@dataclass(repr=False)
class ProfilingOptions(OptionsBase):
    """Set options passed to :meth:`scope_profiler.ProfileManager.session`.

    ``EnvironmentOptions.profiling_activated`` remains the main switch used by
    Struphy to enable profiling. These options configure how scope-profiler
    collects and writes profiling data when that switch is active.

    Parameters
    ----------
    deactivate_file_output : bool | None
        Write no HDF5 profiling output if True.

    use_likwid : bool | None
        Enable LIKWID hardware counter collection.

    use_line_profiler : bool | None
        Enable line-by-line profiling via line_profiler.

    use_nvtx : bool | None
        Add NVTX ranges for NVIDIA Nsight tools.

    use_gpu_timing : bool | None
        Record CUDA-event elapsed device time for each profiled region.

    gpu_timing_backend : str | object | None
        CUDA-event backend for GPU timing. scope-profiler accepts ``"auto"``,
        ``"torch"``, ``"cupy"``, or a backend object.

    recursive_profile : bool | None
        Enable recursive profiling for decorated functions by default.

    buffer_limit : int | None
        Initial number of profiling events preallocated per region.

    file_path : str | os.PathLike | None
        Path to the output profiling data file. If None, Struphy writes
        ``profiling_data.h5`` in the simulation output folder.

    capture_region_source : bool | None
        Record source locations/text for profiling regions.

    config_path : str | os.PathLike | None
        TOML file containing a scope-profiler ``[profiling]`` table.

    verbose : bool
        Print scope-profiler finalization output.

    return_results : bool
        Return finalized results from the session context manager.

    native_traces : str | os.PathLike | tuple[str | os.PathLike, ...] | None
        Native trace path(s) to merge during scope-profiler finalization.
    """

    deactivate_file_output: bool | None = None
    use_likwid: bool | None = None
    use_line_profiler: bool | None = None
    use_nvtx: bool | None = None
    use_gpu_timing: bool | None = None
    gpu_timing_backend: Any | None = None
    recursive_profile: bool | None = None
    buffer_limit: int | None = None
    file_path: str | os.PathLike | None = None
    capture_region_source: bool | None = None
    config_path: str | os.PathLike | None = None
    verbose: bool = False
    return_results: bool = True
    native_traces: Any | None = None

    def __post_init__(self):
        if self.buffer_limit is not None:
            assert isinstance(self.buffer_limit, int) and self.buffer_limit > 0, (
                "buffer_limit must be a positive integer."
            )

    def session_kwargs(
        self,
        *,
        deactivate_profiling: bool,
        file_path: str | os.PathLike,
        label: str | None,
    ) -> dict:
        """Return keyword arguments for ``ProfileManager.session``."""
        kwargs = {
            "deactivate_profiling": deactivate_profiling,
            "file_path": self.file_path if self.file_path is not None else file_path,
            "label": label,
            "verbose": self.verbose,
            "return_results": self.return_results,
        }
        if self.native_traces is not None:
            kwargs["native_traces"] = self.native_traces
        optional_kwargs = {
            "deactivate_file_output": self.deactivate_file_output,
            "use_likwid": self.use_likwid,
            "use_line_profiler": self.use_line_profiler,
            "use_nvtx": self.use_nvtx,
            "use_gpu_timing": self.use_gpu_timing,
            "gpu_timing_backend": self.gpu_timing_backend,
            "recursive_profile": self.recursive_profile,
            "buffer_limit": self.buffer_limit,
            "capture_region_source": self.capture_region_source,
            "config_path": self.config_path,
        }
        kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
        return kwargs

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    def __repr_all_stacked__(self):
        return __dataclass_repr_all_stacked__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)
