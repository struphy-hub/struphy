from dataclasses import dataclass

import cunumpy as xp

from struphy.io.options import LiteralOptions
from struphy.utils.utils import __dataclass_repr_all_stacked__, __dataclass_repr_no_defaults__, check_option


@dataclass
class LoadingParameters:
    """Configuration for particle (marker) loading strategies and data sources.

    This class encapsulates all parameters needed to initialize particles in simulations,
    including population size, spatial and velocity distributions, loading algorithms, and
    restart/external data sources. Supports multiple loading strategies: Monte-Carlo based
    distributions with customizable moments, regular grid tesselation, specific manually-defined
    markers, and loading from restart or external HDF5 files.

    Parameters
    ----------
    Np : int, optional
        Total number of particles to load into the simulation.

    ppc : int, optional
        Particles per cell to load if a grid is defined. Cell divisions follow ``domain_array``.

    ppb : int, default=10
        Particles per sorting box. Sorting boxes are defined by ``boxes_per_dim``.

    loading : LiteralOptions.OptsLoading, default="pseudo_random"
        Loading algorithm strategy. Options include various Monte-Carlo methods or
        'tesselation' for regular grid positioning.

    seed : int, optional
        Seed for the random number generator for reproducible results.
        If None, no seed is applied.

    moments : tuple, optional
        Mean velocities and temperatures defining the Gaussian velocity distribution.
        If None, automatically computed from the background distribution.

    spatial : LiteralOptions.OptsSpatialLoading, default="uniform"
        Spatial sampling method: 'uniform' samples uniformly in (eta1, eta2) coordinates,
        while 'disc' samples uniformly on the disc image of the coordinate space.

    specific_markers : tuple[tuple], optional
        Manually-defined markers as tuples of phase space coordinates (floats).
        Each tuple represents a single particle's initial state.

    n_quad : int, default=1
        Number of quadrature points used for tesselation-based particle loading.

    set_zero_velocity: tuple
        Initialize velocity of Maxwellain along selected axis to be zero.

    dir_external : str, optional
        Absolute path to HDF5 file from which to load external marker data.

    dir_particles_abs : str, optional
        Absolute path to HDF5 file from which to load restart marker data.

    dir_particles : str, optional
        Relative path (relative to output folder) to HDF5 restart file for loading markers.

    restart_key : str, optional
        HDF5 dataset key within the 'restart/' folder containing marker array data.
    """

    Np: int = None
    ppc: int = None
    ppb: int = 10
    loading: LiteralOptions.OptsLoading = "pseudo_random"
    seed: int = None
    moments: tuple = None
    spatial: LiteralOptions.OptsSpatialLoading = "uniform"
    specific_markers: tuple[tuple] = None
    set_zero_velocity: tuple[bool] = (False, False, False)
    n_quad: int = 1
    dir_exrernal: str = None
    dir_particles: str = None
    dir_particles_abs: str = None
    restart_key: str = None

    def __post_init__(self):
        check_option(self.loading, LiteralOptions.OptsLoading)
        check_option(self.spatial, LiteralOptions.OptsSpatialLoading)

    def __repr_no_defaults__(self):
        return __dataclass_repr_all_stacked__(self)


@dataclass
class WeightsParameters:
    """Configuration for particle weight handling and variance reduction.

    Manages particle weighting strategies used in Monte-Carlo type simulations,
    including control variate techniques for variance reduction and weight thresholding
    to eliminate negligibly-weighted particles.

    Parameters
    ----------
    control_variate : bool, default=False
        Whether to apply control variate variance reduction technique to particle weights.

    reject_weights : bool, default=False
        Whether to filter out particles with weights below the specified threshold.

    threshold : float, default=0.0
        Minimum weight threshold. Particles with weights below this value are rejected
        when ``reject_weights`` is True.
    """

    control_variate: bool = False
    reject_weights: bool = False
    threshold: float = 0.0

    def __repr_no_defaults__(self):
        return __dataclass_repr_all_stacked__(self)


@dataclass
class BoundaryParameters:
    """Configuration for boundary conditions applied to particles and kernel reconstructions.

    Defines how particles behave at domain boundaries (particle boundary conditions) and
    how smoothed particle hydrodynamics (SPH) kernel reconstructions are handled at domain
    edges. Supports independent boundary conditions per spatial dimension.

    Parameters
    ----------
    bc : tuple[LiteralOptions.OptsMarkerBC], default=("periodic", "periodic", "periodic")
        Particle boundary conditions for each spatial direction (3D). Options per direction:
        'remove' (delete particles), 'reflect' (specular reflection), 'periodic' (wrap around),
        or 'refill' (reload particles).

    bc_refill : list, optional
        Refill strategy when 'refill' boundary condition is active. Either 'inner' or 'outer'.

    bc_sph : tuple[LiteralOptions.OptsRecontructBC], default=("periodic", "periodic", "periodic")
        Boundary conditions for SPH kernel reconstruction in each spatial direction.
        Typically matches or differs from ``bc`` depending on reconstruction needs.

    mean_velocity_index : int, optional
        If any boundary condition is 'noslip', this index specifies the position in the marker array
        where the mean velocity for the noslip condition is stored.
    """

    bc: tuple[LiteralOptions.OptsMarkerBC] = ("periodic", "periodic", "periodic")
    bc_refill = None
    bc_sph: tuple[LiteralOptions.OptsRecontructBC] = ("periodic", "periodic", "periodic")
    mean_velocity_index: int | None = None

    def __post_init__(self):
        check_option(self.bc, LiteralOptions.OptsMarkerBC)
        check_option(self.bc_sph, LiteralOptions.OptsRecontructBC)

    def __repr_no_defaults__(self):
        return __dataclass_repr_all_stacked__(self)


@dataclass
class SortingParameters:
    """Configuration for particle sorting strategies and parameters.

    Manages the organization of particles into spatial bins (sorting boxes) to optimize
    neighbor searches and interactions. Supports independent control over the number of
    boxes per dimension and whether to apply MPI domain decomposition along each dimension.

    Parameters
    ----------
    do_sort: bool
        Whether to sort particles in memory.

    sorting_frequency: int
        The number of time steps between two sortings (=0 means no sorting is performed).

    boxes_per_dim: tuple[int]
        Number of boxes in each direction of logical space, (n_eta1, n_eta2, n_eta3).

    box_bufsize : float
        Relative buffer size for box array (default = 0.25).
        A number of 1.0 means that the box array is double the size needed to hold N/n_boxes particles,
        where N is the total number of particles.

    dims_mask: tuple[bool]
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If dims_mask[i]=False, the i-th dimension will not be decomposed.
    """

    do_sort: bool = False
    sorting_frequency: int = 0
    boxes_per_dim: tuple = (12, 12, 1)
    box_bufsize: float = 2.0
    dims_mask: tuple = (True, True, True)

    def __repr_no_defaults__(self):
        return __dataclass_repr_all_stacked__(self)


@dataclass
class BinningPlot:
    """Configuration for particle phase-space binning and histogram generation.

    Produces binned distributions from particle data across specified phase-space coordinates.
    Supports arbitrary combinations of spatial (eta) and velocity (v) coordinates with flexible
    binning resolution and coordinate ranges. Automatically computes bin edges and allocates
    storage for full and delta-f distributions.

    Parameters
    ----------
    slice : str, default="e1"
        Phase-space coordinates to bin, specified as underscore-separated coordinate names
        (e.g., 'e1', 'e1_e2', 'e1_v1'). Valid coordinates: 'e1', 'e2', 'e3' (spatial),
        'v1', 'v2', 'v3' (velocity). Example: 'e1' produces 1D binning over eta1;
        'e1_v1' produces 2D binning over eta1 and v1.

    n_bins : int | tuple[int], default=128
        Number of bins per coordinate. If int, applies to all coordinates; if tuple,
        specifies bins for each coordinate separately.

    ranges : tuple[float] | tuple[tuple[float]], default=(0.0, 1.0)
        Binning ranges as intervals [min, max] for each coordinate. If a single tuple,
        applies to all coordinates; if nested tuples, specifies range for each coordinate.

    divide_by_jac : bool, default=True
        Whether to normalize distributions by the Jacobian determinant (volume-to-0-form
        conversion). Set False to use unnormalized bin counts.

    output_quantity : LiteralOptions.BinningQuantity, default="density"
        Quantity to compute in binning: determines weighting scheme and output format.
    """

    slice: str = "e1"
    n_bins: int | tuple[int] = 128
    ranges: tuple[float] | tuple[tuple[float]] = (0.0, 1.0)
    divide_by_jac: bool = True
    output_quantity: LiteralOptions.BinningQuantity = "density"

    def __post_init__(self):
        if isinstance(self.n_bins, int):
            self.n_bins = (self.n_bins,)

        if not isinstance(self.ranges[0], tuple):
            self.ranges = (self.ranges,)

        assert ((len(self.slice) - 2) / 3).is_integer(), (
            f"Binning coordinates must be separated by '_', but reads {self.slice}."
        )
        assert len(self.slice.split("_")) == len(self.ranges) == len(self.n_bins), (
            f"Number of slices names ({len(self.slice.split('_'))}), number of bins ({len(self.n_bins)}), and number of ranges ({len(self.ranges)}) are inconsistent with each other!\n\n"
        )
        check_option(self.output_quantity, LiteralOptions.BinningQuantity)

        # computations and allocations
        self._bin_edges = []
        for nb, rng in zip(self.n_bins, self.ranges):
            self._bin_edges += [xp.linspace(rng[0], rng[1], nb + 1)]
        self._bin_edges = tuple(self.bin_edges)

        self._f = xp.zeros(self.n_bins, dtype=float)
        self._df = xp.zeros(self.n_bins, dtype=float)

    @property
    def bin_edges(self) -> tuple:
        return self._bin_edges

    @property
    def f(self) -> xp.ndarray:
        """The binned distribution function (full-f)."""
        return self._f

    @property
    def df(self) -> xp.ndarray:
        """The binned distribution function minus the background (delta-f)."""
        return self._df

    def __repr__(self):
        return __dataclass_repr_no_defaults__(self)


@dataclass
class KernelDensityPlot:
    """Configuration for smoothed particle hydrodynamics (SPH) density reconstructions.

    Evaluates particle density fields at structured grid points using SPH kernel
    interpolation. Supports 1D, 2D, and 3D evaluations with independent resolution
    control per dimension.

    Parameters
    ----------
    pts_e1 : int, default=16
        Number of evaluation grid points in the first spatial direction (eta1).

    pts_e2 : int, default=16
        Number of evaluation grid points in the second spatial direction (eta2).

    pts_e3 : int, default=1
        Number of evaluation grid points in the third spatial direction (eta3).
        Set to 1 for 2D density plots.
    """

    pts_e1: int = 16
    pts_e2: int = 16
    pts_e3: int = 1

    def __post_init__(self):
        e1 = xp.linspace(0.0, 1.0, self.pts_e1)
        e2 = xp.linspace(0.0, 1.0, self.pts_e2)
        e3 = xp.linspace(0.0, 1.0, self.pts_e3)
        ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")
        self._plot_pts = (ee1, ee2, ee3)
        self._n_sph = xp.zeros(ee1.shape, dtype=float)

    @property
    def plot_pts(self) -> tuple:
        return self._plot_pts

    @property
    def n_sph(self) -> xp.ndarray:
        """The evaluated density."""
        return self._n_sph


@dataclass
class SavingParameters:
    """Configuration for particle data saving strategies and parameters.

    Manages how particle data is saved during simulations, including the frequency of saves,
    the types of data to save (single markers, full distribution vs. delta-f or kernel evaluations),
    and the output formats. Supports flexible control over what particle information is stored and when.

    Parameters
    ----------
    n_markers: int | float
        Number of particles/markers for which to save trajectories.
        If float and <1.0, then understood as the fraction of the total number of markers.

    binned_data: tuple[BinningPlot]
        A tuple of BinningPlot objects.

    kernel_density_plots: tuple[KernelDensityPlot]
        A tuple of KernelDensityPlot objects.
    """

    n_markers: int | float = 3
    binning_plots: tuple[BinningPlot] = ()
    kernel_density_plots: tuple[KernelDensityPlot] = ()

    def __repr_no_defaults__(self):
        return __dataclass_repr_all_stacked__(self)
