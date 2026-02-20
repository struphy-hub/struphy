import cunumpy as xp

from struphy.io.options import LiteralOptions
from struphy.utils.utils import check_option


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

    dir_external : str, optional
        Absolute path to HDF5 file from which to load external marker data.

    dir_particles_abs : str, optional
        Absolute path to HDF5 file from which to load restart marker data.

    dir_particles : str, optional
        Relative path (relative to output folder) to HDF5 restart file for loading markers.

    restart_key : str, optional
        HDF5 dataset key within the 'restart/' folder containing marker array data.
    """

    def __init__(
        self,
        Np: int = None,
        ppc: int = None,
        ppb: int = 10,
        loading: LiteralOptions.OptsLoading = "pseudo_random",
        seed: int = None,
        moments: tuple = None,
        spatial: LiteralOptions.OptsSpatialLoading = "uniform",
        specific_markers: tuple[tuple] = None,
        n_quad: int = 1,
        dir_exrernal: str = None,
        dir_particles: str = None,
        dir_particles_abs: str = None,
        restart_key: str = None,
    ):
        self.Np = Np
        self.ppc = ppc
        self.ppb = ppb
        self.loading = loading
        self.seed = seed
        self.moments = moments
        self.spatial = spatial
        self.specific_markers = specific_markers
        self.n_quad = n_quad
        self.dir_external = dir_exrernal
        self.dir_particles = dir_particles
        self.dir_particles_abs = dir_particles_abs
        self.restart_key = restart_key


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

    def __init__(
        self,
        control_variate: bool = False,
        reject_weights: bool = False,
        threshold: float = 0.0,
    ):
        self.control_variate = control_variate
        self.reject_weights = reject_weights
        self.threshold = threshold


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
    """

    def __init__(
        self,
        bc: tuple[LiteralOptions.OptsMarkerBC] = ("periodic", "periodic", "periodic"),
        bc_refill=None,
        bc_sph: tuple[LiteralOptions.OptsRecontructBC] = ("periodic", "periodic", "periodic"),
    ):
        self.bc = bc
        self.bc_refill = bc_refill
        self.bc_sph = bc_sph


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

    def __init__(
        self,
        slice: str = "e1",
        n_bins: int | tuple[int] = 128,
        ranges: tuple[float] | tuple[tuple[float]] = (0.0, 1.0),
        divide_by_jac: bool = True,
        output_quantity: LiteralOptions.BinningQuantity = "density",
    ):
        if isinstance(n_bins, int):
            n_bins = (n_bins,)

        if not isinstance(ranges[0], tuple):
            ranges = (ranges,)

        assert ((len(slice) - 2) / 3).is_integer(), f"Binning coordinates must be separated by '_', but reads {slice}."
        assert len(slice.split("_")) == len(ranges) == len(n_bins), (
            f"Number of slices names ({len(slice.split('_'))}), number of bins ({len(n_bins)}), and number of ranges ({len(ranges)}) are inconsistent with each other!\n\n"
        )
        check_option(output_quantity, LiteralOptions.BinningQuantity)

        self.slice = slice
        self.n_bins = n_bins
        self.ranges = ranges
        self.divide_by_jac = divide_by_jac
        self.output_quantity = output_quantity

        # computations and allocations
        self._bin_edges = []
        for nb, rng in zip(n_bins, ranges):
            self._bin_edges += [xp.linspace(rng[0], rng[1], nb + 1)]
        self._bin_edges = tuple(self.bin_edges)

        self._f = xp.zeros(n_bins, dtype=float)
        self._df = xp.zeros(n_bins, dtype=float)

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

    def __init__(
        self,
        pts_e1: int = 16,
        pts_e2: int = 16,
        pts_e3: int = 1,
    ):
        e1 = xp.linspace(0.0, 1.0, pts_e1)
        e2 = xp.linspace(0.0, 1.0, pts_e2)
        e3 = xp.linspace(0.0, 1.0, pts_e3)
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
