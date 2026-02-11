import cunumpy as xp

from struphy.io.options import LiteralOptions
from struphy.utils.utils import check_option


class LoadingParameters:
    """Options for particle loading in parameter/launch files.

    Parameters
    ----------
    Np : int
        Total number of particles to load.

    ppc : int
        Particles to load per cell if a grid is defined. Cells are defined from ``domain_array``.

    ppb : int
        Particles to load per sorting box. Sorting boxes are defined from ``boxes_per_dim``.

    loading : LiteralOptions.OptsLoading
        How to load markers: multiple options for Monte-Carlo, or "tesselation" for positioning them on a regular grid.

    seed : int
        Seed for random generator. If None, no seed is taken.

    moments : tuple
        Mean velocities and temperatures for the Gaussian sampling distribution.
        If None, these are auto-calculated form the given background.

    spatial : LiteralOptions.OptsSpatialLoading
        Draw uniformly in eta, or draw uniformly on the "disc" image of (eta1, eta2).

    specific_markers : tuple[tuple]
        Each entry is a tuple of phase space coordinates (floats) of a specific marker to be initialized.
    
    set_zero_velocity: tuple
        Initialize velocity of Maxwellain along selected axis to be zero.

    n_quad : int
        Number of quadrature points for tesselation.

    dir_external : str
        Load markers from external .hdf5 file (absolute path).

    dir_particles_abs : str
        Load markers from restart .hdf5 file (absolute path).

    dir_particles : str
        Load markers from restart .hdf5 file (relative path to output folder).

    restart_key : str
        Key in .hdf5 file's restart/ folder where marker array is stored.
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
        set_zero_velocity: tuple[bool] = (False, False, False),
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
        self.set_zero_velocity = set_zero_velocity
        self.n_quad = n_quad
        self.dir_external = dir_exrernal
        self.dir_particles = dir_particles
        self.dir_particles_abs = dir_particles_abs
        self.restart_key = restart_key


class WeightsParameters:
    """Options for particle weights in parameter/launch files.

    Parameters
    ----------
    control_variate : bool
        Whether to use a control variate for noise reduction.

    reject_weights : bool
        Whether to reject weights below threshold.

    threshold : float
        Threshold for rejecting weights.
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
    """Options for particle boundary conditions and SPH-reconstruction boundary conditions in parameter/launch files.

    Parameters
    ----------
    bc : tuple[LiteralOptions.OptsMarkerBC]
        Boundary conditions for particle movement.
        Either 'remove', 'reflect', 'periodic' or 'refill' in each direction.

    bc_refill : list
        Either 'inner' or 'outer'.

    bc_sph : tuple[LiteralOptions.OptsRecontructBC]
        Boundary conditions for sph kernel reconstruction.
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
    """Options for particle binning (plots) in parameter/launch files.

    Parameters
    ----------
    slice : str
        Coordinate-slice in phase space to bin. A combination of "e1", "e2", "e3", "v1", etc., separated by an underscore "_".
        For example, "e1" showas a 1D binning plot over eta1, whereas "e1_v1" shows a 2D binning plot over eta1 and v1.

    n_bins : int | tuple[int]
        Number of bins for each coordinate.

    ranges : tuple[int] | tuple[tuple[int]] = (0.0, 1.0)
        Binning range (as an interval in R) for each coordinate.

    divide_by_jac : bool
        Whether to divide by the Jacobian determinant (volume-to-0-form).

    output_quantity : BinningOutput
        String literal used to determine weights in binning and the type of output
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
    """Options for SPH density plots in parameter/launch files.

    Parameters
    ----------
    pts_e1, pts_e2, pts_e3 : int
        Number of evaluation points in each direction.
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
