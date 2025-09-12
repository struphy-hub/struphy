import numpy as np

import struphy.pic.utilities_kernels as utils
from struphy.io.options import (
    OptsLoading,
    OptsMarkerBC,
    OptsRecontructBC,
    OptsSpatialLoading,
)


class LoadingParameters:
    """Parameters for particle loading.

    Parameters
    ----------
    Np : int
        Total number of particles to load.

    ppc : int
        Particles to load per cell if a grid is defined. Cells are defined from ``domain_array``.

    ppb : int
        Particles to load per sorting box. Sorting boxes are defined from ``boxes_per_dim``.

    loading : OptsLoading
        How to load markers: multiple options for Monte-Carlo, or "tesselation" for positioning them on a regular grid.

    seed : int
        Seed for random generator. If None, no seed is taken.

    moments : tuple
        Mean velocities and temperatures for the Gaussian sampling distribution.
        If None, these are auto-calculated form the given background.

    spatial : OptsSpatialLoading
        Draw uniformly in eta, or draw uniformly on the "disc" image of (eta1, eta2).

    specific_markers : tuple[tuple]
        Each entry is a tuple of phase space coordinates (floats) of a specific marker to be initialized.

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
    def __init__(self,
                 Np: int = None,
                 ppc: int = None,
                 ppb: int = 10,
                 loading: OptsLoading = "pseudo_random",
                 seed: int = None,
                 moments: tuple = None,
                 spatial: OptsSpatialLoading = "uniform",
                 specific_markers: tuple[tuple] = None,
                 n_quad: int = 1,
                 dir_exrernal: str = None,
                 dir_particles: str = None,
                 dir_particles_abs: str = None,
                 restart_key: str = None,
                 ):

    def __init__(
        self,
        Np: int = None,
        ppc: int = None,
        ppb: int = 10,
        loading: OptsLoading = "pseudo_random",
        seed: int = None,
        moments: tuple = None,
        spatial: OptsSpatialLoading = "uniform",
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
    """Paramters for particle weights.

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
    """Parameters for particle boundary and sph reconstruction boundary conditions.

    Parameters
    ----------
    bc : tuple[OptsMarkerBC]
        Boundary conditions for particle movement.
        Either 'remove', 'reflect', 'periodic' or 'refill' in each direction.

    bc_refill : list
        Either 'inner' or 'outer'.

    bc_sph : tuple[OptsRecontructBC]
        Boundary conditions for sph kernel reconstruction.
    """

    def __init__(
        self,
        bc: tuple[OptsMarkerBC] = ("periodic", "periodic", "periodic"),
        bc_refill=None,
        bc_sph: tuple[OptsRecontructBC] = ("periodic", "periodic", "periodic"),
    ):
        self.bc = bc
        self.bc_refill = bc_refill
        self.bc_sph = bc_sph


class BinningPlot:
    """Binning plot of marker distribution in phase space.

    Parameters
    ----------
    slice : str
        Coordinate-slice in phase space to bin. A combination of "e1", "e2", "e3", "v1", etc., separated by an underscore "_".
        For example, "e1" showas a 1D binning plot over eta1, whereas "e1_v1" shows a 2D binning plot over eta1 and v1.

    n_bins : int | tuple[int]
        Number of bins for each coordinate.

    ranges : tuple[int] | tuple[tuple[int]]= (0.0, 1.0)
        Binning range (as an interval in R) for each coordinate.
    """

    def __init__(
        self,
        slice: str = "e1",
        n_bins: int | tuple[int] = 128,
        ranges: tuple[float] | tuple[tuple[float]] = (0.0, 1.0),
    ):
        self.slice = slice

        if isinstance(n_bins, int):
            n_bins = (n_bins,)
        self.n_bins = n_bins

        if not isinstance(ranges[0], tuple):
            ranges = (ranges,)
        self.ranges = ranges

class BinningPlot:
    """Binning plot of marker distribution in phase space.
        
    Parameters
    ----------
    slice : str
        Coordinate-slice in phase space to bin. A combination of "e1", "e2", "e3", "v1", etc., separated by an underscore "_".
        For example, "e1" showas a 1D binning plot over eta1, whereas "e1_v1" shows a 2D binning plot over eta1 and v1.
    
    n_bins : int | tuple[int]
        Number of bins for each coordinate.
    
    ranges : tuple[int] | tuple[tuple[int]]= (0.0, 1.0)
        Binning range (as an interval in R) for each coordinate.
    """
    def __init__(self, 
                 slice: str = "e1", 
                 n_bins: int | tuple[int] = 128, 
                 ranges: tuple[float] | tuple[tuple[float]]= (0.0, 1.0),):
        self.slice = slice
        
        if isinstance(n_bins, int):
            n_bins = (n_bins,) 
        self.n_bins = n_bins
        
        if not isinstance(ranges[0], tuple):
            ranges = (ranges,)
        self.ranges = ranges


def get_kinetic_energy_particles(fe_coeffs, derham, domain, particles):
    """
    This function is for getting kinetic energy of the case when canonical momentum is used, rather than velocity

    Parameters
    ----------
        fe_coeffs : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
            FE coefficients of 1 form, i.e., vector potential.

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.
    """

    res = np.empty(1, dtype=float)
    utils.canonical_kinetic_particles(
        res,
        particles.markers,
        np.array(derham.p),
        derham.Vh_fem["0"].knots[0],
        derham.Vh_fem["0"].knots[1],
        derham.Vh_fem["0"].knots[2],
        np.array(
            derham.V0.coeff_space.starts,
        ),
        *domain.args_map,
        fe_coeffs.blocks[0]._data,
        fe_coeffs.blocks[1]._data,
        fe_coeffs.blocks[2]._data,
    )

    return res


def get_electron_thermal_energy(density_0_form, derham, domain, nel1, nel2, nel3, nqs1, nqs2, nqs3):
    """
    This function is for getting kinetic energy of the case when canonical momentum is used, rather than velocity

    Parameters
    ----------
        density_0_form : psydac.linalg.stencil.StencilVector
            values of density at quadrature points, 3-form.

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.
    """

    res = np.empty(1, dtype=float)
    utils.thermal_energy(
        res,
        density_0_form._operators[0].matrix._data,
        derham.Vh_fem["0"].coeff_space.pads[0],
        derham.Vh_fem["0"].coeff_space.pads[1],
        derham.Vh_fem["0"].coeff_space.pads[2],
        nel1,
        nel2,
        nel3,
        nqs1,
        nqs2,
        nqs3,
        derham.get_quad_grids(derham.Vh_fem["0"])[0].weights,
        derham.get_quad_grids(derham.Vh_fem["0"])[1].weights,
        derham.get_quad_grids(derham.Vh_fem["0"])[2].weights,
        derham.get_quad_grids(derham.Vh_fem["0"])[0].points,
        derham.get_quad_grids(derham.Vh_fem["0"])[1].points,
        derham.get_quad_grids(derham.Vh_fem["0"])[2].points,
        *domain.args_map,
    )

    return res
