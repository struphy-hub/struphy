import struphy.pic.utilities_kernels as utils
from struphy.io.options import OptsLoading, OptsSpatialLoading


class LoadingParameters:
    """Paramters for particle loading.
    
    Parameters
    ----------
    loading : OptsLoading
        How to load markers: "pseudo_random" for Monte-Carlo, or "tesselation" for positioning them on a regular grid.
    
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
    """
    def __init__(self,
                 loading: OptsLoading = "pseudo_random",
                 seed: int = None,
                 moments: tuple = None,
                 spatial: OptsSpatialLoading = "uniform",
                 specific_markers: tuple[tuple] = None,
                 n_quad: int = 1,
                 dir_exrernal: str = None,
                 dir_particles: str = None,
                 dir_particles_abs: str = None,
                 ):

        self.loading = loading
        self.seed = seed
        self.moments = moments
        self.spatial = spatial
        self.specific_markers = specific_markers
        self.n_quad = n_quad
        self.dir_external = dir_exrernal
        self.dir_particles = dir_particles
        self.dir_particles_abs = dir_particles_abs
        
        
class WeightsParameters:
    """Paramters for particle weights.
    
    Parameters
    ----------
    control_variate : bool
        Whether to use a control variate for noise reduction.
    
    rejct_weights : bool
        Whether to reject weights below threshold.
        
    threshold : float
        Threshold for rejecting weights.
    """
    def __init__(self,
                control_variate: bool = False,
                reject_weights: bool = False,
                threshold: float = 0.0,):
        
        self.control_variate = control_variate
        self.reject_weights = reject_weights
        self.threshold = threshold


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
