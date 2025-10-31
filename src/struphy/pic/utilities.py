import struphy.pic.utilities_kernels as utils
from struphy.utils.arrays import xp as np


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
