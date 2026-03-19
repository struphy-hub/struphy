import cunumpy as xp

import struphy.pic.utilities_kernels as utils
from struphy.feec.utilities import get_quad_grids


def get_kinetic_energy_particles(fe_coeffs, derham, domain, particles):
    """
    This function is for getting kinetic energy of the case when canonical momentum is used, rather than velocity

    Parameters
    ----------
        fe_coeffs : feectools.linalg.stencil.StencilVector or feectools.linalg.block.BlockVector
            FE coefficients of 1 form, i.e., vector potential.

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.
    """

    res = xp.empty(1, dtype=float)
    utils.canonical_kinetic_particles(
        res,
        particles.markers,
        xp.array(derham.p),
        derham.V0fem.knots[0],
        derham.V0fem.knots[1],
        derham.V0fem.knots[2],
        xp.array(
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
        density_0_form : feectools.linalg.stencil.StencilVector
            values of density at quadrature points, 3-form.

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.
    """

    res = xp.empty(1, dtype=float)
    utils.thermal_energy(
        res,
        density_0_form._operators[0].matrix._data,
        derham.V0fem.coeff_space.pads[0],
        derham.V0fem.coeff_space.pads[1],
        derham.V0fem.coeff_space.pads[2],
        nel1,
        nel2,
        nel3,
        nqs1,
        nqs2,
        nqs3,
        get_quad_grids(derham.V0fem)[0].weights,
        get_quad_grids(derham.V0fem)[1].weights,
        get_quad_grids(derham.V0fem)[2].weights,
        get_quad_grids(derham.V0fem)[0].points,
        get_quad_grids(derham.V0fem)[1].points,
        get_quad_grids(derham.V0fem)[2].points,
        *domain.args_map,
    )

    return res
