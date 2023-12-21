import numpy as np

import struphy.pic.utilities_kernels as utils


def eval_field_at_particles(fe_coeffs, derham, space_id, particles):
    """
    TODO

    Parameters
    ----------
        fe_coeffs : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
            FE coefficients of a form.

        derham : struphy.feec.psydac_derham.Derham
            Discrete Derham complex.

        space_id : str
            one of "H1", "Hcurl", "Hdiv", "L2", or "H1vec"

        particles : struphy.pic.particles.Particles6D or struphy.pic.particles.Particles5D
            Particles object.
    """

    if space_id == 'H1':
        res = utils.eval_0_form_at_particles(particles.markers,
                                             np.array(derham.p),
                                             derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                                             np.array(
                                                 derham.V0.vector_space.starts),
                                             fe_coeffs._data)

    elif space_id == 'Hcurl':
        res = np.empty(3, dtype=float)
        utils.eval_1_form_at_particles(particles.markers,
                                       np.array(derham.p),
                                       derham.V1.knots[0], derham.V1.knots[1], derham.V1.knots[2],
                                       np.array(
                                                 derham.V0.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    elif space_id == 'Hdiv':
        res = np.empty(3, dtype=float)
        utils.eval_2_form_at_particles(particles.markers,
                                       np.array(derham.p),
                                       derham.V2.knots[0], derham.V2.knots[1], derham.V2.knots[2],
                                       np.array(
                                                 derham.V0.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    elif space_id == 'L2':
        res = utils.eval_0_form_at_particles(particles.markers,
                                             np.array(derham.p),
                                             derham.V3.knots[0], derham.V3.knots[1], derham.V3.knots[2],
                                             np.array(
                                                 derham.V0.vector_space.starts),
                                             fe_coeffs)
    elif space_id == 'H1vec':
        res = np.empty(3, dtype=float)
        utils.eval_H1vec_at_particles(particles.markers,
                                       np.array(derham.p),
                                       derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                                       np.array(
                                                 derham.V0.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    else:
        raise NotImplementedError(f'The space {space_id} is not implemented!')

    return res


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
    utils.canonical_kinetic_particles(res, particles.markers,
                                      np.array(derham.p),
                                      derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                                      np.array(
                                                 derham.V0.vector_space.starts),
                                      *domain.args_map,
                                      fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data)

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
    utils.thermal_energy(res, density_0_form._operators[0].matrix._data,
                         derham.Vh_fem['0'].vector_space.pads[0],
                         derham.Vh_fem['0'].vector_space.pads[1],
                         derham.Vh_fem['0'].vector_space.pads[2],
                         nel1, nel2, nel3,
                         nqs1, nqs2, nqs3,
                         derham.Vh_fem['0']._quad_grids[0].weights,
                         derham.Vh_fem['0']._quad_grids[1].weights,
                         derham.Vh_fem['0']._quad_grids[2].weights,
                         derham.Vh_fem['0']._quad_grids[0].points,
                         derham.Vh_fem['0']._quad_grids[1].points,
                         derham.Vh_fem['0']._quad_grids[2].points,
                         *domain.args_map)

    return res
