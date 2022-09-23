import numpy as np

import struphy.pic.utilities_kernels as utils

def eval_field_at_particles(fe_coeffs, derham, space_id, particles):
    """
    TODO

    Parameters
    ----------
        fe_coeffs : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
            FE coefficients of a form.

        derham : struphy.psydac_api.psydac_derham.Derham
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
                                             np.array(derham.V0.vector_space.starts),
                                             fe_coeffs._data)

    elif space_id == 'Hcurl':
        res = np.empty(3, dtype=float)
        utils.eval_1_form_at_particles(particles.markers,
                                       np.array(derham.p), 
                                       derham.V1.knots[0], derham.V1.knots[1], derham.V1.knots[2],
                                       np.array(derham.V1.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    elif space_id == 'Hdiv':
        res = np.empty(3, dtype=float)
        utils.eval_2_form_at_particles(particles.markers,
                                       np.array(derham.p), 
                                       derham.V2.knots[0], derham.V2.knots[1], derham.V2.knots[2],
                                       np.array(derham.V2.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    elif space_id == 'L2':
        res = utils.eval_0_form_at_particles(particles.markers,
                                             np.array(derham.p), 
                                             derham.V3.knots[0], derham.V3.knots[1], derham.V3.knots[2],
                                             np.array(derham.V3.vector_space.starts),
                                             fe_coeffs)
    elif space_id == 'H1vec':
        res = np.empty(3, dtype=float)
        utils.eval_2_form_at_particles(particles.markers,
                                       np.array(derham.p), 
                                       derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                                       np.array(derham.V0.vector_space.starts),
                                       fe_coeffs.blocks[0]._data, fe_coeffs.blocks[1]._data, fe_coeffs.blocks[2]._data,
                                       res)

    else:
        raise NotImplementedError(f'The space {space_id} is not implemented!')
    
    return res
