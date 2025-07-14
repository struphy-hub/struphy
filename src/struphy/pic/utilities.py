import numpy as np

import struphy.pic.utilities_kernels as utils
from struphy.geometry.base import Domain


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


def amrex_reflect(markers_array: "dict", outside_inds: "int[:]", axis: "int", domain: "Domain"):
    r"""
    Reflect the particles which are pushed outside of the logical cube.

    .. math::

        \hat{v} = DF^{-1} v \,, \\
        \hat{v}_\text{reflected}[\text{axis}] = -1 * \hat{v} \,, \\
        v_\text{reflected} = DF \hat{v}_\text{reflected} \,.

    Parameters
    ----------
        markers : array[float]
            Local markers array

        outside_inds : array[int]
            inds indicate the particles which are pushed outside of the local cube

        axis : int
            0, 1 or 2
    """

    e1 = markers_array["x"][outside_inds]
    e2 = markers_array["y"][outside_inds]
    e3 = markers_array["z"][outside_inds]
    v1 = markers_array["v1"][outside_inds]
    v2 = markers_array["v2"][outside_inds]
    v3 = markers_array["v3"][outside_inds]

    # evaluate inverse Jacobian matrices for each point
    jacobian = domain.jacobian(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3
    jacobian_inv = domain.jacobian_inv(e1, e2, e3, change_out_order=True, flat_eval=True)  # Npx3x3

    # pull-back of velocity
    v = np.array([v1, v2, v3]).T
    v = v[..., np.newaxis]  # Npx3x1
    # If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. Squeeze to take away the unnecessary 1 dim
    v_logical = np.matmul(jacobian_inv, v)

    # reverse the velocity
    v_logical[:, axis] *= -1

    # push forward of the velocity
    v = np.matmul(jacobian, v_logical)

    # update the particle velocities
    markers_array["v1"][outside_inds] = v[:, 0, 0]
    markers_array["v2"][outside_inds] = v[:, 1, 0]
    markers_array["v3"][outside_inds] = v[:, 2, 0]
