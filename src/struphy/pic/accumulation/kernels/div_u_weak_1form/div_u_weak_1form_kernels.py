"Accumulation kernel for full-orbit (6D) particles."

from numpy import shape

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def div_u_weak_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
):
    r"""
    Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V1 with filling function

    .. math::

        \mathbf{B}_p = \frac{w_p}{n_p} \mathbf{v}_p \,,

    where :math:`w_p` is the marker weight, :math:`\mathbf{v}_p` the marker velocity
    and :math:`n_p` the (previously accumulated) density evaluated at the marker position,
    stored in column ``args_markers.first_free_idx`` of the markers array.
    """

    markers = args_markers.markers
    weight_idx = args_markers.weight_idx
    density_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, filling)
    # -- removed omp: #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if not valid_mks[ip]:
            continue

        # marker positions and velocites
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v1 = markers[ip, 3]
        v2 = markers[ip, 4]
        v3 = markers[ip, 5]

        # weight and density
        weight = markers[ip, weight_idx]
        density = markers[ip, density_idx]

        # filling is just the weights
        fill1 = weight * v1 / density
        fill2 = weight * v2 / density
        fill3 = weight * v3 / density

        particle_to_mat_kernels.vec_fill_b_v1(
            args_derham,
            eta1,
            eta2,
            eta3,
            vec1,
            vec2,
            vec3,
            fill1,
            fill2,
            fill3,
        )
