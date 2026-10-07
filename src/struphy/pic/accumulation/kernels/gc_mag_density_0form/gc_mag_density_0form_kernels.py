"Accumulation kernel for gyro-center (5D) particles."

from numpy import shape

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def gc_mag_density_0form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
    scale: "float",  # model specific argument
):
    r"""
    Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V0 with the filling

    .. math::

        B_p^\mu = \mu \frac{w_p}{N} \,.
    """

    markers = args_markers.markers
    mu_idx = args_markers.mu_idx

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, filling)
    # -- removed omp: #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and magnetic moment
        weight = markers[ip, 5]
        mu = markers[ip, mu_idx]

        # filling =mu*w_p/N
        filling = mu * weight * scale

        particle_to_mat_kernels.vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, filling)
