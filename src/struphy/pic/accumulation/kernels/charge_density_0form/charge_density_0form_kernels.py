"Accumulation kernel for full-orbit (6D) particles."

from numpy import shape

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def charge_density_0form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
):
    r"""
    Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V0 with filling function

    .. math::

        B_p = w_p \,,

    where :math:`w_p` is the marker weight.
    """

    markers = args_markers.markers
    weight_idx = args_markers.weight_idx

    # fmt: off
    #$ omp parallel private(eta1, eta2, eta3, filling, ip, thread_derham)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for reduction ( + :vec)
    # fmt: on
    for ip in range(args_markers.n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # marker positions
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # filling is just the weights
        filling = markers[weight_idx, ip]

        particle_to_mat_kernels.vec_fill_b_v0(
            thread_derham,
            eta1,
            eta2,
            eta3,
            vec,
            filling,
        )

    # fmt: off
    #$ omp end parallel
    # fmt: on
