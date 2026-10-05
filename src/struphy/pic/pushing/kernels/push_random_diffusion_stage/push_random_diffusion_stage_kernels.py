"Pusher kernel for full orbit (6D) particles."

from numpy import sqrt

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


def push_random_diffusion_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    noise: "float[:,:]",
    diffusion_coeff: float,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        {\textnormal d \boldsymbol \eta_p(t)} = \sqrt{2 \, D}\, \textnormal d \boldsymbol B_t\,,

    for each marker :math:`p` in markers array, where :math:`\textnormal d \boldsymbol B_t` is a Brownian Motion and $D$ is a positive diffusion coefficient.

    The right-hand side is the same Wiener increment
    :math:`\sqrt{2 D \Delta t}\, \boldsymbol \xi_p` in every stage, hence stage ``i``
    adds the fraction ``b[i]`` of it. Since :math:`\sum_i b_i = 1`, one full step adds the
    increment exactly once for any explicit tableau (Euler-Maruyama).
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # fraction of the Wiener increment added in this stage
    scale = b[stage] * sqrt(2 * dt * diffusion_coeff)

    # fmt: off
    #$ omp parallel private(ip)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        markers[ip, 0:3] += scale * noise[ip, :]

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
