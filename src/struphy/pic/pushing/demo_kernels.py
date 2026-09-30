"""Minimal pusher kernel with a 1:1 CUDA counterpart in :mod:`struphy.pic.pushing.demo_cuda`,
used to test the pyccel/CUDA kernel dispatch in :mod:`struphy.utils.kernel_backends`."""

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


def push_eta_linear(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
):
    r"""Explicit Euler step of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \mathbf v_p

    for each valid marker :math:`p`, where :math:`\mathbf v_p` is constant (no mapping, no boundary conditions).
    """

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        markers[ip, 0] += dt * markers[ip, 3]
        markers[ip, 1] += dt * markers[ip, 4]
        markers[ip, 2] += dt * markers[ip, 5]
