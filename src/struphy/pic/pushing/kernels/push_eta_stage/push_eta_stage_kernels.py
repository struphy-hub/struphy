"Pusher kernel for full orbit (6D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "v", "k")
def push_eta_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
    n_stages: int,
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_pusher_idx = args_markers.first_pusher_idx

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    # fmt: off
    #$ omp parallel private(dfinv, dfm, e1, e2, e3, ip, k, v)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # check if marker is a hole or a boundary particle
        if markers[first_pusher_idx, ip] == -1.0 or markers[-1, ip] == -2.0:
            continue

        e1 = markers[0, ip]
        e2 = markers[1, ip]
        e3 = markers[2, ip]
        v[:] = markers[3:6, ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k)

        # accumulation for last stage and update of positions (regular at a pole)
        pusher_utilities_kernels.update_position_stage(ip, k, dt, a[stage], b[stage], last, args_markers)

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)

    # fmt: off
    #$ omp end parallel
    # fmt: on
