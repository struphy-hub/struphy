"Pusher kernel for full orbit (6D) particles."

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("ginv", "k", "tmp", "pi_du_value")
def push_deterministic_diffusion_stage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    pi_u: "float[:,:,:]",
    pi_grad_u1: "float[:,:,:]",
    pi_grad_u2: "float[:,:,:]",
    pi_grad_u3: "float[:,:,:]",
    diffusion_coeff: float,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = - D \,G^{-1}(\boldsymbol \eta_p(t)) \frac{\nabla \hat u^0}{\hat u^0}(\boldsymbol \eta_p(t))

    for each marker :math:`p` in markers array, where :math:`\frac{\nabla \hat u^0}{\hat u^0}` is constant in time. :math:`D>0` is a positive, constant diffusion coefficient.
    """

    # allocate arrays
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    tmp3 = zeros((3, 3), dtype=float)
    ginv = zeros((3, 3), dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    pi_du_value = empty(3, dtype=float)

    # fmt: off
    #$ omp parallel private(ip, e1, e2, e3, span1, span2, span3, pi_u_value, pi_du_value, k, tmp, tmp1, tmp2, tmp3, ginv)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # density function: 0-form components
        pi_u_value = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            pi_u,
        )

        # gradient of the density function: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            pi_grad_u1,
            pi_grad_u2,
            pi_grad_u3,
            pi_du_value,
        )

        # evaluate Metric tensor, result in gm
        evaluation_kernels.g_inv(
            e1,
            e2,
            e3,
            args_domain,
            tmp1,
            tmp2,
            tmp3,
            False,
            ginv,
        )

        # updating k
        tmp = -diffusion_coeff * pi_du_value / pi_u_value
        linalg_kernels.matrix_vector(ginv, tmp, k)

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
