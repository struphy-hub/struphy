"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "unit_b1", "e_star", "e_field", "Exb", "k")
def push_gc_bxEstar_explicit_multistage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of an s-stage explicit Runge-Kutta scheme for solving

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\boldsymbol \eta_p(t)) \,,

    where

    .. math::

        \hat{\mathbf E}^{*1} = - \hat \nabla \hat \phi -\varepsilon \mu_p \hat \nabla  \hat B\,,\qquad \hat B^*_\parallel = \hat B + \varepsilon v_{\parallel,p} \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}\,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    unit_b1 = empty(3, dtype=float)
    e_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    Exb = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_pusher_idx = args_markers.first_pusher_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_pusher_idx] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            unit_b1_1,
            unit_b1_2,
            unit_b1_3,
            unit_b1,
        )

        # compute E*
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            e_star,
        )

        e_star *= -epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )
            e_star += e_field

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v
        b_star_parallel += B_dot_b
        b_star_parallel *= det_df

        # calculate k
        linalg_kernels.cross(e_star, unit_b1, Exb)

        k[:] = Exb / b_star_parallel

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_pusher_idx : first_pusher_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
