"Pusher kernel for gyro-center (5D) dynamics, magnetic field as a 1-form."

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "dfinvt", "e_star", "b1", "tmp", "b_star", "k")
def push_gc_Bstar_explicit_multistage_1form(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
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

        \left\{
            \begin{aligned}
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*\textnormal{v}}}{\hat B^{*}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*\textnormal{v}}}{\hat B^{*}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,,
            \end{aligned}
        \right.

    where the magnetic field is given as a 1-form :math:`\hat{\mathbf B}^1` and

    .. math::

        \hat{\mathbf E}^{*1} = - \hat \nabla \hat \phi - \varepsilon \mu_p \hat \nabla \hat B\,,\qquad \hat{\mathbf B}^{*\textnormal{v}} = G^{-1} \hat{\mathbf B}^1 + \frac{\varepsilon v_\parallel}{\sqrt g} \hat \nabla \times \hat{\mathbf b}^1_0\,,\qquad  \hat B^*_\parallel = \hat B + \varepsilon v_{\parallel,p} \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}\,,

    for each marker :math:`p` in markers array. This is
    :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_explicit_multistage.push_gc_Bstar_explicit_multistage_kernels.push_gc_Bstar_explicit_multistage`
    with :math:`\hat{\mathbf B}^{*2} / \sqrt g = \hat{\mathbf B}^{*\textnormal{v}}`: the factor :math:`\sqrt g` cancels in
    the magnetic-field term (:math:`\hat{\mathbf B}^2 = \sqrt g \, G^{-1} \hat{\mathbf B}^1`).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinvt = empty((3, 3), dtype=float)

    # containers for fields
    e_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    b1 = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

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
        if markers[first_pusher_idx, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3, ip]
        mu = markers[mu_idx, ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

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

        # compute B* (vector-field components): G^{-1} b1 + epsilon v curl_unit_b2 / sqrt(g)
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b2_1,
            curl_unit_b2_2,
            curl_unit_b2_3,
            b_star,
        )

        b_star *= epsilon * v / det_df

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1_1,
            b1_2,
            b1_3,
            b1,
        )

        # G^{-1} b1 = DF^{-1} DF^{-T} b1, added to b_star
        linalg_kernels.matrix_vector(dfinvt, b1, tmp)
        linalg_kernels.matrix_vector(dfinv, tmp, b1)
        b_star += b1

        # compute B*_parallel (0-form, no metric factor)
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

        # calculate k for eta
        k[:] = b_star / b_star_parallel * v

        # calculate k_v for v
        k_v = linalg_kernels.scalar_dot(b_star, e_star)
        k_v /= b_star_parallel * epsilon

        # accumulation for last stage and update of positions (regular at a pole)
        pusher_utilities_kernels.update_position_stage(ip, k, dt, a[stage], b[stage], last, args_markers)

        # same for v_parallel
        markers[first_free_idx + 3, ip] += dt * b[stage] * k_v
        markers[3, ip] = (
            markers[first_pusher_idx + 3, ip] + dt * a[stage] * k_v + last * markers[first_free_idx + 3, ip]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
