"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, mod, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "dfm",
    "df_inv_t",
    "df_inv",
    "g_inv",
    "filling_v",
    "tmp",
    "tmp_v",
    "b",
    "b_prod",
    "eta_diff",
    "beq",
    "beq_prod",
    "norm_b_prod",
    "bfull_star",
    "curl_norm_b",
    "norm_b1",
    "grad_PB",
    "grad_PBeq",
    "eta_mid",
    "eta_diff",
)
def cc_lin_mhd_5d_gradB_dg(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,
    ep_scale: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    beq1: "float[:,:,:]",
    beq2: "float[:,:,:]",
    beq3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    grad_PB1: "float[:,:,:]",
    grad_PB2: "float[:,:,:]",
    grad_PB3: "float[:,:,:]",
    grad_PBeq1: "float[:,:,:]",
    grad_PBeq2: "float[:,:,:]",
    grad_PBeq3: "float[:,:,:]",
    basis_u: "int",
    const: "float",
):
    r"""TODO"""

    markers = args_markers.markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx

    # allocate for magnetic field evaluation
    eta_diff = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    beq_prod = zeros((3, 3), dtype=float)
    norm_b_prod = zeros((3, 3), dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_PB = empty(3, dtype=float)
    grad_PBeq = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)
    tmp = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, mid point
        eta_mid[:] = (markers[ip, 0:3] + markers[ip, first_init_idx : first_init_idx + 3]) / 2.0
        eta_mid[:] = mod(eta_mid[:], 1.0)

        eta_diff[:] = markers[ip, 0:3] - markers[ip, first_init_idx : first_init_idx + 3]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta_mid[0], eta_mid[1], eta_mid[2], args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # needed metric coefficients
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)

        # beq; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, beq1, beq2, beq3, beq)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # grad_PBeq; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PBeq1, grad_PBeq2, grad_PBeq3, grad_PBeq)

        # b_star; 2form transformed into H1vec
        bfull_star[:] = b + beq + curl_norm_b * v * epsilon

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        beq_prod[0, 1] = -beq[2]
        beq_prod[0, 2] = +beq[1]
        beq_prod[1, 0] = +beq[2]
        beq_prod[1, 2] = -beq[0]
        beq_prod[2, 0] = -beq[1]
        beq_prod[2, 1] = +beq[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        if basis_u == 0:
            # beq * gradPBeq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * ep_scale

            # beq * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * ep_scale

            # beq * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)
            filling_v[:] += tmp_v / abs_b_star_para * const

            # b * gradPBeq contribution
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * ep_scale

            # b * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * ep_scale

            # b * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)
            filling_v[:] += tmp_v / abs_b_star_para * const

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:
            # beq * gradPBeq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para / det_df * ep_scale

            # beq * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * ep_scale

            # beq * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)

            filling_v[:] += tmp_v / abs_b_star_para / det_df * const

            # b * gradPBeq contribtuion
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * ep_scale

            # b * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * ep_scale

            # b * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)

            filling_v[:] += tmp_v / abs_b_star_para / det_df * const

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )
