"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, mod, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "dfm",
    "df_inv",
    "df_inv_t",
    "g_inv",
    "e",
    "u",
    "ud",
    "bb",
    "b_star",
    "norm_b1",
    "curl_norm_b",
    "tmp1",
    "tmp2",
    "b_prod",
    "norm_b_prod",
    "eta_old",
    "eta_mid",
)
def push_gc_cc_J2_dg_Hdiv(
    dt: float,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
    ud1: "float[:,:,:]",
    ud2: "float[:,:,:]",
    ud3: "float[:,:,:]",
    const: float,
    alpha: float,
):
    r"""TODO"""

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b_prod = zeros((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    ud = empty(3, dtype=float)
    bb = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    eta_old = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, mid point
        eta_old[:] = markers[ip, 0:3]
        eta_mid[:] = (markers[ip, 0:3] + markers[ip, first_init_idx : first_init_idx + 3]) / 2.0
        eta_mid[:] = mod(eta_mid[:], 1.0)

        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham)

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            bb,
        )

        # u; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # ud; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            ud1,
            ud2,
            ud3,
            ud,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -bb[2]
        b_prod[0, 2] = +bb[1]
        b_prod[1, 0] = +bb[2]
        b_prod[1, 2] = -bb[0]
        b_prod[2, 0] = -bb[1]
        b_prod[2, 1] = +bb[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        # b_star; 2form
        b_star[:] = bb + curl_norm_b * v * epsilon

        # calculate 3form abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        linalg_kernels.matrix_matrix(norm_b_prod, b_prod, tmp1)
        linalg_kernels.matrix_vector(tmp1, u, e)
        linalg_kernels.matrix_vector(tmp1, ud, tmp2)
        tmp2 *= const

        e += tmp2

        e /= abs_b_star_para
        e /= det_df

        markers[ip, 0:3] = markers[ip, first_init_idx : first_init_idx + 3] - dt * e
        markers[ip, 0:3] *= alpha
        markers[ip, 0:3] += eta_old * (1.0 - alpha)
