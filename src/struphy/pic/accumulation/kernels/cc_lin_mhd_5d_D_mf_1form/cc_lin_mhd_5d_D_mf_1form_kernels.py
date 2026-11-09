"Data of the density matrix of gyro-center (5D) particles per marker, matrix-free, magnetic field as a 1-form."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "b1", "bv", "b_star", "norm_b1", "curl_norm_b")
def cc_lin_mhd_5d_D_mf_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    epsilon: float,
    ep_scale: "float",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    basis_u: "int",
    column: "int",
):
    r"""Stores, per marker, the upper entries :math:`(F_{p,12}, F_{p,13}, F_{p,23})` of the antisymmetric 3x3 filling
    matrix :math:`F_p` of the density matrix :math:`A = \sum_p \vec \Lambda_p F_p \vec \Lambda_p^\top` of the
    propagator
    :class:`~struphy.propagators.current_coupling_5d_density_1form.CurrentCoupling5DDensity1form` (magnetic field as
    a 1-form) in the marker columns ``column`` to ``column + 2``, for the matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_asym_dot.marker_operator_asym_dot_kernels.marker_operator_asym_dot`.
    The filling matrix is that of the assembled variant
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D_1form.cc_lin_mhd_5d_D_1form_kernels.cc_lin_mhd_5d_D_1form`.
    """

    markers = args_markers.markers

    b1 = empty(3, dtype=float)
    bv = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        v = markers[3, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_1form_spline_mpi(span1, span2, span3, args_derham, b1_1, b1_2, b1_3, b1)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # vector-field components bv = G^{-1} b1 (the 2-form is det_df * bv) and Bstar = bv + eps v curl_norm_b / det_df
        linalg_kernels.matrix_vector(g_inv, b1, bv)
        b_star[:] = bv + epsilon * v * curl_norm_b / det_df
        b_para = linalg_kernels.scalar_dot(norm_b1, bv)
        b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # factor of the filling matrix
        fac = -markers[5, ip] * (1 - b_para / b_star_para) * ep_scale / epsilon

        if basis_u == 0:
            # 2-form bx() = det_df * bvx()
            fac = fac * det_df
            markers[column, ip] = -fac * bv[2]
            markers[column + 1, ip] = +fac * bv[1]
            markers[column + 2, ip] = -fac * bv[0]
        elif basis_u == 1:
            # G^{-1} (2-form bx()) G^{-1} = b1x() / det_df
            fac = fac / det_df
            markers[column, ip] = -fac * b1[2]
            markers[column + 1, ip] = +fac * b1[1]
            markers[column + 2, ip] = -fac * b1[0]
        else:
            # (2-form bx()) / det_df**2 = bvx() / det_df
            fac = fac / det_df
            markers[column, ip] = -fac * bv[2]
            markers[column + 1, ip] = +fac * bv[1]
            markers[column + 2, ip] = -fac * bv[0]
