"Accumulation kernel for full-orbit (6D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv_t", "df_inv", "filling_m", "filling_v", "tmp1", "v", "tmp_v")
def pc_lin_mhd_6d(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11_11: "float[:,:,:,:,:,:]",
    mat12_11: "float[:,:,:,:,:,:]",
    mat13_11: "float[:,:,:,:,:,:]",
    mat22_11: "float[:,:,:,:,:,:]",
    mat23_11: "float[:,:,:,:,:,:]",
    mat33_11: "float[:,:,:,:,:,:]",
    mat11_12: "float[:,:,:,:,:,:]",
    mat12_12: "float[:,:,:,:,:,:]",
    mat13_12: "float[:,:,:,:,:,:]",
    mat22_12: "float[:,:,:,:,:,:]",
    mat23_12: "float[:,:,:,:,:,:]",
    mat33_12: "float[:,:,:,:,:,:]",
    mat11_13: "float[:,:,:,:,:,:]",
    mat12_13: "float[:,:,:,:,:,:]",
    mat13_13: "float[:,:,:,:,:,:]",
    mat22_13: "float[:,:,:,:,:,:]",
    mat23_13: "float[:,:,:,:,:,:]",
    mat33_13: "float[:,:,:,:,:,:]",
    mat11_22: "float[:,:,:,:,:,:]",
    mat12_22: "float[:,:,:,:,:,:]",
    mat13_22: "float[:,:,:,:,:,:]",
    mat22_22: "float[:,:,:,:,:,:]",
    mat23_22: "float[:,:,:,:,:,:]",
    mat33_22: "float[:,:,:,:,:,:]",
    mat11_23: "float[:,:,:,:,:,:]",
    mat12_23: "float[:,:,:,:,:,:]",
    mat13_23: "float[:,:,:,:,:,:]",
    mat22_23: "float[:,:,:,:,:,:]",
    mat23_23: "float[:,:,:,:,:,:]",
    mat33_23: "float[:,:,:,:,:,:]",
    mat11_33: "float[:,:,:,:,:,:]",
    mat12_33: "float[:,:,:,:,:,:]",
    mat13_33: "float[:,:,:,:,:,:]",
    mat22_33: "float[:,:,:,:,:,:]",
    mat23_33: "float[:,:,:,:,:,:]",
    mat33_33: "float[:,:,:,:,:,:]",
    vec1_1: "float[:,:,:]",
    vec2_1: "float[:,:,:]",
    vec3_1: "float[:,:,:]",
    vec1_2: "float[:,:,:]",
    vec2_2: "float[:,:,:]",
    vec3_2: "float[:,:,:]",
    vec1_3: "float[:,:,:]",
    vec2_3: "float[:,:,:]",
    vec3_3: "float[:,:,:]",
    ep_scale: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        {V_{p,i}}_\perp A_p^{\mu, \nu} {V_{p,j}}_\perp &= w_p * [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} * {V_{p,i}}_\perp * {V_{p,j}}_\perp

        {V_{p,i}}_\perp B_p^\mu &= w_p * [DF^{-1}(\eta_p)]_\mu * {V_{p,i}}_\perp

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        # evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # Avoid second computation of dfm, use linear_algebra.linalg_kernels routines to get g_inv:
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)

        linalg_kernels.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        filling_m[:, :] = weight * tmp1 * ep_scale
        filling_v[:] = weight * tmp_v * ep_scale

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_pressure(
            args_derham,
            span1,
            span2,
            span3,
            mat11_11,
            mat12_11,
            mat13_11,
            mat22_11,
            mat23_11,
            mat33_11,
            mat11_12,
            mat12_12,
            mat13_12,
            mat22_12,
            mat23_12,
            mat33_12,
            mat11_22,
            mat12_22,
            mat13_22,
            mat22_22,
            mat23_22,
            mat33_22,
            filling_m[0, 0],
            filling_m[0, 1],
            filling_m[0, 2],
            filling_m[1, 1],
            filling_m[1, 2],
            filling_m[2, 2],
            vec1_1,
            vec2_1,
            vec3_1,
            vec1_2,
            vec2_2,
            vec3_2,
            filling_v[0],
            filling_v[1],
            filling_v[2],
            v[0],
            v[1],
        )
