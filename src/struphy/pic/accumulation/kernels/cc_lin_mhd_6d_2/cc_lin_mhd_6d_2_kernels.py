"Accumulation kernel for full-orbit (6D) particles."

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "b",
    "b_prod",
    "dfm",
    "df_inv",
    "df_inv_t",
    "g_inv",
    "filling_m",
    "filling_v",
    "tmp1",
    "tmp2",
    "tmp_t",
    "tmp_v",
    "tmp_m",
    "v",
)
def cc_lin_mhd_6d_2(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11: "float[:,:,:,:,:,:]",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat22: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    mat33: "float[:,:,:,:,:,:]",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) ]_{\mu, \nu}

        B_p^\mu &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * DF^{-1}(\eta_p) * v_p ]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, b2_1, b2_2, b2_3, b, dfm, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1, tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        if basis_u == 0:
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v0vec_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )

        elif basis_u == 1:
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp2 * tmp2^T and tmp_v = tmp2 * v, where tmp2 = G^(-1) * B^x * DF^(-1)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, df_inv, tmp2)

            linalg_kernels.transpose(tmp2, tmp_t)

            linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp2, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v1_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )

        elif basis_u == 2:
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1) / det(DF)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat / det_df**2
            filling_v[:] = weight * tmp_v * scale_vec / det_df

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v2_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )
