"Accumulation kernel for gyro-center (5D) particles, magnetic field as a 1-form."

from numpy import empty, shape, zeros
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
    "df_inv",
    "df_inv_t",
    "g_inv",
    "filling_m",
    "filling_v",
    "tmp",
    "tmp1",
    "tmp_m",
    "tmp_v",
    "b",
    "bv",
    "bfull_star",
    "b_prod",
    "b_prod_neg",
    "norm_b1",
    "curl_norm_b",
)
def cc_lin_mhd_5d_curlb_1form(
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
    epsilon: float,
    ep_scale: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    basis_u: "int",
):
    r"""Accumulation kernel for the propagator
    :class:`~struphy.propagators.current_coupling_5d_curlb_1form.CurrentCoupling5DCurlb1form`, the 1-form version of
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb.cc_lin_mhd_5d_curlb_kernels.cc_lin_mhd_5d_curlb`.

    The magnetic field ``b1, b2, b3`` is a 1-form :math:`\hat{\mathbf B}^1` with vector-field components
    :math:`\hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1` (the 2-form is
    :math:`\sqrt g\, \hat{\mathbf B}^\textnormal{v}`). With :math:`\mathbf c = \hat \nabla \times \hat{\mathbf b}^1_0`
    (2-form) and :math:`\hat B^{*\textnormal{v}}_\parallel = \hat{\mathbf b}^1_0 \cdot (\hat{\mathbf B}^\textnormal{v} +
    \epsilon v_\parallel \mathbf c / \sqrt g)` the factors :math:`\sqrt g` of the 2-form kernel cancel, and the
    filling functions are

    .. math::

        A_p^{\mu, \nu} &= w_p \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{v_{\parallel,p}^2}{(\hat
        B^{*\textnormal{v}}_\parallel)^2} \left[ \hat{\mathbf B}^\textnormal{v}_\times\, \mathbf c \mathbf c^\top
        (\hat{\mathbf B}^\textnormal{v}_\times)^\top \right]_{\mu, \nu} \times \begin{cases} 1 & (H^1)^3\,, \\
        1/g & H(\textnormal{div})\,, \end{cases}

        B_p^\mu &= w_p \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{v_{\parallel,p}^2}{\hat
        B^{*\textnormal{v}}_\parallel} \left( \hat{\mathbf B}^\textnormal{v} \times \mathbf c \right)_\mu \times
        \begin{cases} 1 & (H^1)^3\,, \\ 1/\sqrt g & H(\textnormal{div})\,, \end{cases}

    where :math:`\mathbf a_\times \mathbf d := \mathbf a \times \mathbf d`.
    """

    markers = args_markers.markers

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    bv = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    b_prod_neg = zeros((3, 3), dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = zeros((3, 3), dtype=float)
    filling_v = zeros(3, dtype=float)

    tmp = empty((3, 3), dtype=float)
    tmp1 = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # marker positions
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # marker weight and velocity
        weight = markers[5, ip]
        v = markers[3, ip]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 1form, and its vector-field components bv = G^{-1} b (the 2-form is det_df * bv)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)
        linalg_kernels.matrix_vector(g_inv, b, bv)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # b_star; vector-field components (2-form divided by det_df)
        bfull_star[:] = bv + curl_norm_b * v * epsilon / det_df

        # calculate abs_b_star_para (divided by det_df, which cancels with the det_df of the 2-form b)
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # calculate tensor product of two curl_norm_b
        linalg_kernels.outer(curl_norm_b, curl_norm_b, tmp)

        # operator bvx() as matrix
        b_prod[0, 1] = -bv[2]
        b_prod[0, 2] = +bv[1]
        b_prod[1, 0] = +bv[2]
        b_prod[1, 2] = -bv[0]
        b_prod[2, 0] = -bv[1]
        b_prod[2, 1] = +bv[0]

        b_prod_neg[:] = -1.0 * b_prod

        if basis_u == 0:
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / abs_b_star_para**2 * ep_scale
            filling_v[:] = weight * tmp_v * v**2 / abs_b_star_para * ep_scale

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

        elif basis_u == 2:
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * ep_scale
            filling_v[:] = weight * tmp_v * v**2 / abs_b_star_para / det_df * ep_scale

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
