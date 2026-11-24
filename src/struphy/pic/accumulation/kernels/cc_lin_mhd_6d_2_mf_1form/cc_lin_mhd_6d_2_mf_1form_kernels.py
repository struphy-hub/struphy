"Accumulation kernel for full-orbit (6D) particles, matrix-free variant of cc_lin_mhd_6d_2_1form."

from numpy import empty, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("b", "b_cart", "b_prod", "dfm", "df_t", "df_inv", "df_inv_t", "tmp2", "tmp_t", "tmp_m", "tmp_v", "v")
def cc_lin_mhd_6d_2_mf_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
    column: "int",
):
    r"""Accumulates the vector of
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2_1form.cc_lin_mhd_6d_2_1form_kernels.cc_lin_mhd_6d_2_1form`
    and stores, per marker, the entries 11, 12, 13, 22, 23, 33 of its symmetric 3x3 filling :math:`F_p` of the matrix
    in the marker columns ``column`` to ``column + 5`` (zero for markers in the boundary cut), for the matrix-free
    application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_symm_dot.marker_operator_symm_dot_kernels.marker_operator_symm_dot`.
    With the Cartesian magnetic field :math:`\mathbf B = DF^{-\top} \hat{\mathbf B}^1` (from the 1-form
    :math:`\hat{\mathbf B}^1`) and :math:`T_p = DF^{-1} \mathbf B_\times` for ``basis_u = 1`` and
    :math:`T_p = DF^\top \mathbf B_\times` otherwise (all evaluated at :math:`\boldsymbol \eta_p`), the fillings are

    .. math::

        F_p = w_p\, T_p T_p^\top\, \frac{s_\textrm{mat}}{J_p^{2k}}\,,\qquad
        B_p = w_p\, T_p\, \mathbf v_p\, \frac{s_\textrm{vec}}{J_p^k}\,,

    with :math:`k = 1` for ``basis_u = 2`` and :math:`k = 0` otherwise.
    """

    markers = args_markers.markers

    b = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)

    tmp2 = empty((3, 3), dtype=float)
    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)
    tmp_v = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # boundary cut
        if markers[0, ip] < boundary_cut or markers[0, ip] > 1.0 - boundary_cut:
            markers[column : column + 6, ip] = 0.0
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field: 1-form components
        eval_1form_spline_mpi(span1, span2, span3, args_derham, b1_1, b1_2, b1_3, b)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # Cartesian components of the magnetic field B = DF^(-T) b1
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_vector(df_inv_t, b, b_cart)

        # operator Bx() as matrix
        b_prod[0, 1] = -b_cart[2]
        b_prod[0, 2] = +b_cart[1]
        b_prod[1, 0] = +b_cart[2]
        b_prod[1, 2] = -b_cart[0]
        b_prod[2, 0] = -b_cart[1]
        b_prod[2, 1] = +b_cart[0]

        weight = markers[6, ip]
        v[:] = markers[3:6, ip]

        # tmp2 = T_p
        if basis_u == 1:
            linalg_kernels.matrix_matrix(df_inv, b_prod, tmp2)
        else:
            linalg_kernels.transpose(dfm, df_t)
            linalg_kernels.matrix_matrix(df_t, b_prod, tmp2)

        linalg_kernels.transpose(tmp2, tmp_t)
        linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)
        linalg_kernels.matrix_vector(tmp2, v, tmp_v)

        fac_m = weight * scale_mat
        fac_v = weight * scale_vec
        if basis_u == 2:
            fac_m = fac_m / det_df**2
            fac_v = fac_v / det_df

        markers[column, ip] = fac_m * tmp_m[0, 0]
        markers[column + 1, ip] = fac_m * tmp_m[0, 1]
        markers[column + 2, ip] = fac_m * tmp_m[0, 2]
        markers[column + 3, ip] = fac_m * tmp_m[1, 1]
        markers[column + 4, ip] = fac_m * tmp_m[1, 2]
        markers[column + 5, ip] = fac_m * tmp_m[2, 2]

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham,
                span1,
                span2,
                span3,
                vec1,
                vec2,
                vec3,
                fac_v * tmp_v[0],
                fac_v * tmp_v[1],
                fac_v * tmp_v[2],
            )
        elif basis_u == 1:
            particle_to_mat_kernels.vec_fill_v1(
                args_derham,
                span1,
                span2,
                span3,
                vec1,
                vec2,
                vec3,
                fac_v * tmp_v[0],
                fac_v * tmp_v[1],
                fac_v * tmp_v[2],
            )
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(
                args_derham,
                span1,
                span2,
                span3,
                vec1,
                vec2,
                vec3,
                fac_v * tmp_v[0],
                fac_v * tmp_v[1],
                fac_v * tmp_v[2],
            )
