"Data of the density matrix of full-orbit (6D) particles per marker, matrix-free variant of cc_lin_mhd_6d_1_1form."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("b", "b_vec", "dfm", "df_inv", "df_inv_t", "g_inv")
def cc_lin_mhd_6d_1_mf_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    boundary_cut: "float",
    column: "int",
):
    r"""Stores, per marker, the upper entries :math:`(F_{p,12}, F_{p,13}, F_{p,23})` of the antisymmetric 3x3 filling
    matrix :math:`F_p` of the density matrix :math:`A = \sum_p \vec \Lambda_p F_p \vec \Lambda_p^\top` of the
    propagator
    :class:`~struphy.propagators.current_coupling_6d_density_1form.CurrentCoupling6DDensity1form` in the marker
    columns ``column`` to ``column + 2`` (zero for markers in the boundary cut), for the matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_asym_dot.marker_operator_asym_dot_kernels.marker_operator_asym_dot`.
    The filling matrix is that of the assembled variant
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1_1form.cc_lin_mhd_6d_1_1form_kernels.cc_lin_mhd_6d_1_1form`
    (magnetic field as a 1-form).
    """

    markers = args_markers.markers

    b = empty(3, dtype=float)
    b_vec = empty(3, dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # boundary cut
        if markers[0, ip] < boundary_cut or markers[0, ip] > 1.0 - boundary_cut:
            markers[column, ip] = 0.0
            markers[column + 1, ip] = 0.0
            markers[column + 2, ip] = 0.0
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field: 1-form components
        eval_1form_spline_mpi(span1, span2, span3, args_derham, b1_1, b1_2, b1_3, b)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        if basis_u == 1:
            # F_p = [b1]_x / det_df
            b_vec[:] = b
            fac = -markers[6, ip] * scale_mat / det_df
        else:
            # vector-field components b^v = G^(-1) b1
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_vector(g_inv, b, b_vec)

            # F_p = det_df * [b^v]_x (H1vec) or [b^v]_x / det_df (Hdiv)
            if basis_u == 0:
                fac = -markers[6, ip] * scale_mat * det_df
            else:
                fac = -markers[6, ip] * scale_mat / det_df

        # entries 12, 13, 23 of the operator bx() as matrix
        markers[column, ip] = -fac * b_vec[2]
        markers[column + 1, ip] = fac * b_vec[1]
        markers[column + 2, ip] = -fac * b_vec[0]
