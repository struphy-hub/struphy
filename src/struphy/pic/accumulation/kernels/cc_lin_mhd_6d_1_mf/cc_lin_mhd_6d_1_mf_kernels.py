"Data of the density matrix of full-orbit (6D) particles per marker, matrix-free variant of cc_lin_mhd_6d_1."

from numpy import empty, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("b", "b_prod", "dfm", "df_inv", "df_inv_t", "g_inv", "tmp1", "tmp2")
def cc_lin_mhd_6d_1_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    boundary_cut: "float",
    column: "int",
):
    r"""Stores, per marker, the upper entries :math:`(F_{p,12}, F_{p,13}, F_{p,23})` of the antisymmetric 3x3 filling
    matrix :math:`F_p` of the density matrix :math:`A = \sum_p \vec \Lambda_p F_p \vec \Lambda_p^\top` of the
    propagator :class:`~struphy.propagators.current_coupling_6d_density.CurrentCoupling6DDensity` in the marker columns
    ``column`` to ``column + 2`` (zero for markers in the boundary cut), for the matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_asym_dot.marker_operator_asym_dot_kernels.marker_operator_asym_dot`.
    The filling matrix is that of the assembled variant
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1.cc_lin_mhd_6d_1_kernels.cc_lin_mhd_6d_1`.
    """

    markers = args_markers.markers

    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

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

        eval_2form_spline_mpi(span1, span2, span3, args_derham, b2_1, b2_2, b2_3, b)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # factor of the filling matrix
        fac = -markers[6, ip] * scale_mat

        if basis_u == 1:
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
            markers[column, ip] = fac * tmp2[0, 1]
            markers[column + 1, ip] = fac * tmp2[0, 2]
            markers[column + 2, ip] = fac * tmp2[1, 2]
        else:
            if basis_u == 2:
                fac = fac / det_df**2
            markers[column, ip] = fac * b_prod[0, 1]
            markers[column + 1, ip] = fac * b_prod[0, 2]
            markers[column + 2, ip] = fac * b_prod[1, 2]
