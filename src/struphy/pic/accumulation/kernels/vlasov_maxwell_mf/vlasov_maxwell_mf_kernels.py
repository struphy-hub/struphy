"Accumulation kernel for full-orbit (6D) particles, matrix-free variant of vlasov_maxwell."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "v", "df_inv_times_v")
def vlasov_maxwell_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    column: "int",
):
    r"""Accumulates the vector of
    :func:`~struphy.pic.accumulation.kernels.vlasov_maxwell.vlasov_maxwell_kernels.vlasov_maxwell` into V1, with the
    filling :math:`B_p^\mu = w_p [DF^{-1}(\boldsymbol \eta_p)\, \mathbf v_p]_\mu`, and stores, per marker, the entries
    11, 12, 13, 22, 23, 33 of the symmetric filling :math:`F_p = w_p\, G^{-1}(\boldsymbol \eta_p)` of the matrix in the
    marker columns ``column`` to ``column + 5``, for the matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_symm_dot.marker_operator_symm_dot_kernels.marker_operator_symm_dot`.
    """

    markers = args_markers.markers

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    df_inv_times_v = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        v[:] = markers[3:6, ip]
        weight = markers[6, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)

        markers[column, ip] = weight * g_inv[0, 0]
        markers[column + 1, ip] = weight * g_inv[0, 1]
        markers[column + 2, ip] = weight * g_inv[0, 2]
        markers[column + 3, ip] = weight * g_inv[1, 1]
        markers[column + 4, ip] = weight * g_inv[1, 2]
        markers[column + 5, ip] = weight * g_inv[2, 2]

        particle_to_mat_kernels.vec_fill_v1(
            args_derham,
            span1,
            span2,
            span3,
            vec1,
            vec2,
            vec3,
            weight * df_inv_times_v[0],
            weight * df_inv_times_v[1],
            weight * df_inv_times_v[2],
        )
