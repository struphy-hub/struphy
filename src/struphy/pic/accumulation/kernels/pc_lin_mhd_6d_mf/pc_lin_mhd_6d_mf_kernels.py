"Accumulation kernel for full-orbit (6D) particles, matrix-free variant of pc_lin_mhd_6d and pc_lin_mhd_6d_full."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "v", "tmp_v")
def pc_lin_mhd_6d_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
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
    n_v: "int",
    column: "int",
):
    r"""Accumulates the three vectors of
    :func:`~struphy.pic.accumulation.kernels.pc_lin_mhd_6d.pc_lin_mhd_6d_kernels.pc_lin_mhd_6d` (``n_v = 2``) or
    :func:`~struphy.pic.accumulation.kernels.pc_lin_mhd_6d_full.pc_lin_mhd_6d_full_kernels.pc_lin_mhd_6d_full`
    (``n_v = 3``) into V1, with the fillings :math:`v_{p,a} B_p^\mu` (:math:`a \leq` ``n_v``),

    .. math::

        B_p = w_p\, s_\textrm{ep}\, DF^{-1}(\boldsymbol \eta_p)\, \mathbf v_p\,,

    and stores, per marker, the entries 11, 12, 13, 22, 23, 33 of the symmetric filling
    :math:`F_p = w_p\, s_\textrm{ep}\, G^{-1}(\boldsymbol \eta_p)` of the matrices in the marker columns ``column`` to
    ``column + 5``, for the matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_pressure_dot.marker_operator_pressure_dot_kernels.marker_operator_pressure_dot`.
    """

    markers = args_markers.markers

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        weight = markers[6, ip]
        v[:] = markers[3:6, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        fac = weight * ep_scale

        markers[column, ip] = fac * g_inv[0, 0]
        markers[column + 1, ip] = fac * g_inv[0, 1]
        markers[column + 2, ip] = fac * g_inv[0, 2]
        markers[column + 3, ip] = fac * g_inv[1, 1]
        markers[column + 4, ip] = fac * g_inv[1, 2]
        markers[column + 5, ip] = fac * g_inv[2, 2]

        particle_to_mat_kernels.vec_fill_v1(
            args_derham,
            span1,
            span2,
            span3,
            vec1_1,
            vec2_1,
            vec3_1,
            fac * tmp_v[0] * v[0],
            fac * tmp_v[1] * v[0],
            fac * tmp_v[2] * v[0],
        )
        particle_to_mat_kernels.vec_fill_v1(
            args_derham,
            span1,
            span2,
            span3,
            vec1_2,
            vec2_2,
            vec3_2,
            fac * tmp_v[0] * v[1],
            fac * tmp_v[1] * v[1],
            fac * tmp_v[2] * v[1],
        )
        if n_v == 3:
            particle_to_mat_kernels.vec_fill_v1(
                args_derham,
                span1,
                span2,
                span3,
                vec1_3,
                vec2_3,
                vec3_3,
                fac * tmp_v[0] * v[2],
                fac * tmp_v[1] * v[2],
                fac * tmp_v[2] * v[2],
            )
