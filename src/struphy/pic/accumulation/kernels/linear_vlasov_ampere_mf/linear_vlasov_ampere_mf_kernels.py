"Accumulation kernel for full-orbit (6D) particles, matrix-free variant of linear_vlasov_ampere."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "v", "df_inv_v")
def linear_vlasov_ampere_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    f0_values: "float[:]",
    column: "int",
):
    r"""Accumulates the vector of
    :func:`~struphy.pic.accumulation.kernels.linear_vlasov_ampere.linear_vlasov_ampere_kernels.linear_vlasov_ampere`
    into V1, with the filling :math:`B_p^\mu = w_p [DF^{-1}(\boldsymbol \eta_p)\, \mathbf v_p]_\mu`, and stores, per
    marker, the rank-one filling :math:`F_p = c_p\, \mathbf e_p \mathbf e_p^\top` of the matrix, with

    .. math::

        \mathbf e_p = DF^{-1}(\boldsymbol \eta_p)\, \mathbf v_p\,,\qquad c_p = \frac{f_0(\boldsymbol \eta_p, \mathbf v_p)}{N\, s_{0,p}}\,,

    in the marker columns ``column`` to ``column + 3`` (:math:`c_p = 0` for markers that are not accumulated), for the
    matrix-free application by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_rank1_dot.marker_operator_rank1_dot_kernels.marker_operator_rank1_dot`.

    Parameters
    ----------
    f0_values : array[float]
        Value of f0 for each particle.

    column : int
        First marker column of the filling.
    """

    markers = args_markers.markers

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)

    n_markers = args_markers.n_markers

    # total number of markers (weights are w_p = delta f_p / (N * s_0))
    n_markers_tot = args_markers.Np

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        if markers[-1, ip] == -2.0:
            markers[column + 3, ip] = 0.0
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        v[:] = markers[3:6, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        markers[column, ip] = df_inv_v[0]
        markers[column + 1, ip] = df_inv_v[1]
        markers[column + 2, ip] = df_inv_v[2]
        markers[column + 3, ip] = f0_values[ip] / (markers[7, ip] * n_markers_tot)

        weight = markers[6, ip]

        particle_to_mat_kernels.vec_fill_v1(
            args_derham,
            span1,
            span2,
            span3,
            vec1,
            vec2,
            vec3,
            weight * df_inv_v[0],
            weight * df_inv_v[1],
            weight * df_inv_v[2],
        )
