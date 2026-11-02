"Accumulation kernel for gyro-center (5D) particles, matrix-free variant of cc_lin_mhd_5d_curlb."

from numpy import empty
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


@stack_array("dfm", "b", "bfull_star", "norm_b1", "curl_norm_b", "e")
def cc_lin_mhd_5d_curlb_mf(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
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
    column: "int",
):
    r"""Accumulation kernel for the propagator
    :class:`~struphy.propagators.current_coupling_5d_curlb.CurrentCoupling5DCurlb` (matrix-free). Accumulates the
    vector and stores, per marker, the data of the matrix with the rank-one filling
    :math:`F_p = c_p\, \mathbf e_p \mathbf e_p^\top` (see :class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`)
    in the marker columns ``column`` to ``column + 3`` (:math:`\mathbf e_p` and :math:`c_p`), which is applied by
    :func:`~struphy.pic.accumulation.kernels.marker_operator_rank1_dot.marker_operator_rank1_dot_kernels.marker_operator_rank1_dot`:

    .. math::

        B_p^\mu = c_p\, e_{p,\mu}\,,\qquad c_p = w_p v_{\parallel,p}^2 \, \frac{A_\textnormal{h}}{A_\textnormal{b}}\,,\qquad
        \mathbf e_p = \frac{\hat{\mathbf B}^2 \times (\hat \nabla \times \hat{\mathbf b}^1_0)}{\hat B^*_\parallel \sqrt{g}}

    for :math:`\mathbf U` in :math:`H(\textnormal{div})` (``basis_u = 2``; without :math:`\sqrt g` for
    ``basis_u = 0``). The assembled variant is
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_curlb.cc_lin_mhd_5d_curlb_kernels.cc_lin_mhd_5d_curlb`.
    """

    markers = args_markers.markers

    b = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)

    n_markers_loc = args_markers.n_markers

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        weight = markers[5, ip]
        v = markers[3, ip]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # total magnetic field (2-form), unit vector b0 (1-form) and its curl (2-form)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        bfull_star[:] = b + curl_norm_b * v * epsilon
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # e = B x curl(b0) / B*_par (/ sqrt(g) for 2-forms)
        linalg_kernels.cross(b, curl_norm_b, e)
        e[:] = e / abs_b_star_para
        if basis_u == 2:
            e[:] = e / det_df

        c = weight * v**2 * ep_scale

        markers[column, ip] = e[0]
        markers[column + 1, ip] = e[1]
        markers[column + 2, ip] = e[2]
        markers[column + 3, ip] = c

        if basis_u == 0:
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, c * e[0], c * e[1], c * e[2]
            )
        elif basis_u == 2:
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, c * e[0], c * e[1], c * e[2]
            )
