"Pusher kernel for gyro-center (5D) dynamics, magnetic field as a 1-form."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "e", "u", "b", "bv", "b_star", "norm_b1", "curl_norm_b")
def push_gc_cc_J1_H1vec_1form(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
):
    r"""Velocity update step for the :class:`~struphy.propagators.current_coupling_5d_curlb_1form.CurrentCoupling5DCurlb1form`.

    Marker update :

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^v (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.

    1-form version of
    :func:`~struphy.pic.pushing.kernels.push_gc_cc_J1_H1vec.push_gc_cc_J1_H1vec_kernels.push_gc_cc_J1_H1vec`: ``b1, b2, b3`` are the FE coefficients of
    the magnetic field as a 1-form :math:`\hat{\mathbf B}^1`. The kernel uses its vector-field components
    :math:`\hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1` (the 2-form is :math:`\hat{\mathbf B}^2 = \sqrt g\,
    \hat{\mathbf B}^\textnormal{v}`) and :math:`\hat B^{*\textnormal{v}}_\parallel = \hat{\mathbf b}^1_0 \cdot
    (\hat{\mathbf B}^\textnormal{v} + \epsilon v_\parallel \hat \nabla \times \hat{\mathbf b}^1_0 / \sqrt g) =
    \hat B^*_\parallel / \sqrt g`; the factor :math:`\sqrt g` cancels in the ratio of the field and
    :math:`\hat B^*_\parallel`.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    bv = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v = markers[3, ip]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            b,
        )

        # u; H1vec
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # vector-field components bv = G^{-1} b (the 2-form is det_df * bv)
        linalg_kernels.matrix_vector(g_inv, b, bv)

        # b_star; vector-field components (2-form divided by det_df)
        b_star[:] = bv + curl_norm_b * v * epsilon / det_df

        # calculate abs_b_star_para (divided by det_df, which cancels with the det_df of the 2-form b)
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # electric field E = bv x u (the det_df of the 2-form b cancels with the one of abs_b_star_para)
        linalg_kernels.cross(bv, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b)

        markers[3, ip] += temp / abs_b_star_para * v * dt
