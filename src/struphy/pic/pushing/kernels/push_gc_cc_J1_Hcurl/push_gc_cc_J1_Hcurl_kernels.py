"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_t", "g", "g_inv", "e", "u", "u0", "b", "b_star", "norm_b1", "curl_norm_b")
def push_gc_cc_J1_Hcurl(
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
    r"""Velocity update step for the :class:`~struphy.propagators.current_coupling_5d_curlb.CurrentCoupling5DCurlb`.

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^1 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    u0 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            b,
        )

        # u; 1form
        eval_1form_spline_mpi(
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

        # b_star; in H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        linalg_kernels.matrix_vector(g_inv, u, u0)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u0, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp / abs_b_star_para * v * dt
