"""TODO."""

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_t", "g", "g_inv", "gradB, grad_PB_b", "tmp", "eta_mid", "eta_diff")
def eval_gradB_ediff(
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    gradB1: "float[:,:,:]",
    gradB2: "float[:,:,:]",
    gradB3: "float[:,:,:]",
    grad_PB_b1: "float[:,:,:]",
    grad_PB_b2: "float[:,:,:]",
    grad_PB_b3: "float[:,:,:]",
    idx: int,
):
    r"""TODO"""

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for magnetic field evaluation
    gradB = empty(3, dtype=float)
    grad_PB_b = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, mid point
        eta_mid[:] = (markers[ip, 0:3] + markers[ip, first_init_idx : first_init_idx + 3]) / 2.0
        eta_mid[:] = mod(eta_mid[:], 1.0)

        eta_diff = markers[ip, 0:3] - markers[ip, first_init_idx : first_init_idx + 3]

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, mu_idx]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham)
        # logger.info(span1, span2, span3)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_domain,
            dfm,
        )

        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # gradB; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            gradB1,
            gradB2,
            gradB3,
            gradB,
        )

        # grad_PB_b; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_PB_b1,
            grad_PB_b2,
            grad_PB_b3,
            grad_PB_b,
        )

        tmp = gradB + grad_PB_b

        markers[ip, idx] = linalg_kernels.scalar_dot(eta_diff, tmp)
        markers[ip, idx] *= mu
