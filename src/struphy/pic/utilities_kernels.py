from numpy import abs, empty, log, mod, pi, shape, sign, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments

# fmt: off
    #$ omp end parallel
    

# fmt: on


@stack_array("dfm", "norm_b1", "b")
def eval_magnetic_energy(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    first_diagnostics_idx: int,
    abs_B0: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
):
    r"""
    Evaluate :math:`mu_p |B(\boldsymbol \eta_p)_\parallel|` for each marker.
    The result is stored at markers[:, first_diagnostics_idx].
    """
    norm_b1 = empty(3, dtype=float)
    b = empty(3, dtype=float)

    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, mu, span1, span2, span3, b, b_para, abs_B, norm_b1, dfm, det_df)
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        mu = markers[ip, first_diagnostics_idx + 1]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            abs_B0,
        )

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

        b_para = linalg_kernels.scalar_dot(norm_b1, b)
        b_para /= det_df

        markers[ip, first_diagnostics_idx] = mu * (abs_B + b_para)

    # fmt: off
    #$ omp end parallel
    # fmt: on

    # fmt: off
    #$ omp end parallel
    # fmt: on

    # fmt: off
    #$ omp end parallel
    # fmt: on
