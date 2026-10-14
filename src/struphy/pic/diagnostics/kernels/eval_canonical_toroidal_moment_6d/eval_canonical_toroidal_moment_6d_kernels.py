"""Evaluate canonical toroidal momentum of each particles and assign it into markers[ip,first_diagnostics_idx+5]."""

from numpy import shape, sign, sqrt

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments


def eval_canonical_toroidal_moment_6d(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    first_diagnostics_idx: int,
    epsilon: float,
    B0: float,
    R0: float,
    absB: "float[:,:,:]",
):
    """
    Evaluate canonical toroidal momentum of each particles and assign it into markers[ip,first_diagnostics_idx+5].
    """

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        energy = markers[ip, first_diagnostics_idx + 3]
        mu = markers[ip, first_diagnostics_idx + 4]
        psi = markers[ip, first_diagnostics_idx + 5]
        v_para = markers[ip, first_diagnostics_idx + 6]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            absB,
        )

        # shifted canonical toroidal momentum
        markers[ip, first_diagnostics_idx + 5] = psi - epsilon * B0 * R0 / abs_B * v_para

        if energy - mu * B0 > 0:
            markers[ip, first_diagnostics_idx + 5] += epsilon * sign(v_para) * sqrt(2 * (energy - mu * B0)) * R0
