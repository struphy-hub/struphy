"""Evaluate canonical toroidal momentum of each particles and assign it into markers[first_diagnostics_idx+5, ip]."""

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
    Evaluate canonical toroidal momentum of each particles and assign it into markers[first_diagnostics_idx+5, ip].
    """

    # get number of markers
    n_markers = shape(markers)[1]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        energy = markers[first_diagnostics_idx + 3, ip]
        mu = markers[first_diagnostics_idx + 4, ip]
        psi = markers[first_diagnostics_idx + 5, ip]
        v_para = markers[first_diagnostics_idx + 6, ip]

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
        markers[first_diagnostics_idx + 5, ip] = psi - epsilon * B0 * R0 / abs_B * v_para

        if energy - mu * B0 > 0:
            markers[first_diagnostics_idx + 5, ip] += epsilon * sign(v_para) * sqrt(2 * (energy - mu * B0)) * R0
