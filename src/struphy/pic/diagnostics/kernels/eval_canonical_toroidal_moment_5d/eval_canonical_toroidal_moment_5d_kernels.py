"""Evaluate canonical toroidal momentum of each particles and assign it into markers[ip,idx_can_momentum]."""

from numpy import shape, sign, sqrt

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments


def eval_canonical_toroidal_moment_5d(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    first_diagnostics_idx: int,
    mu_idx: int,
    idx_can_momentum: int,
    epsilon: float,
    B0: float,
    R0: float,
    absB: "float[:,:,:]",
):
    """
    Evaluate canonical toroidal momentum of each particles and assign it into markers[ip,idx_can_momentum].
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

        v_para = markers[ip, 3]
        mu = markers[ip, mu_idx]
        energy = markers[ip, first_diagnostics_idx]
        psi = markers[ip, idx_can_momentum]

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
        markers[ip, idx_can_momentum] = psi - epsilon * B0 * R0 / abs_B * v_para

        if energy - mu * B0 > 0:
            markers[ip, idx_can_momentum] += epsilon * sign(v_para) * sqrt(2 * (energy - mu * B0)) * R0
