"""Evaluate total energy of each particles and assign it into markers[first_diagnostics_idx, ip]."""

from numpy import shape

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments


def eval_energy_5d(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    first_diagnostics_idx: int,
    mu_idx: int,
    absB: "float[:,:,:]",
):
    """
    Evaluate total energy of each particles and assign it into markers[first_diagnostics_idx, ip].
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

        v_parallel = markers[3, ip]
        mu = markers[mu_idx, ip]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            absB,
        )

        # total energy
        markers[first_diagnostics_idx, ip] = 1 / 2 * v_parallel**2 + mu * abs_B
