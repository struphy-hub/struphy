"""Evaluate parallel velocity and magnetic moment of each particles."""

from numpy import shape

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments


def eval_magnetic_moment_5d(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    first_diagnostics_idx: int,
    absB: "float[:,:,:]",
):
    """
    Evaluate parallel velocity and magnetic moment of each particles
    and assign it into markers[ip,first_diagnostics_idx+1].
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

        v_perp = markers[ip, 4]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            absB,
        )

        # magnetic moment
        markers[ip, first_diagnostics_idx + 1] = 1 / 2 * v_perp**2 / abs_B
