"""Evaluate canonical toroidal momentum of each particles and assign it into markers[first_diagnostics_idx+5, ip]."""

from numpy import floor, shape, sign, sqrt

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
    shift_mode: int,
    psi_axis: float,
    psi_edge: float,
    bmax_table: "float[:]",
):
    """
    Evaluate the shifted canonical toroidal momentum of each particles and assign it into markers[first_diagnostics_idx+5, ip].

    The shift uses the reference field B_ref = B0 (``shift_mode=1``) or the maximum of abs(B) on the flux surface
    psi = P_phi (``shift_mode=2``, linear interpolation in ``bmax_table`` on equidistant values of the square root of the
    normalized flux in [0, 1] between ``psi_axis`` and ``psi_edge``); no shift for ``shift_mode=0``, see
    :class:`~struphy.kinetic_background.maxwellians.CanonicalMaxwellian2D`.
    """

    # get number of markers
    n_markers = shape(markers)[1]
    n_table = shape(bmax_table)[0]

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

        # canonical toroidal momentum
        p_phi = psi - epsilon * B0 * R0 / abs_B * v_para
        markers[first_diagnostics_idx + 5, ip] = p_phi

        if shift_mode == 1:
            if energy - mu * B0 > 0:
                markers[first_diagnostics_idx + 5, ip] += epsilon * sign(v_para) * sqrt(2 * (energy - mu * B0)) * R0
        elif shift_mode == 2:
            # maximum of abs(B) on the flux surface psi = p_phi (table in the square root of the clamped normalized flux)
            s = min(max((p_phi - psi_axis) / (psi_edge - psi_axis), 0.0), 1.0)
            rho = sqrt(s) * (n_table - 1)
            i_tab = min(int(floor(rho)), n_table - 2)
            b_ref = bmax_table[i_tab] + (rho - i_tab) * (bmax_table[i_tab + 1] - bmax_table[i_tab])

            if energy - mu * b_ref > 0:
                markers[first_diagnostics_idx + 5, ip] += (
                    epsilon * sign(v_para) * sqrt(2 * (energy - mu * b_ref)) * R0 * (B0 / b_ref)
                )
