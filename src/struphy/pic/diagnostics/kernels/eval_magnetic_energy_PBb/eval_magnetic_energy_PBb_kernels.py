r"""Evaluate :math:`mu_p |B(\boldsymbol \eta_p)_\parallel|` for each marker."""

from numpy import empty, mod, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("dfm", "eta")
def eval_magnetic_energy_PBb(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    first_diagnostics_idx: int,
    mu_idx: int,
    abs_B0: "float[:,:,:]",
    PBb: "float[:,:,:]",
):
    r"""
    Evaluate :math:`mu_p |B(\boldsymbol \eta_p)_\parallel|` for each marker.
    The result is stored at markers[:, first_diagnostics_idx].
    """
    eta = empty(3, dtype=float)

    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta[:] = mod(markers[ip, 0:3], 1.0)

        weight = markers[ip, 7]
        dweight = markers[ip, 5]

        mu = markers[ip, mu_idx]

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta[0],
            eta[1],
            eta[2],
            args_domain,
            dfm,
        )

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            abs_B0,
        )

        # PBb; 0form
        PB_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            PBb,
        )

        markers[ip, first_diagnostics_idx] = mu * (abs_B + PB_b)
