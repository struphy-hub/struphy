r"""Evaluate :math:`mu_p |B_0(\boldsymbol \eta_p)|` for each marker."""

from numpy import shape
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("dfm", "norm_b1", "b")
def eval_magnetic_background_energy(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    first_diagnostics_idx: int,
    mu_idx: int,
    abs_B0: "float[:,:,:]",
):
    r"""
    Evaluate :math:`mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
    The result is stored at markers[:, first_diagnostics_idx].
    """

    # get number of markers
    n_markers = shape(markers)[0]

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, mu, span1, span2, span3, abs_B)
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        mu = markers[ip, mu_idx]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            abs_B0,
        )

        markers[ip, first_diagnostics_idx] = mu * abs_B
