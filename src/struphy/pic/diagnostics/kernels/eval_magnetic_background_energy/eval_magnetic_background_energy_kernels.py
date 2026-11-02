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
    The result is stored at markers[first_diagnostics_idx, :].
    """

    # get number of markers
    n_markers = shape(markers)[1]

    # fmt: off
    #$ omp parallel private(abs_B, eta1, eta2, eta3, ip, mu, span1, span2, span3, thread_derham)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        mu = markers[mu_idx, ip]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # abs_B0; 0form
        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            abs_B0,
        )

        markers[first_diagnostics_idx, ip] = mu * abs_B

    # fmt: off
    #$ omp end parallel
    # fmt: on
