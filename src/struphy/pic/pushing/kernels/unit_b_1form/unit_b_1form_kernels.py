"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "unit_b1")
def unit_b_1form(
    alpha: "float[:]",
    output_indices: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
):
    r"""Evaluate :math:`\hat{\mathbf b}^1_0(\boldsymbol \eta_p)`,
    where the evaluation point is the weighted average
    :math:`\eta_{p,i} = \alpha_i \eta_{p,i}^{n+1,k} + (1 - \alpha_i) \eta_{p,i}^n`,
    for :math:`i=1,2,3`. Markers must be sorted according to the evaluation point
    :math:`\boldsymbol \eta_p` beforehand.

    Component j is saved at marker column ``output_indices[j]``. Supply three indices;
    An index of -1 skips that component for every particle.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        # spline evaluation
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            unit_b1_1,
            unit_b1_2,
            unit_b1_3,
            unit_b1,
        )

        # save
        for j in range(3):
            if output_indices[j] >= 0:
                markers[ip, output_indices[j]] = unit_b1[j]
