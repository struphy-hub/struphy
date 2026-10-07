"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_isotherm_kappa(
    alpha: "float[:]",
    output_indices: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
):
    """Store a constant isothermal coefficient of one at the requested column."""

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        if output_indices[0] >= 0:
            markers[ip, output_indices[0]] = 1.0
