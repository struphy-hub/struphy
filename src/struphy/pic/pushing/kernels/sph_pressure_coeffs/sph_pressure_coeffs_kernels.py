"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from numpy import shape
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_pressure_coeffs(
    alpha: "float[:]",
    output_indices: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:, :]",
    neighbours: "int[:, :]",
    holes: "bool[:]",
    periodic1: "bool",
    periodic2: "bool",
    periodic3: "bool",
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
):
    r"""For each particle, evaluate

    * the density :math:`\rho^{N,h}(\boldsymbol \eta_i)` and store it at ``markers[:, output_indices[0]]``)
    * the coefficient :math:`w_i/\rho^{N,h}(\boldsymbol \eta_i)` and store it at ``markers[:, output_indices[1]]``)
    * the coefficient :math:`w_i (\rho^{N,h}(\boldsymbol \eta_i))^{\gamma - 2}` and store it at ``markers[:, output_indices[2]]``)

    where the smoothed SPH density is given by

    .. math::

        \rho^{N,h}(\boldsymbol \eta_i) = \sum_j w_j \, W_h(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.
    """

    gamma = 5 / 3

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    Np = args_markers.Np
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])
        n_at_eta = sph_eval_kernels.box_based_kernel(
            args_markers,
            eta1,
            eta2,
            eta3,
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            weight_idx,
            kernel_type,
            h1,
            h2,
            h3,
        )
        weight = markers[ip, weight_idx]
        # save
        if output_indices[0] >= 0:
            markers[ip, output_indices[0]] = n_at_eta
        if output_indices[1] >= 0:
            markers[ip, output_indices[1]] = weight / n_at_eta
        if output_indices[2] >= 0:
            markers[ip, output_indices[2]] = weight * n_at_eta ** (gamma - 2)
