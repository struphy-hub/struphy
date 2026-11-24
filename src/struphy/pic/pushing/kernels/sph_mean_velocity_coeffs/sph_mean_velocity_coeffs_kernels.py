"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("values", "grads")
def sph_mean_velocity_coeffs(
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
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    kernel: "int",
    dim: "int",
    h: "float",
    scale: "float",
):
    r"""For each marker :math:`i`, evaluate the SPH density :math:`\rho_i = \sum_j w_j W_h(\mathbf x_i - \mathbf x_j)`
    and store the coefficient

    * :math:`w_i v_{k,i} / \rho_i` at ``markers[output_indices[k], :]`` for :math:`k = 0, 1, 2`.

    These coefficients serve as kernel weights to evaluate the SPH mean velocity

    .. math::

        \mathbf v^{N,h}(\mathbf x) = \sum_j \frac{w_j \, \mathbf v_j}{\rho_j} \, W_h(\mathbf x - \mathbf x_j)\,.

    The kernel arguments (``periodic1`` to ``scale``) are those of
    :func:`~struphy.pic.sph_eval_kernels.box_based_physical`.
    """
    values = zeros(1, dtype=float)
    grads = zeros((1, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[0]
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        loc_box = int(markers[n_cols - 2, ip])
        sph_eval_kernels.box_based_columns(
            args_markers,
            markers[x_idx, ip],
            markers[x_idx + 1, ip],
            markers[x_idx + 2, ip],
            markers[0, ip],
            markers[1, ip],
            markers[2, ip],
            loc_box,
            boxes,
            neighbours,
            holes,
            periodic1,
            periodic2,
            periodic3,
            period,
            proj,
            x_idx,
            weight_idx,
            1,
            kernel,
            dim,
            h,
            scale,
            values,
            grads,
        )
        coeff = markers[weight_idx, ip] / values[0]
        # save
        for k in range(3):
            if output_indices[k] >= 0:
                markers[output_indices[k], ip] = coeff * markers[3 + k, ip]
