"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("values", "grads")
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
    period: "float[:,:]",
    proj: "float[:,:]",
    x_idx: "int",
    kernel: "int",
    dim: "int",
    h: "float",
    scale: "float",
    gamma: "float",
):
    r"""For each marker :math:`i`, evaluate the SPH density (in physical space)

    .. math::

        \rho_i = \sum_j w_j \, W_h(\mathbf x_i - \mathbf x_j)\,,

    and store

    * :math:`\rho_i` at ``markers[output_indices[0], :]``,
    * :math:`w_i/\rho_i` at ``markers[output_indices[1], :]`` (isothermal closure),
    * :math:`w_i \rho_i^{\gamma - 2}` at ``markers[output_indices[2], :]`` (polytropic closure with adiabatic
      index ``gamma``).

    The kernel arguments (``periodic1`` to ``scale``) are those of
    :func:`~struphy.pic.sph_eval_kernels.box_based_physical`, set up by
    :func:`~struphy.pic.sph_setup.setup_sph_kernel`.
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
        rho = values[0]
        weight = markers[weight_idx, ip]
        # save
        if output_indices[0] >= 0:
            markers[output_indices[0], ip] = rho
        if output_indices[1] >= 0:
            markers[output_indices[1], ip] = weight / rho
        if output_indices[2] >= 0:
            markers[output_indices[2], ip] = weight * rho ** (gamma - 2.0)
