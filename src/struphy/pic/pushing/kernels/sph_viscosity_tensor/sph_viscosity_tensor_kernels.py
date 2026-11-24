"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("rho_val", "rho_grad", "v_val", "grad_v", "sigma")
def sph_viscosity_tensor(
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
    mu: "float",
):
    r"""For each marker :math:`i`, evaluate the SPH density :math:`\rho_i = \sum_l w_l W_h(\mathbf x_i - \mathbf x_l)`
    and the viscous stress (momentum flux)

    .. math::

        \sigma_{jm,i} = -\mu\bigl[ \partial_m v_j^{N,h}(\mathbf x_i) + \partial_j v_m^{N,h}(\mathbf x_i)
        - \tfrac{2}{3}\delta_{jm} \, \nabla \cdot \mathbf v^{N,h}(\mathbf x_i)\bigr]\,,
        \qquad
        \partial_m v_j^{N,h}(\mathbf x_i) = \sum_l \frac{w_l v_{j,l}}{\rho_l}\, \partial_m W_h(\mathbf x_i - \mathbf x_l)\,,

    and store the 9 coefficients

    * :math:`w_i \, \sigma_{jm,i} / \rho_i` at ``markers[output_indices[3*j + m], :]`` for :math:`j, m = 0, 1, 2`.

    These serve as kernel weights to evaluate the viscous force
    :math:`(\nabla \cdot \sigma)^{N,h}_j(\mathbf x) = \sum_l (w_l \sigma_{jm,l}/\rho_l)\, \partial_m W_h(\mathbf x - \mathbf x_l)`.

    This kernel requires the mean-velocity coefficients :math:`w_l \mathbf v_l / \rho_l` at
    ``markers[first_free_idx:first_free_idx + 3, :]``, computed by
    :func:`~struphy.pic.pushing.kernels.sph_mean_velocity_coeffs.sph_mean_velocity_coeffs_kernels.sph_mean_velocity_coeffs`.
    The kernel arguments (``periodic1`` to ``scale``) are those of
    :func:`~struphy.pic.sph_eval_kernels.box_based_physical`.
    """
    rho_val = zeros(1, dtype=float)
    rho_grad = zeros((1, 3), dtype=float)
    v_val = zeros(3, dtype=float)
    grad_v = zeros((3, 3), dtype=float)
    sigma = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[0]
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        x1 = markers[x_idx, ip]
        x2 = markers[x_idx + 1, ip]
        x3 = markers[x_idx + 2, ip]
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        loc_box = int(markers[n_cols - 2, ip])

        # density
        sph_eval_kernels.box_based_columns(
            args_markers,
            x1,
            x2,
            x3,
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
            period,
            proj,
            x_idx,
            weight_idx,
            1,
            kernel,
            dim,
            h,
            scale,
            rho_val,
            rho_grad,
        )

        # Cartesian velocity gradient grad_v[j, m] = d v_j / d x_m
        sph_eval_kernels.box_based_columns(
            args_markers,
            x1,
            x2,
            x3,
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
            period,
            proj,
            x_idx,
            first_free_idx,
            3,
            kernel,
            dim,
            h,
            scale,
            v_val,
            grad_v,
        )

        div_v = grad_v[0, 0] + grad_v[1, 1] + grad_v[2, 2]
        factor = -mu * markers[weight_idx, ip] / rho_val[0]
        for j in range(3):
            for m in range(3):
                sigma[j, m] = grad_v[j, m] + grad_v[m, j]
            sigma[j, j] -= 2.0 / 3.0 * div_v

        for j in range(3):
            for m in range(3):
                if output_indices[3 * j + m] >= 0:
                    markers[output_indices[3 * j + m], ip] = factor * sigma[j, m]
