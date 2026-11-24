"Pusher kernel for SPH (6D) particles."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("rho_val", "rho_grad", "s_val", "s_grad")
def push_v_viscosity(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    boxes: "int[:,:]",
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
    r"""Update each marker :math:`p` according to

    .. math::

        \frac{v_{p,j}^{n+1} - v_{p,j}^n}{\Delta t}
        = -\frac{1}{\rho_p} \sum_{l} \frac{w_l \, \sigma_{jm,l}}{\rho_l} \, \partial_m W_h(\mathbf x_p - \mathbf x_l)\,,

    the SPH discretization of :math:`-\frac{1}{\rho} \nabla \cdot \sigma` with the viscous stress (momentum flux)
    :math:`\sigma_{jm} = -\mu \left[\partial_m v_j^{N,h} + \partial_j v_m^{N,h} - \tfrac{2}{3}\delta_{jm}\nabla \cdot
    \mathbf v^{N,h}\right]`, the physical kernel :math:`W_h` and the SPH density
    :math:`\rho_p = \sum_l w_l W_h(\mathbf x_p - \mathbf x_l)`.

    This kernel requires the 9 coefficients :math:`w_l \sigma_{jm,l}/\rho_l` at
    ``markers[first_free_idx + 3*(j+1) + m, :]`` for :math:`j, m = 0, 1, 2`, computed by the init kernel
    :func:`~struphy.pic.pushing.kernels.sph_viscosity_tensor.sph_viscosity_tensor_kernels.sph_viscosity_tensor`,
    which itself requires the mean-velocity coefficients at ``markers[first_free_idx:first_free_idx + 3, :]``
    (:func:`~struphy.pic.pushing.kernels.sph_mean_velocity_coeffs.sph_mean_velocity_coeffs_kernels.sph_mean_velocity_coeffs`).

    Parameters
    ----------
    boxes, neighbours : 2d array
        Sorting-box arrays.

    holes : bool
        1D array of length markers.shape[1]. True if markers[:, i] is a hole.

    periodic1, periodic2, periodic3, period, proj, x_idx, kernel, dim, h, scale
        Kernel arguments, see :func:`~struphy.pic.sph_eval_kernels.box_based_physical`.
    """
    rho_val = zeros(1, dtype=float)
    rho_grad = zeros((1, 3), dtype=float)
    s_val = zeros(9, dtype=float)
    s_grad = zeros((9, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_free_idx = args_markers.first_free_idx
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[0]

    # fmt: off
    #$ omp parallel firstprivate(rho_val, rho_grad, s_val, s_grad) private(ip, loc_box, j, div_s)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        loc_box = int(markers[n_cols - 2, ip])

        # density
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
            rho_val,
            rho_grad,
        )

        # gradients of the 9 stress coefficients
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
            first_free_idx + 3,
            9,
            kernel,
            dim,
            h,
            scale,
            s_val,
            s_grad,
        )

        # momentum equation rho du/dt = -div(sigma)
        for j in range(3):
            div_s = s_grad[3 * j, 0] + s_grad[3 * j + 1, 1] + s_grad[3 * j + 2, 2]
            markers[3 + j, ip] -= dt * div_s / rho_val[0]

    # fmt: off
    #$ omp end parallel
    # fmt: on
