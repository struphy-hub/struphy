"Pusher kernel for SPH (6D) particles."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("val_w", "grad_w", "val_c", "grad_c")
def push_v_sph_pressure_ideal_gas(
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
    gravity: "float[:]",
    kappa: "float",
    gamma: "float",
):
    r"""Update each marker :math:`p` according to

    .. math::

        \frac{\mathbf v_p^{n+1} - \mathbf v_p^n}{\Delta t} = \mathbf g - \kappa \sum_{j} w_j \left( \rho_p^{\gamma - 2}
        + \rho_j^{\gamma - 2} \right) \nabla W_h(\mathbf x_p - \mathbf x_j)\,,

    the pressure force of the polytropic closure (constant :math:`\kappa`, adiabatic index :math:`\gamma`) plus a
    constant acceleration :math:`\mathbf g`, with the physical kernel :math:`W_h` and the SPH density
    :math:`\rho_p = \sum_j w_j W_h(\mathbf x_p - \mathbf x_j)`.

    This kernel requires :math:`\rho_p` at ``markers[first_free_idx, :]`` and :math:`w_j \rho_j^{\gamma - 2}` at
    ``markers[first_free_idx + 2, :]``, computed by the init kernel
    :func:`~struphy.pic.pushing.kernels.sph_pressure_coeffs.sph_pressure_coeffs_kernels.sph_pressure_coeffs`.

    Parameters
    ----------
    boxes, neighbours : 2d array
        Sorting-box arrays.

    holes : bool
        1D array of length markers.shape[1]. True if markers[:, i] is a hole.

    periodic1, periodic2, periodic3, period, proj, x_idx, kernel, dim, h, scale
        Kernel arguments, see :func:`~struphy.pic.sph_eval_kernels.box_based_physical`.

    gravity: xp.ndarray
        Constant acceleration as 3-vector.

    kappa: float
        Polytropic coefficient.

    gamma: float
        Adiabatic index of the polytropic closure.
    """
    val_w = zeros(1, dtype=float)
    grad_w = zeros((1, 3), dtype=float)
    val_c = zeros(1, dtype=float)
    grad_c = zeros((1, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[0]

    # fmt: off
    #$ omp parallel firstprivate(val_w, grad_w, val_c, grad_c) private(ip, loc_box, rho_fac, k)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        rho_fac = markers[first_free_idx, ip] ** (gamma - 2.0)
        loc_box = int(markers[n_cols - 2, ip])

        # sum_j w_j grad W
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
            val_w,
            grad_w,
        )

        # sum_j w_j rho_j^(gamma - 2) grad W
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
            first_free_idx + 2,
            1,
            kernel,
            dim,
            h,
            scale,
            val_c,
            grad_c,
        )

        # update velocities
        for k in range(3):
            markers[3 + k, ip] -= dt * (kappa * (rho_fac * grad_w[0, k] + grad_c[0, k]) - gravity[k])

    # fmt: off
    #$ omp end parallel
    # fmt: on
