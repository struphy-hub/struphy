"Pusher kernel for SPH (6D) particles."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("grad_u", "grad_u_cart", "tmp1", "dfinv", "dfinvT")
def push_v_sph_pressure(
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
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    gravity: "float[:]",
    kappa: "float",
):
    r"""Update each marker :math:`p` according to

    .. math::

        \frac{\mathbf v_p^{n+1} - \mathbf v_p^n}{\Delta t} = \mathbf g - \sum_{i=1}^N w_i \left( \frac{\kappa}{\rho^{N,h}(\boldsymbol \eta_p)} + \frac{\kappa}{\rho^{N,h}(\boldsymbol \eta_i)} \right) DF^{-\top}\nabla W_h(\boldsymbol \eta_p - \boldsymbol \eta_i) \,,

    where :math:`\mathbf g` is a constant acceleration, the second term corresponds to the pressure gradient
    in the isothermal closure (with constant :math:`\kappa`), and :math:`DF^{-\top}` denotes the inverse transpose Jacobian
    arising in the pull back of the gradient of the smoothing kernel :math:`W_h`
    chosen from :mod:`~struphy.pic.sph_smoothing_kernels`.

    The smoothed SPH density is given by

    .. math::

        \rho^{N,h}(\boldsymbol \eta_p) = \sum_j w_j \, W_h(\boldsymbol \eta_p - \boldsymbol \eta_j)\,.

    This kernel requires:

    * The density :math:`\rho^{N,h}(\boldsymbol \eta_p)` to be pre-computed for each particle and stored at ``markers[:, first_free_idx]``)
    * The coefficient :math:`w_i/\rho^{N,h}(\boldsymbol \eta_i)` to be pre-computed for each particle and stored at ``markers[:, first_free_idx + 1]``)

    This is accomplished by the kernel :func:`~struphy.pic.pushing.eval_kernels_sph.sph_pressure_coeffs`, which needs
    to be passed as an ``init_kernel`` to the :class:`~struphy.pic.pushing.pusher.Pusher`.

    Parameters
    ----------
    boxes : 2d array
        Box array of the sorting boxes structure.

    neighbours : 2d array
        Array containing the 27 neighbouring boxes of each box.

    holes : bool
        1D array of length markers.shape[0]. True if markers[i] is a hole.

    periodic1, periodic2, periodic3 : bool
        True if periodic in that dimension.

    kernel_type : int
        Number of the smoothing kernel.

    h1, h2, h3 : float
        Kernel width in respective dimension.

    gravity: xp.ndarray
        Constant gravitational force as 3-vector.

    kappa: float
        Constant isothermal coefficient.
    """
    # allocate arrays
    grad_u = zeros(3, dtype=float)
    grad_u_cart = zeros(3, dtype=float)
    tmp1 = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[1]

    # fmt: off
    #$ omp parallel firstprivate(grad_u) private(ip, eta1, eta2, eta3, n_at_eta, loc_box, sum2, sum4, sum6, grad_u_cart, tmp1, dfinv, dfinvT)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        n_at_eta = markers[ip, first_free_idx]
        loc_box = int(markers[ip, n_cols - 2])

        # first component
        grad_u[0] = sph_eval_kernels.box_based_kernel(
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
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        grad_u[0] *= kappa / n_at_eta

        sum2 = sph_eval_kernels.box_based_kernel(
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
            first_free_idx + 1,
            kernel_type + 1,
            h1,
            h2,
            h3,
        )
        sum2 *= kappa

        grad_u[0] += sum2

        if kernel_type >= 340:
            # second component
            grad_u[1] = sph_eval_kernels.box_based_kernel(
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
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            grad_u[1] *= kappa / n_at_eta

            sum4 = sph_eval_kernels.box_based_kernel(
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
                first_free_idx + 1,
                kernel_type + 2,
                h1,
                h2,
                h3,
            )
            sum4 *= kappa
            grad_u[1] += sum4

        if kernel_type >= 670:
            # third component
            grad_u[2] = sph_eval_kernels.box_based_kernel(
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
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            grad_u[2] *= kappa / n_at_eta

            sum6 = sph_eval_kernels.box_based_kernel(
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
                first_free_idx + 1,
                kernel_type + 3,
                h1,
                h2,
                h3,
            )
            sum6 *= kappa
            grad_u[2] += sum6

        # push to Cartesian coordinates
        evaluation_kernels.df_inv(
            eta1,
            eta2,
            eta3,
            args_domain,
            tmp1,
            False,
            dfinv,
        )
        linalg_kernels.transpose(dfinv, dfinvT)
        linalg_kernels.matrix_vector(dfinvT, grad_u, grad_u_cart)

        # update velocities
        markers[ip, 3:6] -= dt * (grad_u_cart - gravity)
