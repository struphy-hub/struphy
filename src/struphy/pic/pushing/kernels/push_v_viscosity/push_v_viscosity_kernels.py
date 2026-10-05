"Pusher kernel for SPH (6D) particles."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("tmp1")
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
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
):
    r"""Update each marker :math:`p` according to

    .. math::

        \frac{v_{p,j}^{n+1} - v_{p,j}^n}{\Delta t}
        = \frac{1}{\sqrt g(\boldsymbol \eta_p)} \sum_{i=1}^N
          \frac{w_i \, \sqrt g(\boldsymbol \eta_i) \, \sigma_{jm}(\boldsymbol \eta_i) \,
          (DF^{-1})_{km}(\boldsymbol \eta_i)}{\rho^{N,h}(\boldsymbol \eta_i)} \,
          (\nabla W_h)_k(\boldsymbol \eta_p - \boldsymbol \eta_i)\,,

    where :math:`\sigma_{jm} = \mu \left[(\partial_{x_m} v_j^{N,h} + \partial_{x_j} v_m^{N,h}) - \tfrac{2}{3}\delta_{jm}\partial_{x_l} v_l^{N,h}\right]`
    is the (Cartesian) deviatoric strain rate and :math:`\sqrt g = \det DF`. This is the SPH discretization
    of the Cartesian divergence :math:`\nabla_x \cdot \sigma` written in logical coordinates (Piola form),
    :math:`(\nabla_x \cdot \sigma)_j = \frac{1}{\sqrt g} \partial_{\eta_k} \bigl(\sqrt g \, (DF^{-1})_{km} \sigma_{jm}\bigr)`,
    with the gradient of the smoothing kernel :math:`W_h` chosen from :mod:`~struphy.pic.sph_smoothing_kernels`.

    This kernel requires the 9 coefficients

    * :math:`- w_i \, \sqrt g(\boldsymbol \eta_i) \, \sigma_{jm}(\boldsymbol \eta_i) \, (DF^{-1})_{km}(\boldsymbol \eta_i)
      / \rho^{N,h}(\boldsymbol \eta_i)` to be
      pre-computed for each particle and stored at ``markers[:, first_free_idx + 3*(j+1) + k]``
      for :math:`j, k = 0, 1, 2`

    This is accomplished by the kernel
    :func:`~struphy.pic.pushing.eval_kernels_sph.sph_viscosity_tensor`, which itself requires
    the mean velocity coefficients
    :math:`w_i v_{k,i} / \rho^{N,h}(\boldsymbol \eta_i)` to be stored at
    ``markers[:, first_free_idx:first_free_idx + 3]`` via
    :func:`~struphy.pic.pushing.eval_kernels_sph.sph_mean_velocity_coeffs`.
    Both kernels must be passed as ``init_kernel`` entries to the
    :class:`~struphy.pic.pushing.pusher.Pusher`.

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
    """
    # allocate arrays
    tmp1 = zeros((3, 3), dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks
    n_cols = shape(markers)[1]
    f_visc = zeros(3, dtype=float)

    # fmt: off
    #$ omp parallel private(ip, eta1, eta2, eta3, loc_box, j, k, coeff_idx, f_visc, f_visc_cart, tmp1, dfinv, dfinvT)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        loc_box = int(markers[ip, n_cols - 2])

        f_visc[:] = 0.0
        for j in range(3):  # row of viscosity tensor
            for k in range(3):  # column = derivative direction
                coeff_idx = first_free_idx + 3 * (j + 1) + k

                # if k == 0:
                #     deriv_type = kernel_type + 1
                #     use_component = True
                # elif k == 1 and kernel_type >= 340:
                #     deriv_type = kernel_type + 2
                #     use_component = True
                # elif k == 2 and kernel_type >= 670:
                #     deriv_type = kernel_type + 3
                #     use_component = True
                # else:
                #     use_component = False

                # if use_component:
                f_visc[j] += sph_eval_kernels.box_based_kernel(
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
                    coeff_idx,
                    kernel_type + 1 + k,
                    h1,
                    h2,
                    h3,
                )

        # Cartesian divergence from Piola form: divide by the Jacobian determinant
        detdf = evaluation_kernels.det_df(
            eta1,
            eta2,
            eta3,
            args_domain,
            tmp1,
        )

        # update velocities
        markers[ip, 3:6] -= dt * f_visc / detdf
