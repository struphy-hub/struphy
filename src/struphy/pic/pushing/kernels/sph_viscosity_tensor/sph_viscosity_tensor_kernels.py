"SPH marker evaluation. Output indices are absolute marker columns; -1 skips a component."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.sph_eval_kernels as sph_eval_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
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
    kernel_type: "int",
    h1: "float",
    h2: "float",
    h3: "float",
    mu: "float",
):
    r"""For each particle, evaluate the smoothed SPH density :math:`\rho^{N,h}(\boldsymbol \eta_i)` and the
    deviatoric strain rate, and store the 9 coefficients

    * :math:`- w_i \, \sqrt g(\boldsymbol \eta_i) \, \sigma_{jm}(\boldsymbol \eta_i) \, (DF^{-1})_{km}(\boldsymbol \eta_i)
      / \rho^{N,h}(\boldsymbol \eta_i)` at ``markers[:, output_indices[3*j + k]]`` for :math:`j, k = 0, 1, 2`

    where :math:`\sqrt g = \det DF`, the smoothed SPH density is given by

    .. math::

        \rho^{N,h}(\boldsymbol \eta_i) = \sum_l w_l \, W_h(\boldsymbol \eta_i - \boldsymbol \eta_l)\,,

    and the deviatoric strain rate is the traceless symmetric part of the Cartesian mean velocity gradient,

    .. math::

        \sigma_{jm}(\boldsymbol \eta_i)
        = \mu\bigl[ \partial_{x_m} v_j^{N,h}(\boldsymbol \eta_i) + \partial_{x_j} v_m^{N,h}(\boldsymbol \eta_i)
        - \tfrac{2}{3}\delta_{jm} \, \partial_{x_l} v_l^{N,h}(\boldsymbol \eta_i)\bigr]\,,
        \qquad
        \partial_{x_m} v_j^{N,h} = \sum_k \partial_{\eta_k} v_j^{N,h} \, (DF^{-1})_{km}\,.

    The factor :math:`\sqrt g \, DF^{-1}` puts the tensor in divergence (Piola) form,
    :math:`(\nabla_x \cdot \sigma)_j = \frac{1}{\sqrt g} \partial_{\eta_k} \bigl(\sqrt g \, (DF^{-1})_{km} \sigma_{jm}\bigr)`,
    such that these coefficients serve as kernel weights to evaluate the viscous force

    .. math::

        (-\nabla \cdot \Pi_{\textrm{vis}})^{N,h}_j(\boldsymbol \eta_i)
        = -\frac{1}{\sqrt g(\boldsymbol \eta_i)} \sum_l \frac{ w_l \, \sqrt g(\boldsymbol \eta_l) \,
          \sigma_{jm}(\boldsymbol \eta_l) \, (DF^{-1})_{km}(\boldsymbol \eta_l)}{\rho^{N,h}(\boldsymbol \eta_l)} \,
          (\nabla W_h)_k(\boldsymbol \eta_i - \boldsymbol \eta_l)\,.

    This kernel requires the coefficients of the mean velocity :math:`v_k^{N,h}`
    for each particle to be pre-evaluated and stored at ``markers[:, first_free_idx:first_free_idx + 3]``,
    which can be achieved by the kernel :func:`~struphy.pic.pushing.eval_kernels_sph.sph_mean_velocity_coeffs`.
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    weight_idx = args_markers.weight_idx
    first_free_idx = args_markers.first_free_idx
    valid_mks = args_markers.valid_mks

    grad_v_at_eta = zeros((3, 3), dtype=float)
    grad_v_cart = zeros((3, 3), dtype=float)
    # d_tensor = zeros((3, 3), dtype=float)
    d_dev = zeros((3, 3), dtype=float)
    d_piola = zeros((3, 3), dtype=float)
    df_mat = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvT = zeros((3, 3), dtype=float)
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
        for j in range(3):
            for k in range(3):
                grad_v_at_eta[j, k] = sph_eval_kernels.box_based_kernel(
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
                    first_free_idx + j,
                    kernel_type + 1 + k,
                    h1,
                    h2,
                    h3,
                )

        # Cartesian velocity gradient: d v_j / d x_m = sum_k d v_j / d eta_k * (DF^{-1})_km
        evaluation_kernels.df_inv(
            eta1,
            eta2,
            eta3,
            args_domain,
            df_mat,
            False,
            dfinv,
        )
        detdf = linalg_kernels.det(df_mat)
        linalg_kernels.matrix_matrix(grad_v_at_eta, dfinv, grad_v_cart)

        d_dev[:] = 0.5 * (grad_v_cart + grad_v_cart.T)

        mean_trace = (d_dev[0, 0] + d_dev[1, 1] + d_dev[2, 2]) / 3.0

        d_dev[0, 0] -= mean_trace
        d_dev[1, 1] -= mean_trace
        d_dev[2, 2] -= mean_trace

        d_dev *= -2 * mu * (weight / n_at_eta)

        # Piola form for the divergence in logical coordinates: sqrt(g) * sigma_jm * (DF^{-1})_km
        linalg_kernels.transpose(dfinv, dfinvT)
        linalg_kernels.matrix_matrix(d_dev, dfinvT, d_piola)
        d_piola *= detdf

        for j in range(3):
            for k in range(3):
                if output_indices[3 * j + k] >= 0:
                    markers[ip, output_indices[3 * j + k]] = d_piola[j, k]
