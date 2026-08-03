"Initialization routines (initial guess, evaluations) for sph kernel evaluations."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels  # noqa: PLR0402
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d  # noqa: PLR0402
import struphy.geometry.evaluation_kernels as evaluation_kernels  # noqa: PLR0402
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies  # noqa: PLR0402
import struphy.linear_algebra.linalg_kernels as linalg_kernels  # noqa: PLR0402
import struphy.pic.sph_eval_kernels as sph_eval_kernels  # noqa: PLR0402
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_pressure_coeffs(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
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

    * the density :math:`\rho^{N,h}(\boldsymbol \eta_i)` abd stored it at ``markers[:, column_nr]``)
    * the coefficient :math:`w_i/\rho^{N,h}(\boldsymbol \eta_i)` and stored it at ``markers[:, column_nr + 1]``)
    * the coefficient :math:`w_i (\rho^{N,h}(\boldsymbol \eta_i))^{\gamma - 2}` and stored it at ``markers[:, column_nr + 2]``)

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
        markers[ip, column_nr] = n_at_eta
        markers[ip, column_nr + 1] = weight / n_at_eta
        markers[ip, column_nr + 2] = weight * n_at_eta ** (gamma - 2)


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_isotherm_kappa(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
    args_markers: "MarkerArguments",
):
    r"""None yet."""

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_diagnostic_idx = args_markers.first_diagnostics_idx

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        markers[ip, first_diagnostic_idx] = 1.0


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_mean_velocity_coeffs(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
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
    r"""For each particle, evaluate the smoothed SPH density :math:`\rho^{N,h}(\boldsymbol \eta_i)` and store the
    coefficient

    * :math:`w_i v_{k,i} / \rho^{N,h}(\boldsymbol \eta_i)` at ``markers[:, column_nr + k]`` for :math:`k = 0, 1, 2`

    where the smoothed SPH density is given by

    .. math::

        \rho^{N,h}(\boldsymbol \eta_i) = \sum_j w_j \, W_h(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.

    These coefficients serve as kernel weights so that one can evaluate
    the SPH mean velocity field

    .. math::

        v_k^{N,h}(\boldsymbol \eta_i)
        = \sum_j \frac{w_j \, v_{k,j}}{\rho^{N,h}(\boldsymbol \eta_j)} \,
          W_h(\boldsymbol \eta_i - \boldsymbol \eta_j)\,.
    """

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    n_cols = shape(markers)[1]
    weight_idx = args_markers.weight_idx
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is a "true" particle
        if not valid_mks[ip]:
            continue

        # also evaluate and save for ghost particles, only skip holes (!)
        # if holes[ip]:
        #     continue

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
        velocities = markers[ip, 3:6]
        # save
        markers[ip, column_nr] = weight / n_at_eta * velocities[0]
        markers[ip, column_nr + 1] = weight / n_at_eta * velocities[1]
        markers[ip, column_nr + 2] = weight / n_at_eta * velocities[2]

        # logger.info(f"{ip = }, {weight = }, {n_at_eta = }, {velocities[0] = }")


# @stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
# def sph_mean_velocity(
#     alpha: "float[:]",
#     column_nr: int,
#     comps: "int[:]",
#     args_markers: "MarkerArguments",
#     args_domain: "DomainArguments",
#     boxes: "int[:, :]",
#     neighbours: "int[:, :]",
#     holes: "bool[:]",
#     periodic1: "bool",
#     periodic2: "bool",
#     periodic3: "bool",
#     kernel_type: "int",
#     h1: "float",
#     h2: "float",
#     h3: "float",
# ):
#     r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

#     .. math::

#         H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
#         + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

#     that is

#     .. math::

#         \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
#         + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

#     where the evaluation point is the weighted average
#     :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
#     for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
#     :math:`\boldsymbol \eta_p` beforehand.

#     The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
#     in markers array for each particle.
#     """

#     gamma = 5 / 3

#     # get marker arguments
#     markers = args_markers.markers
#     n_markers = args_markers.n_markers
#     n_cols = shape(markers)[1]
#     Np = args_markers.Np
#     vdim = args_markers.vdim
#     weight_idx = args_markers.weight_idx
#     first_free_idx = args_markers.first_free_idx
#     valid_mks = args_markers.valid_mks

#     for ip in range(n_markers):
#         # only do something if particle is a "true" particle
#         if not valid_mks[ip]:
#             continue

#         eta1 = markers[ip, 0]
#         eta2 = markers[ip, 1]
#         eta3 = markers[ip, 2]
#         loc_box = int(markers[ip, n_cols - 2])
#         v1_at_eta = sph_eval_kernels.box_based_kernel(
#             args_markers,
#             eta1,
#             eta2,
#             eta3,
#             loc_box,
#             boxes,
#             neighbours,
#             holes,
#             periodic1,
#             periodic2,
#             periodic3,
#             first_free_idx,
#             kernel_type,
#             h1,
#             h2,
#             h3,
#         )

#         v2_at_eta = sph_eval_kernels.box_based_kernel(
#             args_markers,
#             eta1,
#             eta2,
#             eta3,
#             loc_box,
#             boxes,
#             neighbours,
#             holes,
#             periodic1,
#             periodic2,
#             periodic3,
#             first_free_idx + 1,
#             kernel_type,
#             h1,
#             h2,
#             h3,
#         )

#         v3_at_eta = sph_eval_kernels.box_based_kernel(
#             args_markers,
#             eta1,
#             eta2,
#             eta3,
#             loc_box,
#             boxes,
#             neighbours,
#             holes,
#             periodic1,
#             periodic2,
#             periodic3,
#             first_free_idx + 2,
#             kernel_type,
#             h1,
#             h2,
#             h3,
#         )
#         # save
#         markers[ip, column_nr] = v1_at_eta
#         markers[ip, column_nr + 1] = v2_at_eta
#         markers[ip, column_nr + 2] = v3_at_eta


# @stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
# def sph_grad_mean_velocity(
#     alpha: "float[:]",
#     column_nr: int,
#     comps: "int[:]",
#     args_markers: "MarkerArguments",
#     args_domain: "DomainArguments",
#     boxes: "int[:, :]",
#     neighbours: "int[:, :]",
#     holes: "bool[:]",
#     periodic1: "bool",
#     periodic2: "bool",
#     periodic3: "bool",
#     kernel_type: "int",
#     h1: "float",
#     h2: "float",
#     h3: "float",
# ):
#     r"""Evaluate the :math:`\boldsymbol \eta`-gradient of the Hamiltonian

#     .. math::

#         H(\mathbf Z_p) = H(\boldsymbol \eta_p, v_{\parallel,p}) = \varepsilon \frac{v_{\parallel,p}^2}{2}
#         + \varepsilon \mu |\hat \mathbf B| (\boldsymbol \eta_p) + \hat \phi(\boldsymbol \eta_p)\,,

#     that is

#     .. math::

#         \hat \nabla H(\mathbf Z_p) = \varepsilon \mu \hat \nabla |\hat \mathbf B| (\boldsymbol \eta_p)
#         + \hat \nabla \hat \phi(\boldsymbol \eta_p)\,,

#     where the evaluation point is the weighted average
#     :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`,
#     for :math:`i=1,2,3,4`. Markers must be sorted according to the evaluation point
#     :math:`\boldsymbol \eta_p` beforehand.

#     The components specified in ``comps`` are save at ``column_nr:column_nr + len(comps)``
#     in markers array for each particle.
#     """

#     gamma = 5 / 3

#     # get marker arguments
#     markers = args_markers.markers
#     n_markers = args_markers.n_markers
#     n_cols = shape(markers)[1]
#     Np = args_markers.Np
#     vdim = args_markers.vdim
#     weight_idx = args_markers.weight_idx
#     first_free_idx = args_markers.first_free_idx
#     valid_mks = args_markers.valid_mks

#     grad_v_at_eta = zeros((3, 3), dtype=float)
#     for ip in range(n_markers):
#         # only do something if particle is a "true" particle
#         if not valid_mks[ip]:
#             continue

#         eta1 = markers[ip, 0]
#         eta2 = markers[ip, 1]
#         eta3 = markers[ip, 2]
#         loc_box = int(markers[ip, n_cols - 2])
#         for j in range(3):
#             for k in range(3):
#                 grad_v_at_eta[j, k] = sph_eval_kernels.box_based_kernel(
#                     args_markers,
#                     eta1,
#                     eta2,
#                     eta3,
#                     loc_box,
#                     boxes,
#                     neighbours,
#                     holes,
#                     periodic1,
#                     periodic2,
#                     periodic3,
#                     first_free_idx + j,
#                     kernel_type + 1 + k,
#                     h1,
#                     h2,
#                     h3,
#                 )

#                 # save
#                 markers[ip, column_nr + 3 * j + k] = grad_v_at_eta[j, k]


@stack_array("eta_k", "eta_n", "eta", "grad_H", "e_field")
def sph_viscosity_tensor(
    alpha: "float[:]",
    column_nr: int,
    comps: "int[:]",
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

    * :math:`- w_i \, \sigma_{jk}(\boldsymbol \eta_i) / \rho^{N,h}(\boldsymbol \eta_i)` at
      ``markers[:, column_nr + 3*j + k]`` for :math:`j, k = 0, 1, 2`

    where the smoothed SPH density is given by

    .. math::

        \rho^{N,h}(\boldsymbol \eta_i) = \sum_l w_l \, W_h(\boldsymbol \eta_i - \boldsymbol \eta_l)\,,

    and the deviatoric strain rate is the traceless symmetric part of the mean velocity gradient,

    .. math::

        \sigma_{jk}(\boldsymbol \eta_i)
        = \mu\bigl[ \partial_j v_k^{N,h}(\boldsymbol \eta_i) + \partial_k v_j^{N,h}(\boldsymbol \eta_i)
        - \tfrac{2}{3}\delta_{jk}\bigr] \, \partial_l v_l^{N,h}(\boldsymbol \eta_i)\,.

    These coefficients serve as kernel weights so that one can evaluate the viscous force

    .. math::

        (-\nabla \cdot \Pi_{\textrm{vis}})^{N,h}_j(\boldsymbol \eta_i)
        = \sum_l \frac{ w_l \, \sigma_{jk}(\boldsymbol \eta_l)}{\rho^{N,h}(\boldsymbol \eta_l)} \,
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
    # d_tensor = zeros((3, 3), dtype=float)
    d_dev = zeros((3, 3), dtype=float)
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

        d_dev[:] = 0.5 * (grad_v_at_eta + grad_v_at_eta.T)

        mean_trace = (d_dev[0, 0] + d_dev[1, 1] + d_dev[2, 2]) / 3.0

        d_dev[0, 0] -= mean_trace
        d_dev[1, 1] -= mean_trace
        d_dev[2, 2] -= mean_trace

        d_dev *= -2 * mu * (weight / n_at_eta)

        for j in range(3):
            for k in range(3):
                markers[ip, column_nr + 3 * j + k] = d_dev[j, k]
