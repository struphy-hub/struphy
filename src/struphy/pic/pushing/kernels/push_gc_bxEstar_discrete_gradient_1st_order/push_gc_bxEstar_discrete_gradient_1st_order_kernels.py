"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, mod, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "unit_b1", "e_field", "Exb", "k")
def push_gc_bxEstar_discrete_gradient_1st_order(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Picard iteration (index :math:`k`) for

    .. math::

        \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} =
        \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n} )  \,,

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2}
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})
        = \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)
        + (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \,
        \frac{H(\mathbf Z_p^{n+1, k}) - H(\mathbf Z_p^{n}) - (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \cdot
        \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)}{||\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}||}\,,

    is the Gonzalez discrete gradient.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)

    unit_b1 = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    Exb = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n (from init_kernel)
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n (from init_kernel)
        b_star_parallel = markers[ip, first_free_idx + 1]
        unit_b1[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at (n+1, k) (from eval_kernel)
        H_k = markers[ip, first_free_idx + 5]

        # mid-point spline evaluation
        span1, span2, span3 = get_spans(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_derham,
        )

        # compute grad_H at n + 1/2
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute grad_I
        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H)
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff)

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        # calculate k
        linalg_kernels.cross(unit_b1, grad_I, Exb)

        k[:] = Exb / b_star_parallel

        # accumulation for last stage
        markers[ip, 0:3] = eta_n + dt * k

        # residual
        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2 + (markers[ip, 1] - eta_k[1]) ** 2 + (markers[ip, 2] - eta_k[2]) ** 2,
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
