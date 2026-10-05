"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "eta_k",
    "eta_n",
    "eta_k_shifted",
    "eta_diff",
    "grad_H_12",
    "grad_H",
    "unit_b1",
    "e_field",
    "grad_I",
    "Ddg",
    "bcross_mat",
    "func",
    "Dfunc",
    "Dfunc_inv",
    "k",
)
def push_gc_bxEstar_dg_1st_order_newton(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    phi_coeffs: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Newton iteration for

    .. math::

        \frac{\boldsymbol \eta_p^{n+1} - \boldsymbol \eta_p^{n}}{\Delta t} =
        \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n} )  \,,

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2}
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n})
        = \begin{pmatrix}
        \frac{H(\eta_{p,1}^{n+1}) - H}{\eta_{p,1}^{n+1} - \eta_{p,1}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}) - H(\eta_{p,1}^{n+1})}{\eta_{p,2}^{n+1} - \eta_{p,2}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1})}{\eta_{p,3}^{n+1} - \eta_{p,3}^n}
        \end{pmatrix}\,,

    is the Itoh-Abe discrete gradient. The Newton algorithm searches the roots of

    .. math::

        \mathbf F(\boldsymbol \eta_p^{n+1}) = \boldsymbol \eta_p^{n+1} - \boldsymbol \eta_p^{n}
        - \Delta t \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n}) = 0\,,

    via (iteration index :math:`k`)

    .. math::

        \boldsymbol \eta_p^{n+1, k+1} = \boldsymbol \eta_p^{n+1, k}
        - D\mathbf F^{-1}(\boldsymbol \eta_p^{n+1, k}) \mathbf F( \boldsymbol \eta_p^{n+1, k})\,,

    where the Jacobian is given by

    .. math::

        D\mathbf F(\boldsymbol \eta_p^{n+1, k}) = \mathbb I_{3\times 3}
        - \Delta t \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times
        D\frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n})\,.

    Notes
    -----
    This kernel performs evaluations at :math:`\boldsymbol \eta_p^{n+1, k}`.
    Other evaluations are performed in ``init_kernels`` and ``eval_kernels``,
    respectively.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_k_shifted = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H_12 = empty(2, dtype=float)
    grad_H = empty(3, dtype=float)

    unit_b1 = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    grad_I = zeros(3, dtype=float)
    Ddg = zeros((3, 3), dtype=float)
    bcross_mat = zeros((3, 3), dtype=float)
    func = zeros(3, dtype=float)
    Dfunc = zeros((3, 3), dtype=float)
    Dfunc_inv = zeros((3, 3), dtype=float)

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

        eta_k[:] = markers[ip, 0:3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]
        eta_k_shifted[:] = eta_k + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_diff[:] = eta_k_shifted - eta_n

        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # Hamiltonian at n
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n
        b_star_parallel = markers[ip, first_free_idx + 1]
        unit_b1[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k)
        H_k1 = markers[ip, first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        H_k12 = markers[ip, first_free_idx + 6]

        # 1st comp of gradient of Hamiltonian at eta_1^(n+1, k)
        grad_H_1 = markers[ip, first_free_idx + 7]

        # 1st and 2nd comps of gradient of Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        grad_H_12[:] = markers[ip, first_free_idx + 8 : first_free_idx + 10]

        # evaluate H at (n+1, k)
        span1, span2, span3 = get_spans(eta_k[0], eta_k[1], eta_k[2], args_derham)

        if evaluate_e_field:
            phi = eval_0form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                phi_coeffs,
            )
        else:
            phi = 0.0

        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        H_k = epsilon * v**2 / 2.0 + epsilon * mu * B_dot_b + phi

        # compute grad_H at (n+1, k)
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

        # compute the Itoh discrete gradient
        if eta_diff[0] == 0.0:
            grad_I[0] = grad_H[0]
        else:
            grad_I[0] = (H_k1 - H_n) / (eta_diff[0])

        if eta_diff[1] == 0.0:
            grad_I[1] = grad_H[1]
        else:
            grad_I[1] = (H_k12 - H_k1) / (eta_diff[1])

        if eta_diff[2] == 0.0:
            grad_I[2] = grad_H[2]
        else:
            grad_I[2] = (H_k - H_k12) / (eta_diff[2])

        # compute matrix for cross product
        bcross_mat[0, 1] = -unit_b1[2]
        bcross_mat[0, 2] = unit_b1[1]
        bcross_mat[1, 0] = unit_b1[2]
        bcross_mat[1, 2] = -unit_b1[0]
        bcross_mat[2, 0] = -unit_b1[1]
        bcross_mat[2, 1] = unit_b1[0]
        bcross_mat /= b_star_parallel

        # compute F
        linalg_kernels.matrix_vector(bcross_mat, grad_I, func)
        func *= -dt
        func += eta_diff

        # compute the Jacobian of the discrete gradient
        if eta_diff[0] == 0.0:
            Ddg[0, 0] = 0.0
        else:
            Ddg[0, 0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / eta_diff[0] ** 2

        if eta_diff[1] == 0.0:
            Ddg[1, 1] = 0.0
            Ddg[1, 0] = 0.0
        else:
            Ddg[1, 1] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / eta_diff[1] ** 2
            Ddg[1, 0] = (grad_H_12[0] - grad_H_1) / eta_diff[1]

        if eta_diff[2] == 0.0:
            Ddg[2, 2] = 0.0
            Ddg[2, 0] = 0.0
            Ddg[2, 1] = 0.0
        else:
            Ddg[2, 2] = (grad_H[2] * eta_diff[2] - (H_k - H_k12)) / eta_diff[2] ** 2
            Ddg[2, 0] = (grad_H[0] - grad_H_12[0]) / eta_diff[2]
            Ddg[2, 1] = (grad_H[1] - grad_H_12[1]) / eta_diff[2]

        # compute Jacobian matrix DF
        linalg_kernels.matrix_matrix(bcross_mat, Ddg, Dfunc)
        Dfunc *= -dt
        Dfunc[0, 0] += 1.0
        Dfunc[1, 1] += 1.0
        Dfunc[2, 2] += 1.0

        # comute inverse and update
        linalg_kernels.matrix_inv(Dfunc, Dfunc_inv)
        linalg_kernels.matrix_vector(Dfunc_inv, func, k)

        markers[ip, 0:3] -= k

        # residual
        markers[ip, residual_idx] = sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, True)
