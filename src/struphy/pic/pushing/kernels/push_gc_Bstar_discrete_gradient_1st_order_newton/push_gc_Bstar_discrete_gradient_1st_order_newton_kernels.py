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
    "b_star",
    "e_field",
    "grad_I",
    "J_vec",
    "Ddg",
    "DdgT",
    "func",
    "B",
    "C",
    "A_inv",
    "k",
)
def push_gc_Bstar_discrete_gradient_1st_order_newton(
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

        \left\{ 
            \begin{aligned} 
                \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} &= 
                \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g \,\hat B^{*}_\parallel} (\mathbf Z_p^{n}) 
                \frac{\partial \overline H}{\partial v_{\parallel}} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
                \\
                \frac{v_{\parallel,p}^{n+1,k+1} - v_{\parallel,p}^{n}}{\Delta t} &= 
                - \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^{*}_\parallel} (\mathbf Z_p^{n}) \cdot
                \frac{\partial \overline H}{\partial \boldsymbol \eta} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
            \end{aligned}
        \right.

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2} 
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \mathbf Z}
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n})
        = \begin{pmatrix}
        \frac{H(\eta_{p,1}^{n+1}) - H}{\eta_{p,1}^{n+1} - \eta_{p,1}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}) - H(\eta_{p,1}^{n+1})}{\eta_{p,2}^{n+1} - \eta_{p,2}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1})}{\eta_{p,3}^{n+1} - \eta_{p,3}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}, v_{\parallel, p}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1})}{v_{\parallel, p}^{n+1} - v_{\parallel,p}^n}
        \end{pmatrix}\,,

    is the Itoh-Abe discrete gradient. The Newton algorithm searches the roots of

    .. math::

        \mathbf F(\mathbf Z_p^{n+1}) = \mathbf Z_p^{n+1} - \mathbf Z_p^{n} 
        - \Delta t \mathbb J (\mathbf Z_p^{n}) \frac{\partial \overline H}{\partial \mathbf Z} 
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n}) = 0\,,

    via (iteration index :math:`k`)

    .. math::

        \mathbf Z^{n+1, k+1} = \mathbf Z_p^{n+1, k} 
        - D\mathbf F^{-1}(\mathbf Z_p^{n+1, k}) \mathbf F( \mathbf Z_p^{n+1, k})\,,

    where the Jacobian is given by

    .. math::

        D\mathbf F(\boldsymbol \eta_p^{n+1, k}) = \mathbb I_{3\times 3} 
        - \Delta t \mathbb J (\mathbf Z_p^{n}) 
        D\frac{\partial \overline H}{\partial \mathbf Z} 
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n})\,.

    Notes
    -----
    This kernel performs evaluations at :math:`\mathbf Z_p^{n+1, k}`. 
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

    b_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    grad_I = zeros(3, dtype=float)
    J_vec = empty(3, dtype=float)
    Ddg = zeros((3, 3), dtype=float)
    DdgT = zeros((3, 3), dtype=float)
    func = zeros(3, dtype=float)
    B = empty(3, dtype=float)
    C = empty(3, dtype=float)
    A_inv = zeros((3, 3), dtype=float)

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

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v_diff = v_k - v_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n
        b_star_parallel = epsilon * markers[ip, first_free_idx + 1]
        b_star[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

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

        H_k = epsilon * v_k**2 / 2.0 + epsilon * mu * B_dot_b + phi
        H_k123 = epsilon * v_n**2 / 2.0 + epsilon * mu * B_dot_b + phi

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
        grad_H_v = epsilon * v_k

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
            grad_I[2] = (H_k123 - H_k12) / (eta_diff[2])

        if v_diff == 0.0:
            grad_I_v = grad_H_v
        else:
            grad_I_v = (H_k - H_k123) / (v_diff)

        # compute F; the Poisson matrix is [[0, Jvec], [-Jvec^T, 0]]
        J_vec[:] = b_star / b_star_parallel
        func[:] = J_vec * grad_I_v
        func *= -dt
        func += eta_diff

        func_v = linalg_kernels.scalar_dot(J_vec, grad_I)
        func_v *= -1.0
        func_v *= -dt
        func_v += v_diff

        # compute the Jacobian of the discrete gradient; it has the form [[Ddg, 0], [0, Ddg_v]]
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
            Ddg[2, 2] = (grad_H[2] * eta_diff[2] - (H_k123 - H_k12)) / eta_diff[2] ** 2
            Ddg[2, 0] = (grad_H[0] - grad_H_12[0]) / eta_diff[2]
            Ddg[2, 1] = (grad_H[1] - grad_H_12[1]) / eta_diff[2]

        if v_diff == 0.0:
            Ddg_v = 0.0
        else:
            Ddg_v = (grad_H_v * v_diff - (H_k - H_k123)) / v_diff**2

        # the matrix DF is [[I_3x3, -dt*J_vec*Ddg_v], [dt*(Ddg^T*J_vec)^T, 1]]
        # we compute its inverse with the Schur complement of [[I_3x3, B], [C, 1]]
        # block matrix B
        B[:] = J_vec
        B *= Ddg_v
        B *= -dt
        # block matrix C
        linalg_kernels.transpose(Ddg, DdgT)
        linalg_kernels.matrix_vector(DdgT, J_vec, C)
        C *= dt
        # Schur complement M/A
        schur_comp = 1.0 - linalg_kernels.scalar_dot(C, B)
        # inverse blocks
        linalg_kernels.outer(B, C, A_inv)
        A_inv /= schur_comp
        A_inv[0, 0] += 1.0
        A_inv[1, 1] += 1.0
        A_inv[2, 2] += 1.0

        B /= -schur_comp
        C /= -schur_comp

        # update
        linalg_kernels.matrix_vector(A_inv, func, k)
        k += B * func_v
        k_v = linalg_kernels.scalar_dot(C, func)
        k_v += func_v / schur_comp

        markers[ip, 0:3] -= k
        markers[ip, 3] -= k_v

        # residual (relative in v, absolute if v_k == 0)
        v_scale = abs(v_k)
        if v_scale == 0.0:
            v_scale = 1.0

        markers[ip, residual_idx] = sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2 + (k_v / v_scale) ** 2)

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, True)
