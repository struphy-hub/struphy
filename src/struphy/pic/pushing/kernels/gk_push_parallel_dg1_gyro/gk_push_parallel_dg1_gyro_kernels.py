"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, mod, sqrt
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "tmp", "b_star", "k", "d1", "d2")
def gk_push_parallel_dg1_gyro(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    absB0: "float[:,:,:]",
    gradB1_1: "float[:,:,:]",
    gradB1_2: "float[:,:,:]",
    gradB1_3: "float[:,:,:]",
    grad_B_tilde_par_1: "float[:,:,:]",
    grad_B_tilde_par_2: "float[:,:,:]",
    grad_B_tilde_par_3: "float[:,:,:]",
):
    r"""One Picard iteration of the first-order discrete-gradient (Gonzalez) scheme for the parallel gyrocentre
    motion with finite Larmor radius, see
    :func:`~struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_1st_order.push_gc_Bstar_discrete_gradient_1st_order_kernels.push_gc_Bstar_discrete_gradient_1st_order`.
    The Hamiltonian is

    .. math::

        H(\mathbf Z) = \varepsilon \frac{v_\parallel^2}{2} + \varepsilon \mu_p \left(|B_0| + \langle\langle \tilde B_\parallel \rangle\rangle\right)(\boldsymbol \eta)\,,

    whose gradient at the mid-point uses the disc average of the gradient of the parallel magnetic perturbation.
    The Hamiltonian at :math:`n` and at :math:`(n+1,k)` and the effective field at :math:`n` are read from the
    marker columns ``first_free_idx`` (:math:`H_n`), ``+1`` (:math:`\hat B^{*3}_\parallel`), ``+2:+5``
    (:math:`\hat{\mathbf B}^{*2}`) and ``+5`` (:math:`H_k`), filled by
    :func:`~struphy.pic.pushing.kernels.gk_hamiltonian_gyro.gk_hamiltonian_gyro_kernels.gk_hamiltonian_gyro` and
    :func:`~struphy.pic.pushing.kernels.gk_bstar_gyro.gk_bstar_gyro_kernels.gk_bstar_gyro`.
    """

    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)
    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    k = empty(3, dtype=float)
    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_pusher_idx = args_markers.first_pusher_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_pusher_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_pusher_idx : first_pusher_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        v_k = markers[ip, 3]
        v_n = markers[ip, first_pusher_idx + 3]
        v_mid = (v_k + v_n) / 2.0
        v_diff = v_k - v_n

        mu = markers[ip, mu_idx]

        H_n = markers[ip, first_free_idx]
        b_star_parallel = epsilon * markers[ip, first_free_idx + 1]
        b_star[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]
        H_k = markers[ip, first_free_idx + 5]

        # grad H at the mid-point: equilibrium part and disc-averaged perturbation
        span1, span2, span3 = get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, gradB1_1, gradB1_2, gradB1_3, grad_H)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta_mid[0], eta_mid[1], eta_mid[2], args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )
        gyroaverage_kernels.eval_1form_gyro(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            d1,
            d2,
            rho,
            n_gyro,
            disc_s,
            disc_w,
            args_derham,
            grad_B_tilde_par_1,
            grad_B_tilde_par_2,
            grad_B_tilde_par_3,
            tmp,
        )
        grad_H += tmp
        grad_H *= epsilon * mu

        grad_H_v = epsilon * v_mid

        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H) + v_diff * grad_H_v
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff) + v_diff * v_diff

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
            grad_I_v = grad_H_v
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared
            grad_I_v = grad_H_v + v_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        k[:] = b_star / b_star_parallel * grad_I_v

        k_v = linalg_kernels.scalar_dot(b_star, grad_I)
        k_v /= -b_star_parallel

        markers[ip, 0:3] = eta_n + dt * k
        markers[ip, 3] = v_n + dt * k_v

        v_scale = abs(v_k)
        if v_scale == 0.0:
            v_scale = 1.0

        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2
            + (markers[ip, 1] - eta_k[1]) ** 2
            + (markers[ip, 2] - eta_k[2]) ** 2
            + ((markers[ip, 3] - v_k) / v_scale) ** 2,
        )

        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
