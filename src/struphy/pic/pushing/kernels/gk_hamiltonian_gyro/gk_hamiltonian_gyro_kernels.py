"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "d1", "d2")
def gk_hamiltonian_gyro(
    alpha: "float[:]",
    output_indices: "int[:]",
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
    B_tilde_par: "float[:,:,:]",
):
    r"""Evaluate the gyrocentre Hamiltonian with the disc-averaged parallel magnetic perturbation

    .. math::

        H(\mathbf Z_p) = \varepsilon \frac{v_{\parallel,p}^2}{2} + \varepsilon \mu_p \left(|B_0| + \langle\langle \tilde B_\parallel \rangle\rangle\right)(\boldsymbol \eta_p)\,,

    at the weighted average :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`.
    The result is saved at ``output_indices[0]`` in the marker array; -1 skips the output.
    """

    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_pusher_idx = args_markers.first_pusher_idx
    first_shift_idx = args_markers.first_shift_idx

    for ip in range(n_markers):
        if markers[ip, 0] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_pusher_idx : first_pusher_idx + 3]

        eta[:] = alpha[:3] * eta_k + (1.0 - alpha[:3]) * eta_n
        eta[:] = mod(eta, 1.0)

        v_k = markers[ip, 3]
        v_n = markers[ip, first_pusher_idx + 3]
        v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n

        mu = markers[ip, mu_idx]

        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta[0], eta[1], eta[2], args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        b_par = b0 + gyroaverage_kernels.eval_0form_gyro(
            eta[0], eta[1], eta[2], d1, d2, rho, n_gyro, disc_s, disc_w, args_derham, B_tilde_par
        )

        if output_indices[0] >= 0:
            markers[ip, output_indices[0]] = epsilon * v**2 / 2.0 + epsilon * mu * b_par
