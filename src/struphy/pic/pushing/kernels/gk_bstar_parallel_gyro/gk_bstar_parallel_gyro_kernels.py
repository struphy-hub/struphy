"Initialization routine (initial guess, evaluations) for 5D gyro-center pusher kernels."

from numpy import empty, mod
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("eta_k", "eta_n", "eta", "d1", "d2", "dfm", "b2", "b_star")
def gk_bstar_parallel_gyro(
    alpha: "float[:]",
    output_indices: "int[:]",
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    n_gyro: int,
    ring_s: "float[:]",
    ring_w: "float[:]",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    absB0: "float[:,:,:]",
    b2_eq_1: "float[:,:,:]",
    b2_eq_2: "float[:,:,:]",
    b2_eq_3: "float[:,:,:]",
    b2_tilde_1: "float[:,:,:]",
    b2_tilde_2: "float[:,:,:]",
    b2_tilde_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    B_tilde_par: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
):
    r"""Evaluate the effective magnetic field with the gyro-averaged perturbation,

    .. math::

        \hat B^{*3}_\parallel = \sqrt g \left(|B_0| + \langle \tilde B_\parallel \rangle + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}\right)\,,
        \qquad \hat{\mathbf B}^{*2} = \hat{\mathbf B}^2_0 + \langle \hat{\tilde{\mathbf B}}^2 \rangle + \varepsilon v_\parallel \hat\nabla \times \hat{\mathbf b}^1_0\,,

    at the weighted average :math:`Z_{p,i} = \alpha_i Z_{p,i}^{n+1,k} + (1 - \alpha_i) Z_{p,i}^n`.
    The result :math:`\hat B^{*3}_\parallel` is saved at ``output_indices[0]`` in the marker array; -1 skips the output
    (see :func:`~struphy.pic.pushing.kernels.gk_bstar_gyro.gk_bstar_gyro_kernels.gk_bstar_gyro` for :math:`\hat{\mathbf B}^{*2}`).
    """

    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta = empty(3, dtype=float)
    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

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

        evaluation_kernels.df(eta[0], eta[1], eta[2], args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta[0], eta[1], eta[2], args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        gyroaverage_kernels.eval_2form_gyro(
            eta[0],
            eta[1],
            eta[2],
            d1,
            d2,
            rho,
            n_gyro,
            ring_s,
            ring_w,
            args_derham,
            b2_tilde_1,
            b2_tilde_2,
            b2_tilde_3,
            b_star,
        )
        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b2_eq_1, b2_eq_2, b2_eq_3, b2)
        b_star += b2
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3, b2)
        b_star += epsilon * v * b2

        b_star_para = b0 + epsilon * v * eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)
        b_star_para += gyroaverage_kernels.eval_0form_gyro(
            eta[0], eta[1], eta[2], d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, B_tilde_par
        )
        b_star_para *= det_df

        if output_indices[0] >= 0:
            markers[ip, output_indices[0]] = b_star_para
