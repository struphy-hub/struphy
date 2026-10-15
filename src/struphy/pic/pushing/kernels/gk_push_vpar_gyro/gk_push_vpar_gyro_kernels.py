"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2", "dfm", "b2", "b_star", "e_avg")
def gk_push_vpar_gyro(
    dt: float,
    stage: int,
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
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
):
    r"""Updates the parallel velocity of each marker with the gyro-averaged electric field,

    .. math::

        \frac{v_{\parallel,p}^{n+1} - v_{\parallel,p}^n}{\Delta t} = \frac{1}{\varepsilon}\, \mathbf c_p \cdot \langle \hat{\mathbf e}^1 \rangle_p\,,

    with :math:`\mathbf c_p` (evaluated with :math:`v_{\parallel,p}^n`) exactly as in
    :func:`~struphy.pic.accumulation.kernels.gk_epar_gyro_vector.gk_epar_gyro_vector_kernels.gk_epar_gyro_vector`
    (same arguments).
    """

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    e_avg = empty(3, dtype=float)

    for ip in range(n_markers):
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, args_markers.mu_idx]

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)
        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta1, eta2, eta3, args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        gyroaverage_kernels.eval_2form_gyro(
            eta1,
            eta2,
            eta3,
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
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b2_eq_1, b2_eq_2, b2_eq_3, b2)
        b_star += b2
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3, b2)
        b_star += epsilon * v * b2

        b_star_para = b0 + epsilon * v * eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)
        b_star_para += gyroaverage_kernels.eval_0form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, B_tilde_par
        )
        b_star_para *= det_df

        gyroaverage_kernels.eval_1form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, e1_1, e1_2, e1_3, e_avg
        )

        markers[ip, 3] += dt / epsilon * linalg_kernels.scalar_dot(b_star, e_avg) / b_star_para
