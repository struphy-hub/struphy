"Pusher kernel for gyro-center (5D) dynamics."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2", "dfm", "unit_b1", "e_star", "tmp", "k")
def gk_push_drift_gyro(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    n_gyro: int,
    ring_s: "float[:]",
    ring_w: "float[:]",
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
    B_tilde_par: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of an s-stage explicit Runge-Kutta scheme for the gyro-averaged perpendicular drifts

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1}_p \times \hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}}\,,
        \qquad \hat{\mathbf E}^{*1}_p = \langle \hat{\mathbf e}^1 \rangle_p - \varepsilon \mu_p \hat\nabla \left(B_0 + \langle\langle \tilde B_\parallel \rangle\rangle_p\right)\,,

    with :math:`\hat B^*_\parallel = |B_0| + \langle \tilde B_\parallel \rangle_p + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}`
    (same field arguments as
    :func:`~struphy.pic.accumulation.kernels.gk_drift_gyro_vector.gk_drift_gyro_vector_kernels.gk_drift_gyro_vector`,
    plus the electric field ``e1_1, e1_2, e1_3`` and the Butcher coefficients ``a, b, c``).
    """

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_pusher_idx = args_markers.first_pusher_idx

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    unit_b1 = empty(3, dtype=float)
    e_star = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    k = empty(3, dtype=float)

    n_stages = shape(b)[0]
    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        if markers[ip, first_pusher_idx] == -1.0:
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
        eval_1form_spline_mpi(span1, span2, span3, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, unit_b1)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, gradB1_1, gradB1_2, gradB1_3, e_star)
        curl_b_par = eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta1, eta2, eta3, args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        # mirror force with the disc-averaged parallel perturbation
        gyroaverage_kernels.eval_1form_gyro(
            eta1,
            eta2,
            eta3,
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
        e_star += tmp
        e_star *= -epsilon * mu

        # gyro-averaged electric field
        gyroaverage_kernels.eval_1form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, e1_1, e1_2, e1_3, tmp
        )
        e_star += tmp

        b_star_para = b0 + epsilon * v * curl_b_par
        b_star_para += gyroaverage_kernels.eval_0form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, B_tilde_par
        )
        b_star_para *= det_df

        linalg_kernels.cross(e_star, unit_b1, k)
        k /= b_star_para

        pusher_utilities_kernels.update_position_stage(ip, k, dt, a[stage], b[stage], last, args_markers)
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
