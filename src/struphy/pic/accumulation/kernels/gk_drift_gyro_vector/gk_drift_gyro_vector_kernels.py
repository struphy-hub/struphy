"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2", "dfm", "unit_b1", "grad_b", "tmp", "k")
def gk_drift_gyro_vector(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    scale: float,
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
):
    r"""Gyro-averaged grad-B drift current of :class:`~struphy.propagators.gyrokinetic_maxwell.GyroDriftCoupling`,
    deposited into V1 with the filling

    .. math::

        B_p^\mu = s\, w_p\, \dot{\boldsymbol \eta}_{\nabla B, p} \langle \vec\Lambda^1 \rangle_p\,,
        \qquad \dot{\boldsymbol \eta}_{\nabla B, p} = - \varepsilon \mu_p \frac{\hat\nabla (B_0 + \langle\langle \tilde B_\parallel \rangle\rangle_p) \times \hat{\mathbf b}^1_0}{\sqrt g\, \hat B^*_\parallel}\,,

    with the disc-averaged parallel magnetic perturbation in the mirror force and the gyro-averaged one in
    :math:`\hat B^*_\parallel = |B_0| + \langle \tilde B_\parallel \rangle_p + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}`.

    Parameters
    ----------
        scale : float
            Scaling of the current of the species (:math:`\alpha^2/\varepsilon`).

        gradB1_1, gradB1_2, gradB1_3 : array[float]
            FE coefficients of the gradient of the equilibrium field strength as a 1-form.

        grad_B_tilde_par_1, grad_B_tilde_par_2, grad_B_tilde_par_3 : array[float]
            FE coefficients of the gradient of the parallel magnetic perturbation as a 1-form.

        B_tilde_par : array[float]
            FE coefficients of the parallel magnetic perturbation as a 0-form.

        curl_unit_b_dot_b0 : array[float]
            FE coefficients of (curl b_0) . b_0 as a 0-form.
    """

    markers = args_markers.markers

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    unit_b1 = empty(3, dtype=float)
    grad_b = empty(3, dtype=float)
    tmp = empty(3, dtype=float)
    k = empty(3, dtype=float)

    for ip in range(shape(markers)[0]):
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, args_markers.mu_idx]
        weight = markers[ip, 5]

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, unit_b1)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, gradB1_1, gradB1_2, gradB1_3, grad_b)
        curl_b_par = eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta1, eta2, eta3, args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        # grad of B_0 + <<B_tilde_par>> (disc average)
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
        grad_b += tmp

        # B*_parallel (times det DF) with the gyro-averaged perturbation
        b_star_para = b0 + epsilon * v * curl_b_par
        b_star_para += gyroaverage_kernels.eval_0form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, B_tilde_par
        )
        b_star_para *= det_df

        linalg_kernels.cross(grad_b, unit_b1, k)
        fac = -scale * weight * epsilon * mu / b_star_para

        gyroaverage_kernels.deposit_1form_gyro(
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
            vec1,
            vec2,
            vec3,
            fac * k[0],
            fac * k[1],
            fac * k[2],
        )
