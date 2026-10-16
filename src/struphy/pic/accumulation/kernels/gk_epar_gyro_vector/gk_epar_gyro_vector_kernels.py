"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2", "dfm", "b2", "b_star", "c")
def gk_epar_gyro_vector(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
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
    r"""Gyro-averaged parallel current of :class:`~struphy.propagators.gyrokinetic_maxwell.GyroEparCoupling`,
    deposited into V1 with the filling

    .. math::

        B_p^\mu = w_p\, v_{\parallel,p}\, c_{p,\mu} \langle \vec\Lambda^1 \rangle_p\,,
        \qquad \mathbf c_p = \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p)\,,

    where :math:`\hat{\mathbf B}^{*2} = \hat{\mathbf B}^2_0 + \langle \hat{\tilde{\mathbf B}}^2 \rangle_p + \varepsilon v_\parallel \hat\nabla \times \hat{\mathbf b}^1_0`
    (the FLR-corrected magnetic field :math:`\mathbf B_1^\star = \langle \mathbf B_1 \rangle`) and
    :math:`\hat B^*_\parallel = |B_0| + \langle \tilde B_\parallel \rangle_p + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}`.

    Parameters
    ----------
        epsilon, n_gyro, ring_s, ring_w, unit_b1_1, unit_b1_2, unit_b1_3, absB0 :
            Gyro-averaging arguments (``ring_s = [1.0]``, ``ring_w = [1.0]`` for the gyro-average),
            see :mod:`struphy.pic.gyroaverage_kernels`.

        b2_eq_1, b2_eq_2, b2_eq_3 : array[float]
            FE coefficients of the equilibrium magnetic field as a 2-form.

        b2_tilde_1, b2_tilde_2, b2_tilde_3 : array[float]
            FE coefficients of the magnetic perturbation as a 2-form.

        curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3 : array[float]
            FE coefficients of the curl of the equilibrium unit magnetic field as a 2-form.

        B_tilde_par : array[float]
            FE coefficients of the parallel magnetic perturbation as a 0-form.

        curl_unit_b_dot_b0 : array[float]
            FE coefficients of (curl b_0) . b_0 as a 0-form.
    """

    markers = args_markers.markers

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    c = empty(3, dtype=float)

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
        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta1, eta2, eta3, args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        # B* = B_0 + <B_tilde> + eps v curl b_0 (2-form)
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

        # B*_parallel (times det DF)
        b_star_para = b0 + epsilon * v * eval_0form_spline_mpi(span1, span2, span3, args_derham, curl_unit_b_dot_b0)
        b_star_para += gyroaverage_kernels.eval_0form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, B_tilde_par
        )
        b_star_para *= det_df

        c[:] = b_star / b_star_para

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
            weight * v * c[0],
            weight * v * c[1],
            weight * v * c[2],
        )
