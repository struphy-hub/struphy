"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2", "dfm", "b2", "b_star", "c", "e_avg")
def gk_epar_gyro_matvec(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
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
    r"""Matrix-free product of the gyro-averaged parallel coupling matrix of
    :class:`~struphy.propagators.gyrokinetic_maxwell.GyroEparCoupling` with a 1-form coefficient vector ``e``:

    .. math::

        \mathbf v \mathrel{+}= \sum_p w_p \langle \vec\Lambda^1 \rangle_p\, \mathbf c_p \left(\mathbf c_p \cdot \langle \vec\Lambda^{1\top} \rangle_p \mathbf e\right)\,,

    with :math:`\mathbf c_p` as in :func:`~struphy.pic.accumulation.kernels.gk_epar_gyro_vector.gk_epar_gyro_vector_kernels.gk_epar_gyro_vector`
    (same arguments). The scaling is applied in the propagator.
    """

    markers = args_markers.markers

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    c = empty(3, dtype=float)
    e_avg = empty(3, dtype=float)

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

        c[:] = b_star / b_star_para

        # gyro-averaged electric field (1-form) and its projection on c
        gyroaverage_kernels.eval_1form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, ring_s, ring_w, args_derham, e1_1, e1_2, e1_3, e_avg
        )
        s = weight * linalg_kernels.scalar_dot(c, e_avg)

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
            s * c[0],
            s * c[1],
            s * c[2],
        )
