"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "dfm",
    "df_inv_t",
    "df_inv",
    "g_inv",
    "filling_v",
    "tmp",
    "tmp_v",
    "b",
    "b_prod",
    "norm_b_prod",
    "b_star",
    "curl_norm_b",
    "norm_b1",
    "grad_PB",
    "grad_PBeq",
)
def cc_lin_mhd_5d_gradB(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11: "float[:,:,:,:,:,:]",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat22: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    mat33: "float[:,:,:,:,:,:]",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,
    ep_scale: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    grad_PB1: "float[:,:,:]",
    grad_PB2: "float[:,:,:]",
    grad_PB3: "float[:,:,:]",
    grad_PBeq1: "float[:,:,:]",
    grad_PBeq2: "float[:,:,:]",
    grad_PBeq3: "float[:,:,:]",
    basis_u: "int",
):
    r"""Accumulation kernel for the propagator :class:`~struphy.propagators.propagators_coupling.CurrentCoupling5DGradB`.

    Accumulates math:`\alpha` -form vector with the filling functions

    .. math::

        B_p^\mu &= \omega_p \left[\left(\frac{\mu_p}{\sqrt{g}\hat B^*_\parallel}\right) \mathbf B^2_{\times} G^{-1} \mathbf b^2_{0 \times} G^{-1} \nabla B_\parallel¹\right]_\mu \,,

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b1, b2, b3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of normalized magnetic field as a 2-form.

        grad_PB1, grad_PB2, grad_PB3 : array[float]
            FE coefficients c_ijk of gradient of parallel magnetic field as a 1-form.

        grad_PBeq1, grad_PBeq2, grad_PBeq3 : array[float]
            FE coefficients c_ijk of gradient of equilibrium parallel magnetic field as a 1-form;
            added to grad_PB for all u-spaces.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    mu_idx = args_markers.mu_idx

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b_prod = zeros((3, 3), dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_PB = empty(3, dtype=float)
    grad_PBeq = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)
    tmp = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # if particle is refilled
        if markers[ip, first_init_idx] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # needed metric coefficients
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # grad_PBeq; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PBeq1, grad_PBeq2, grad_PBeq3, grad_PBeq)

        # b_star; 2form transformed into H1vec
        b_star[:] = b + curl_norm_b * v * epsilon

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        # full gradient of the parallel magnetic field (perturbation + equilibrium)
        grad_PB += grad_PBeq

        if basis_u == 0:
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * ep_scale

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para / det_df * ep_scale

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )
