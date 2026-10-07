"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape, zeros
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


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "tmp1", "tmp2", "b", "b_prod", "bstar", "norm_b1", "curl_norm_b")
def cc_lin_mhd_5d_D(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    epsilon: float,
    ep_scale: "float",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    basis_u: "int",
):
    r"""Accumulation kernel for the propagator :class:`~struphy.propagators.propagators_fields.CurrentCoupling5DDensity`.

    Accumulates :math:`\alpha`-form matrix with the filling functions (:math:`\alpha = 2`)

    .. math::

        A_p^{\mu, \nu} = w_p \frac{1}{\epsilon} \left( 1-\frac{\hat B_\parallel}{\hat B^*_\parallel} \right)  g^{-1} (\mathbf B^2_\times)_{\mu, \nu} \,.

    Parameters
    ----------
        epsilon : float
            scaling factor.

        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        norm_b11, norm_b12, norm_b12 : array[float]
            FE coefficients c_ijk of the unit magnetic field as a 1-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of the unit magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # allocate some temporary buffers for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, boundary_cut, eta1, eta2, eta3, v, weight, span1, span2, span3, b2_1, b2_2, b2_3, b, b_para, curl_norm_b, b_star, norm_b1, b_star_para, density_const, dfm, df_inv, df_inv_t, g_inv, det_df, tmp1, tmp2, filling_m12, filling_m13, filling_m23)
    # -- removed omp: #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v = markers[ip, 3]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_2form_spline_mpi(span1, span2, span3, args_derham, b2_1, b2_2, b2_3, b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian matrix and Jacobian determinant
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # calculate Bstar and transform to H1vec
        b_star[:] = b + epsilon * v * curl_norm_b

        # calculate b_para and b_star_para
        b_para = linalg_kernels.scalar_dot(norm_b1, b)

        b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate scaling constant
        density_const = 1 - b_para / b_star_para

        # marker weight
        weight = markers[ip, 5]

        if basis_u == 0:
            # filling functions
            filling_m12 = -weight * density_const * b_prod[0, 1] * ep_scale / epsilon
            filling_m13 = -weight * density_const * b_prod[0, 2] * ep_scale / epsilon
            filling_m23 = -weight * density_const * b_prod[1, 2] * ep_scale / epsilon

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v0vec_asym(
                args_derham,
                span1,
                span2,
                span3,
                mat12,
                mat13,
                mat23,
                filling_m12,
                filling_m13,
                filling_m23,
            )

        elif basis_u == 1:
            # filling functions
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)

            filling_m12 = -weight * density_const * tmp2[0, 1] * ep_scale / epsilon
            filling_m13 = -weight * density_const * tmp2[0, 2] * ep_scale / epsilon
            filling_m23 = -weight * density_const * tmp2[1, 2] * ep_scale / epsilon

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v1_asym(
                args_derham,
                span1,
                span2,
                span3,
                mat12,
                mat13,
                mat23,
                filling_m12,
                filling_m13,
                filling_m23,
            )

        elif basis_u == 2:
            # filling functions
            filling_m12 = -weight * density_const * b_prod[0, 1] * ep_scale / epsilon / det_df**2
            filling_m13 = -weight * density_const * b_prod[0, 2] * ep_scale / epsilon / det_df**2
            filling_m23 = -weight * density_const * b_prod[1, 2] * ep_scale / epsilon / det_df**2

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v2_asym(
                args_derham,
                span1,
                span2,
                span3,
                mat12,
                mat13,
                mat23,
                filling_m12,
                filling_m13,
                filling_m23,
            )
