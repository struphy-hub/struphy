"""Accumulation kernels for gyro-center (5D) particles.

Function naming conventions:

* use the model name, all lower-case letters (e.g. ``lin_vlasov_maxwell``)
* in case of multiple accumulations in one model, attach ``_1``, ``_2`` or the species name.

These kernels are passed to :class:`struphy.pic.accumulation.particles_to_grid.Accumulator`.
"""

from numpy import empty, shape, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels

# do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def gc_density_0form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
):
    r"""
    Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V0 with the filling

    .. math::

        B_p^\mu = \frac{w_p}{N} \,.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, filling)
    # -- removed omp: #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # filling = w_p/N
        filling = markers[ip, 5] / Np

        particle_to_mat_kernels.vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, filling)

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "tmp1", "tmp2", "b", "b_prod", "bstar", "norm_b1", "curl_norm_b")
def cc_lin_mhd_5d_D(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    epsilon: float,  # model specific argument
    b2_1: "float[:,:,:]",  # model specific argument
    b2_2: "float[:,:,:]",  # model specific argument
    b2_3: "float[:,:,:]",  # model specific argument
    # model specific argument
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    # model specific argument
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    boundary_cut: float,
):  # model specific argument
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
    Np = args_markers.Np

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

        if eta1 < boundary_cut or eta1 > 1.0 - boundary_cut:
            continue

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
        b_star /= det_df

        # calculate b_para and b_star_para
        b_para = linalg_kernels.scalar_dot(norm_b1, b)
        b_para /= det_df

        b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate scaling constant
        density_const = 1 - b_para / b_star_para

        # marker weight
        weight = markers[ip, 5]

        if basis_u == 0:
            # filling functions
            filling_m12 = -weight * density_const * b_prod[0, 1] * scale_mat
            filling_m13 = -weight * density_const * b_prod[0, 2] * scale_mat
            filling_m23 = -weight * density_const * b_prod[1, 2] * scale_mat

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v0vec_asym(
                args_derham, span1, span2, span3, mat12, mat13, mat23, filling_m12, filling_m13, filling_m23
            )

        elif basis_u == 1:
            # filling functions
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)

            filling_m12 = -weight * density_const * tmp2[0, 1] * scale_mat
            filling_m13 = -weight * density_const * tmp2[0, 2] * scale_mat
            filling_m23 = -weight * density_const * tmp2[1, 2] * scale_mat

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v1_asym(
                args_derham, span1, span2, span3, mat12, mat13, mat23, filling_m12, filling_m13, filling_m23
            )

        elif basis_u == 2:
            # filling functions
            filling_m12 = -weight * density_const * b_prod[0, 1] * scale_mat / det_df**2
            filling_m13 = -weight * density_const * b_prod[0, 2] * scale_mat / det_df**2
            filling_m23 = -weight * density_const * b_prod[1, 2] * scale_mat / det_df**2

            # call the appropriate matvec filler
            particle_to_mat_kernels.mat_fill_v2_asym(
                args_derham, span1, span2, span3, mat12, mat13, mat23, filling_m12, filling_m13, filling_m23
            )

    # -- removed omp: #$ omp end parallel

    mat12 /= Np
    mat13 /= Np
    mat23 /= Np


@stack_array(
    "dfm",
    "df_inv_t",
    "df_inv",
    "g_inv",
    "filling_m",
    "filling_v",
    "tmp",
    "tmp1",
    "tmp2",
    "tmp_m",
    "tmp_v",
    "b",
    "b_prod",
    "b_prod_negb_star",
    "norm_b1",
    "curl_norm_b",
)
def cc_lin_mhd_5d_J1(
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
    epsilon: float,  # model specific argument
    b1: "float[:,:,:]",  # model specific argument
    b2: "float[:,:,:]",  # model specific argument
    b3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_mat: "float",  # model specific argument
    scale_vec: "float",  # model specific argument
    boundary_cut: "float",
):  # model specific argument
    r"""Accumulation kernel for the propagator :class:`~struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb`.

    Accumulates :math:`\alpha`-form matrix and vector with the filling functions (:math:`\alpha = 2`)

    .. math::

        A_p^{\mu, \nu} &= w_p \left[\left( \frac{v_{\parallel,p}}{g\hat B^*_\parallel}\right)^2  \mathbf B^2_{\times} \left| \hat \nabla \times \hat{\mathbf b}^1_0 \right|^2 (\mathbf B^2_{\times})^\top \right]_{\mu, \nu}\,,

        B_p^\mu &= w_p \left( \frac{v^2_{\parallel,p}}{g\hat B^*_\parallel} \mathbf B^2_{\times} \right)_\mu \,,

    where :math:`\mathbf B^2_{\times} \mathbf a := \hat{\mathbf B}^2 \times \mathbf a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b1, b2, b3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of normalized magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    b_prod_neg = zeros((3, 3), dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp = empty((3, 3), dtype=float)
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, boundary_cut, eta1, eta2, eta3, v, weight, span1, span2, span3, b1, b2, b3, b, b_star, b_prod_neg, norm_b1, curl_norm_b, abs_b_star_para, dfm, df_inv, df_inv_t, g_inv, det_df, tmp, tmp1, tmp2, tmp_m, tmp_v, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]

        if eta1 < boundary_cut or eta1 > 1.0 - boundary_cut:
            continue

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate tensor product of two curl_norm_b
        linalg_kernels.outer(curl_norm_b, curl_norm_b, tmp)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        b_prod_neg[:] = -1.0 * b_prod

        if basis_u == 0:
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v0vec_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )

        elif basis_u == 1:
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_vector(tmp1, curl_norm_b, tmp_v)

            linalg_kernels.matrix_matrix(tmp1, tmp, tmp2)
            linalg_kernels.matrix_matrix(tmp2, b_prod_neg, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp_m)

            filling_m[:, :] = weight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v1_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )

        elif basis_u == 2:
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = weight * tmp_m * v**2 / abs_b_star_para**2 / det_df**4 * scale_mat
            filling_v[:] = weight * tmp_v * v**2 / abs_b_star_para / det_df**2 * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.m_v_fill_v2_symm(
                args_derham,
                span1,
                span2,
                span3,
                mat11,
                mat12,
                mat13,
                mat22,
                mat23,
                mat33,
                filling_m[0, 0],
                filling_m[0, 1],
                filling_m[0, 2],
                filling_m[1, 1],
                filling_m[1, 2],
                filling_m[2, 2],
                vec1,
                vec2,
                vec3,
                filling_v[0],
                filling_v[1],
                filling_v[2],
            )

    mat11 /= Np
    mat12 /= Np
    mat13 /= Np
    mat22 /= Np
    mat23 /= Np
    mat33 /= Np

    vec1 /= Np
    vec2 /= Np
    vec3 /= Np

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "norm_b1", "filling_v")
def cc_lin_mhd_5d_M(
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
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    scale_vec: "float",  # model specific argument
    boundary_cut: "float",
):  # model specific argument
    r"""Accumulation kernel for the propagator :class:`~struphy.propagators.propagators_fields.ShearAlfvenCurrentCoupling5D` and :class:`~struphy.propagators.propagators_fields.MagnetosonicCurrentCoupling5D`.

    Accumulates 2-form vector with the filling functions:

    .. math::

        B^\mu_p = \omega_p \mu_p\left(\sqrt{g}^{-1} \hat{\mathbf{b}}¹_0\right)_\mu \,.

    Parameters
    ----------

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for a field evaluation
    norm_b1 = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel private(ip, boundary_cut, eta1, eta2, eta3, mu, weight, norm_b1, dfm, det_df, span1, span2, span3, norm_b11, norm_b12, norm_b13, filling_v)
    # -- removed omp: #$ omp for reduction ( + : vec1, vec2, vec3)

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, 9]

        if eta1 < boundary_cut or eta1 > 1.0 - boundary_cut:
            continue

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        filling_v[:] = weight * mu / det_df * scale_vec * norm_b1

        particle_to_mat_kernels.vec_fill_v2(
            args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
        )

    vec1 /= Np
    vec2 /= Np
    vec3 /= Np

    # -- removed omp: #$ omp end parallel


@stack_array(
    "dfm",
    "df_inv_t",
    "df_inv",
    "g_inv",
    "filling_v",
    "tmp1",
    "tmp2",
    "tmp_v",
    "b",
    "b_prod",
    "norm_b2_prod",
    "b_star",
    "curl_norm_b",
    "norm_b1",
    "norm_b2",
    "grad_PB",
)
def cc_lin_mhd_5d_J2(
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
    epsilon: float,  # model specific argument
    b1: "float[:,:,:]",  # model specific argument
    b2: "float[:,:,:]",  # model specific argument
    b3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    norm_b21: "float[:,:,:]",  # model specific argument
    norm_b22: "float[:,:,:]",  # model specific argument
    norm_b23: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    grad_PB1: "float[:,:,:]",  # model specific argument
    grad_PB2: "float[:,:,:]",  # model specific argument
    grad_PB3: "float[:,:,:]",  # model specific argument
    basis_u: "int",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: float,
):  # model specific argument
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

        norm_b21, norm_b22, norm_b23 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 2-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of normalized magnetic field as a 2-form.

        grad_PB1, grad_PB2, grad_PB3 : array[float]
            FE coefficients c_ijk of gradient of parallel magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = zeros((3, 3), dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    grad_PB = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, boundary_cut, eta1, eta2, eta3, v, mu, weight, span1, span2, span3, b1, b2, b3, b, b_star, norm_b1, norm_b2, norm_b2_prod, curl_norm_b, grad_PB, abs_b_star_para, dfm, df_inv, df_inv_t, g_inv, det_df, tmp1, tmp2, tmp_v, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        if eta1 < boundary_cut or eta1 > 1.0 - boundary_cut:
            continue

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]
        mu = markers[ip, 9]

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

        # norm_b2; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, norm_b21, norm_b22, norm_b23, norm_b2)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # b_star; 2form transformed into H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        if basis_u == 0:
            linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
            linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)

            linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 1:
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
            linalg_kernels.matrix_matrix(tmp2, norm_b2_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)

            linalg_kernels.matrix_vector(tmp2, grad_PB, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v1(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:
            linalg_kernels.matrix_matrix(b_prod, g_inv, tmp1)
            linalg_kernels.matrix_matrix(tmp1, norm_b2_prod, tmp2)
            linalg_kernels.matrix_matrix(tmp2, g_inv, tmp1)

            linalg_kernels.matrix_vector(tmp1, grad_PB, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

    vec1 /= Np
    vec2 /= Np
    vec3 /= Np

    # -- removed omp: #$ omp end parallel
