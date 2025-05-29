"""Accumulation kernels for gyro-center (5D) particles.

Function naming conventions:

* use the model name, all lower-case letters (e.g. ``lin_vlasov_maxwell``)
* in case of multiple accumulations in one model, attach ``_1``, ``_2`` or the species name.

These kernels are passed to :class:`struphy.pic.accumulation.particles_to_grid.Accumulator`.
"""

from numpy import empty, shape, zeros, mod
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
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments


def gc_density_0form(
    markers: "float[:,:]",
    n_markers_tot: "int",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
):
    r"""
    Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V0 with the filling

    .. math::

        B_p^\mu = \frac{w_p}{N} \,.
    """

    #$ omp parallel private (ip, eta1, eta2, eta3, f0, filling)
    #$ omp for reduction ( + :vec)
    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # filling = w_p/N
        filling = markers[ip, 5] / n_markers_tot

        particle_to_mat_kernels.vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, filling)

    #$ omp end parallel


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "tmp1", "tmp2", "b", "beq", "b_prod", "bstar", "norm_b1", "curl_norm_b")
def cc_lin_mhd_5d_D(
    markers: "float[:,:]",
    n_markers_tot: "int",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    epsilon: float,  # model specific argument
    b2_1: "float[:,:,:]",  # model specific argument
    b2_2: "float[:,:,:]",  # model specific argument
    b2_3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_mat: "float",  # model specific argument
    boundary_cut: float,  # model specific argument
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

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
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
        weight = markers[ip, 7]

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

    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat23 /= n_markers_tot


@stack_array("dfm", "norm_b1", "filling_v")
def cc_lin_mhd_5d_M(
    markers: "float[:,:]",
    n_markers_tot: "int",
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
    boundary_cut: "float",  # model specific argument
):
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

    # allocate for a field evaluation
    norm_b1 = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

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

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


@stack_array(
    "dfm",
    "df_inv",
    "df_inv_t",
    "g_inv",
    "filling_m",
    "filling_v",
    "tmp",
    "tmp1",
    "tmp2",
    "tmp_m",
    "tmp_v",
    "b",
    "beq",
    "bfull_star",
    "b_prod",
    "b_prod_negb",
    "beq_prod",
    "beq_prod_negb",
    "norm_b1",
    "curl_norm_b",
)
def cc_lin_mhd_5d_curlb(
    markers: "float[:,:]",
    n_markers_tot: "int",
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
    beq1: "float[:,:,:]",  # model specific argument
    beq2: "float[:,:,:]",  # model specific argument
    beq3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_mat: "float",  # model specific argument
    scale_vec: "float",  # model specific argument
    boundary_cut: "float",  # model specific argument
    include_feq_b: "bool",  # model specific argument
):
    r"""TODO"""

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    beq_prod = zeros((3, 3), dtype=float)
    beq_prod_neg = zeros((3, 3), dtype=float)
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
    filling_m = zeros((3, 3), dtype=float)
    filling_v = zeros(3, dtype=float)

    tmp = empty((3, 3), dtype=float)
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 7]
        dweight = markers[ip, 5]
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

        # beq; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, beq1, beq2, beq3, beq)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # b_star; 2form in H1vec
        bfull_star[:] = (b + beq + curl_norm_b * v * epsilon)

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

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

        beq_prod[0, 1] = -beq[2]
        beq_prod[0, 2] = +beq[1]
        beq_prod[1, 0] = +beq[2]
        beq_prod[1, 2] = -beq[0]
        beq_prod[2, 0] = -beq[1]
        beq_prod[2, 1] = +beq[0]

        beq_prod_neg[:] = -1.0 * beq_prod

        if basis_u == 0:

            # beq contribution
            linalg_kernels.matrix_matrix(beq_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, beq_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(beq_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = dweight * tmp_m * v**2 / abs_b_star_para**2 * scale_mat
            filling_v[:] = dweight * tmp_v * v**2 / abs_b_star_para * scale_vec

            # b contribution
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)
            if include_feq_b:
                filling_m[:, :] += weight * tmp_m * v**2 / abs_b_star_para**2 * scale_mat
                filling_v[:] += weight * tmp_v * v**2 / abs_b_star_para * scale_vec
            else:
                filling_m[:, :] += dweight * tmp_m * v**2 / abs_b_star_para**2 * scale_mat
                filling_v[:] += dweight * tmp_v * v**2 / abs_b_star_para * scale_vec

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
        
        elif basis_u == 2:

            # beq contribution
            linalg_kernels.matrix_matrix(beq_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, beq_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(beq_prod, curl_norm_b, tmp_v)

            filling_m[:, :] = dweight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
            filling_v[:] = dweight * tmp_v * v**2 / abs_b_star_para / det_df * scale_vec

            # b contribution
            linalg_kernels.matrix_matrix(b_prod, tmp, tmp1)
            linalg_kernels.matrix_matrix(tmp1, b_prod_neg, tmp_m)
            linalg_kernels.matrix_vector(b_prod, curl_norm_b, tmp_v)

            if include_feq_b:
                filling_m[:, :] += weight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
                filling_v[:] += weight * tmp_v * v**2 / abs_b_star_para / det_df * scale_vec
            else:
                filling_m[:, :] += dweight * tmp_m * v**2 / abs_b_star_para**2 / det_df**2 * scale_mat
                filling_v[:] += dweight * tmp_v * v**2 / abs_b_star_para / det_df * scale_vec

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

    mat11 /= n_markers_tot
    mat12 /= n_markers_tot
    mat13 /= n_markers_tot
    mat22 /= n_markers_tot
    mat23 /= n_markers_tot
    mat33 /= n_markers_tot

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


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
    "beq",
    "beq_prod",
    "norm_b_prod",
    "bfull_star",
    "curl_norm_b",
    "norm_b1",
    "grad_PB",
    "grad_PBeq",
)
def cc_lin_mhd_5d_gradB(
    markers: "float[:,:]",
    n_markers_tot: "int",
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
    beq1: "float[:,:,:]",  # model specific argument
    beq2: "float[:,:,:]",  # model specific argument
    beq3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    grad_PB1: "float[:,:,:]",  # model specific argument
    grad_PB2: "float[:,:,:]",  # model specific argument
    grad_PB3: "float[:,:,:]",  # model specific argument
    grad_PBeq1: "float[:,:,:]",  # model specific argument
    grad_PBeq2: "float[:,:,:]",  # model specific argument
    grad_PBeq3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_mat: "float",  # model specific argument
    scale_vec: "float",  # model specific argument
    boundary_cut: "float",  # model specific argument
    include_feq_b: "bool",  # model specific argument
):
    r"""TODO"""

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    beq_prod = zeros((3, 3), dtype=float)
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

    # get number of markers
    n_markers_loc = shape(markers)[0]

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
        dweight = markers[ip, 5]
        weight = markers[ip, 7]

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

        # beq; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, beq1, beq2, beq3, beq)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # grad_PBeq; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PBeq1, grad_PBeq2, grad_PBeq3, grad_PBeq)

        # b_star; 2form transformed into H1vec
        bfull_star[:] = (b + beq + curl_norm_b * v * epsilon)

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        beq_prod[0, 1] = -beq[2]
        beq_prod[0, 2] = +beq[1]
        beq_prod[1, 0] = +beq[2]
        beq_prod[1, 2] = -beq[0]
        beq_prod[2, 0] = -beq[1]
        beq_prod[2, 1] = +beq[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        if basis_u == 0:

            # beq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = dweight * tmp_v * mu / abs_b_star_para * scale_vec

            # beq x grad b para
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # b x grad beq para
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            if include_feq_b:
                filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec
            else:
                filling_v[:] += dweight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b x grad b para
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:

            # beq x grad beq para
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = dweight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # beq x grad b para
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b x grad beq para
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            if include_feq_b:
                filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec
            else:
                filling_v[:] += dweight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b x grad b para
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


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
    "beq",
    "beq_prod",
    "norm_b_prod",
    "bfull_star",
    "curl_norm_b",
    "norm_b1",
    "grad_PB",
    "grad_PBeq",
)
def cc_lin_mhd_5d_gradB_dg_init(
    markers: "float[:,:]",
    n_markers_tot: "int",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,  # model specific argument
    b1: "float[:,:,:]",  # model specific argument
    b2: "float[:,:,:]",  # model specific argument
    b3: "float[:,:,:]",  # model specific argument
    beq1: "float[:,:,:]",  # model specific argument
    beq2: "float[:,:,:]",  # model specific argument
    beq3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    grad_PB1: "float[:,:,:]",  # model specific argument
    grad_PB2: "float[:,:,:]",  # model specific argument
    grad_PB3: "float[:,:,:]",  # model specific argument
    grad_PBeq1: "float[:,:,:]",  # model specific argument
    grad_PBeq2: "float[:,:,:]",  # model specific argument
    grad_PBeq3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_vec: "float",  # model specific argument
):
    r"""TODO"""

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    beq_prod = zeros((3, 3), dtype=float)
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

    # get number of markers
    n_markers_loc = shape(markers)[0]

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

        # beq; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, beq1, beq2, beq3, beq)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # grad_PBeq; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PBeq1, grad_PBeq2, grad_PBeq3, grad_PBeq)

        # b_star; 2form transformed into H1vec
        bfull_star[:] = (b + beq + curl_norm_b * v * epsilon)

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        beq_prod[0, 1] = -beq[2]
        beq_prod[0, 2] = +beq[1]
        beq_prod[1, 0] = +beq[2]
        beq_prod[1, 2] = -beq[0]
        beq_prod[2, 0] = -beq[1]
        beq_prod[2, 1] = +beq[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        if basis_u == 0:

            # beq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # b contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:

            # beq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


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
    "eta_diff",
    "beq",
    "beq_prod",
    "norm_b_prod",
    "bfull_star",
    "curl_norm_b",
    "norm_b1",
    "grad_PB",
    "grad_PBeq",
    "eta_mid",
    "eta_diff",
)
def cc_lin_mhd_5d_gradB_dg(
    markers: "float[:,:]",
    n_markers_tot: "int",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,  # model specific argument
    b1: "float[:,:,:]",  # model specific argument
    b2: "float[:,:,:]",  # model specific argument
    b3: "float[:,:,:]",  # model specific argument
    beq1: "float[:,:,:]",  # model specific argument
    beq2: "float[:,:,:]",  # model specific argument
    beq3: "float[:,:,:]",  # model specific argument
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    curl_norm_b1: "float[:,:,:]",  # model specific argument
    curl_norm_b2: "float[:,:,:]",  # model specific argument
    curl_norm_b3: "float[:,:,:]",  # model specific argument
    grad_PB1: "float[:,:,:]",  # model specific argument
    grad_PB2: "float[:,:,:]",  # model specific argument
    grad_PB3: "float[:,:,:]",  # model specific argument
    grad_PBeq1: "float[:,:,:]",  # model specific argument
    grad_PBeq2: "float[:,:,:]",  # model specific argument
    grad_PBeq3: "float[:,:,:]",  # model specific argument
    basis_u: "int",  # model specific argument
    scale_vec: "float",  # model specific argument
    const: "float",  # model specific argument
):
    r"""TODO"""

    # allocate for magnetic field evaluation
    eta_diff = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    b = empty(3, dtype=float)
    beq = empty(3, dtype=float)
    bfull_star = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    beq_prod = zeros((3, 3), dtype=float)
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

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, mid point
        eta_mid[:] = (markers[ip, 0:3] + markers[ip, 11:14])/2.
        eta_mid[:] = mod(eta_mid[:], 1.)

        eta_diff[:] = markers[ip, 0:3] - markers[ip, 11:14]

        # marker weight and velocity
        weight = markers[ip, 5]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta_mid[0], eta_mid[1], eta_mid[2], args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # needed metric coefficients
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, b1, b2, b3, b)

        # beq; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, beq1, beq2, beq3, beq)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, args_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # grad_PB; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PB1, grad_PB2, grad_PB3, grad_PB)

        # grad_PBeq; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, grad_PBeq1, grad_PBeq2, grad_PBeq3, grad_PBeq)

        # b_star; 2form transformed into H1vec
        bfull_star[:] = (b + beq + curl_norm_b * v * epsilon)

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, bfull_star)

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        beq_prod[0, 1] = -beq[2]
        beq_prod[0, 2] = +beq[1]
        beq_prod[1, 0] = +beq[2]
        beq_prod[1, 2] = -beq[0]
        beq_prod[2, 0] = -beq[1]
        beq_prod[2, 1] = +beq[0]

        norm_b_prod[0, 1] = -norm_b1[2]
        norm_b_prod[0, 2] = +norm_b1[1]
        norm_b_prod[1, 0] = +norm_b1[2]
        norm_b_prod[1, 2] = -norm_b1[0]
        norm_b_prod[2, 0] = -norm_b1[1]
        norm_b_prod[2, 1] = +norm_b1[0]

        if basis_u == 0:

            # beq * gradPBeq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para * scale_vec

            # beq * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # beq * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)
            filling_v[:] += tmp_v / abs_b_star_para * const

            # b * gradPBeq contribution
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # b * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para * scale_vec

            # b * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)
            filling_v[:] += tmp_v / abs_b_star_para * const

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v0vec(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

        elif basis_u == 2:

            # beq * gradPBeq contribution
            linalg_kernels.matrix_matrix(beq_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] = weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # beq * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # beq * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)

            filling_v[:] += tmp_v / abs_b_star_para / det_df * const

            # b * gradPBeq contribtuion
            linalg_kernels.matrix_matrix(b_prod, norm_b_prod, tmp)
            linalg_kernels.matrix_vector(tmp, grad_PBeq, tmp_v)

            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b * gradPB contribution
            linalg_kernels.matrix_vector(tmp, grad_PB, tmp_v)
            
            filling_v[:] += weight * tmp_v * mu / abs_b_star_para / det_df * scale_vec

            # b * dg term contribution
            linalg_kernels.matrix_vector(tmp, eta_diff, tmp_v)

            filling_v[:] += tmp_v / abs_b_star_para / det_df * const

            # call the appropriate matvec filler
            particle_to_mat_kernels.vec_fill_v2(
                args_derham, span1, span2, span3, vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]
            )

    vec1 /= n_markers_tot
    vec2 /= n_markers_tot
    vec3 /= n_markers_tot


def cc_lin_mhd_5d_M_scalar(
    markers: "float[:,:]",
    n_markers_tot: "int",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
    scale_vec: "float",  # model specific argument
    include_feq_b: "bool",  # model specific argument
):
    r"""TODO"""

    # get number of markers
    n_markers_loc = shape(markers)[0]

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        dweight = markers[ip, 5]
        weight = markers[ip, 7]
        mu = markers[ip, 9]

        if include_feq_b:
            filling = weight * mu * scale_vec
        else:
            filling = dweight * mu * scale_vec

        particle_to_mat_kernels.vec_fill_b_v0(
            args_derham, eta1, eta2, eta3, vec, filling)

    vec /= n_markers_tot