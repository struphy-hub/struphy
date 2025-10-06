"""Accumulation kernels for full-orbit (6D) particles.

Function naming conventions:

* use the model name, all lower-case letters (e.g. ``lin_vlasov_maxwell``)
* in case of multiple accumulations in one model, attach ``_1``, ``_2`` or the species name.

These kernels are passed to :class:`struphy.pic.accumulation.particles_to_grid.Accumulator`.
"""

from numpy import empty, floor, log, shape, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


def charge_density_0form(
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
    weight_idx = args_markers.weight_idx

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
        filling = markers[ip, weight_idx] / Np

        particle_to_mat_kernels.vec_fill_b_v0(
            args_derham,
            eta1,
            eta2,
            eta3,
            vec,
            filling,
        )

    # -- removed omp: #$ omp end parallel


@stack_array(
    "cell_left",
    "point_left",
    "point_right",
    "cell_number",
    "temp1",
    "temp4",
    "compact",
    "grids_shapex",
    "grids_shapey",
    "grids_shapez",
)
def hybrid_fA_density(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat: "float[:,:,:,:,:,:]",
    Nel: "int[:]",
    quad: "int[:]",
    quad_pts_x: "float[:]",
    quad_pts_y: "float[:]",
    quad_pts_z: "float[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
):
    r"""
    Accumulates the values of density at quadrature points with the filling functions

    .. math::
        n = \sum_p w_p S(x - x_p)

    Parameters
    ----------
        To do
    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate
    cell_left = empty(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = empty(3, dtype=int)

    temp1 = zeros(3, dtype=float)
    temp4 = zeros(3, dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0] + 1.0) * p_size[0]
    compact[1] = (p_shape[1] + 1.0) * p_size[1]
    compact[2] = (p_shape[2] + 1.0) * p_size[2]

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)

    dfm = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # -- removed omp: #$ omp parallel private (dfm, det_df, cell_left, point_left, point_right, cell_number, temp1, temp4, compact, grids_shapex, grids_shapey, grids_shapez, n_markers, ip, eta1, eta2, eta3, weight, ie1, ie2, ie3, span1, span2, span3)
    # -- removed omp: #$ omp for reduction ( + : mat)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        weight = markers[ip, 6] / (p_size[0] * p_size[1] * p_size[2]) / Np / det_df

        ie1 = int(eta1 * Nel[0])
        ie2 = int(eta2 * Nel[1])
        ie3 = int(eta3 * Nel[2])

        # the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0] = eta1 - 0.5 * compact[0]
        point_right[0] = eta1 + 0.5 * compact[0]
        point_left[1] = eta2 - 0.5 * compact[1]
        point_right[1] = eta2 + 0.5 * compact[1]
        point_left[2] = eta3 - 0.5 * compact[2]
        point_right[2] = eta3 + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        span1 = int(eta1 * Nel[0]) + int(args_derham.pn[0])
        span2 = int(eta2 * Nel[1]) + int(args_derham.pn[1])
        span3 = int(eta3 * Nel[2]) + int(args_derham.pn[2])

        # =========== kernel part (periodic bundary case) ==========
        particle_to_mat_kernels.hybrid_density(
            Nel,
            args_derham,
            cell_left,
            cell_number,
            span1,
            span2,
            span3,
            ie1,
            ie2,
            ie3,
            temp1,
            temp4,
            quad,
            quad_pts_x,
            quad_pts_y,
            quad_pts_z,
            compact,
            eta1,
            eta2,
            eta3,
            mat,
            weight,
            p_shape,
            p_size,
            grids_shapex,
            grids_shapey,
            grids_shapez,
        )
    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "df_t", "df_inv", "df_inv_times_v", "filling_m", "filling_v", "v")
def hybrid_fA_Arelated(
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
):
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= f_0(\eta_p, v_p) * [ DF^{-1}(\eta_p) * v_p ]_\mu * [ DF^{-1}(\eta_p) * v_p ]_\nu

        B_p^\mu &= \sqrt{f_0(\eta_p, v_p)} * w_p * [ DF^{-1}(\eta_p) * v_p ]_\mu

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    df_inv_times_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, v, dfm, df_inv, df_inv_times_v, weight, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, vec1, vec2, vec3)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate background
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)

        weight = markers[ip, 6]

        # filling_m
        filling_m[0, 0] = (
            weight / Np * (df_inv[0, 0] * df_inv[0, 0] + df_inv[0, 1] * df_inv[0, 1] + df_inv[0, 2] * df_inv[0, 2])
        )
        filling_m[0, 1] = (
            weight / Np * (df_inv[0, 0] * df_inv[1, 0] + df_inv[0, 1] * df_inv[1, 1] + df_inv[0, 2] * df_inv[1, 2])
        )
        filling_m[0, 2] = (
            weight / Np * (df_inv[0, 0] * df_inv[2, 0] + df_inv[0, 1] * df_inv[2, 1] + df_inv[0, 2] * df_inv[2, 2])
        )

        filling_m[1, 1] = (
            weight / Np * (df_inv[1, 0] * df_inv[1, 0] + df_inv[1, 1] * df_inv[1, 1] + df_inv[1, 2] * df_inv[1, 2])
        )
        filling_m[1, 2] = (
            weight / Np * (df_inv[1, 0] * df_inv[2, 0] + df_inv[1, 1] * df_inv[2, 1] + df_inv[1, 2] * df_inv[2, 2])
        )

        filling_m[2, 2] = (
            weight / Np * (df_inv[2, 0] * df_inv[2, 0] + df_inv[2, 1] * df_inv[2, 1] + df_inv[2, 2] * df_inv[2, 2])
        )

        # filling_v
        filling_v[:] = weight / Np * df_inv_times_v

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(
            args_derham,
            eta1,
            eta2,
            eta3,
            mat11,
            mat12,
            mat13,
            mat22,
            mat23,
            mat33,
            filling_m[0, 0],
            filling_m[
                0,
                1,
            ],
            filling_m[0, 2],
            filling_m[1, 1],
            filling_m[
                1,
                2,
            ],
            filling_m[2, 2],
            vec1,
            vec2,
            vec3,
            filling_v[0],
            filling_v[1],
            filling_v[2],
        )

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "df_inv", "v", "df_inv_times_v", "filling_m", "filling_v")
def linear_vlasov_ampere(
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
    f0_values: "float[:]",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= \frac{\alpha^2 \kappa^2}{v_{\text{th}}^2} \frac{1}{N\, s_0} f_0(\mathbf{\eta}_p, \mathbf{v}_p)
            [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\nu \,,

        B_p^\mu &= \alpha^2 \kappa \sqrt{f_0(\mathbf{\eta}_p, \mathbf{v}_p)} w_p [ DF^{-1}(\mathbf{\eta}_p) \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------
    f0_values ; array[float]
        Value of f0 for each particle.

    Note
    ----
    The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)

    # allocate for filling
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0 or markers[ip, -1] == -2.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # get velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # compute DF^{-1} v
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # filling_m = alpha^2 * kappa^2 * f0 / (N * s_0 * v_th^2) * (DF^{-1} v_p)_mu * (DF^{-1} v_p)_nu
        linalg_kernels.outer(df_inv_v, df_inv_v, filling_m)
        filling_m[:, :] *= f0_values[ip] / (Np * markers[ip, 7])

        # filling_v = alpha^2 * kappa / N * w_p * DL^{-1} * v_p
        filling_v[:] = markers[ip, 6] * df_inv_v / Np

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(
            args_derham,
            eta1,
            eta2,
            eta3,
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

    # -- removed omp: #$ omp end parallel


def vlasov_maxwell_poisson(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
):
    r"""
    Accumulates the charge density in V0

    .. math::

        \rho_p^\mu = w_p \,.

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
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

        # filling = w_p
        filling = markers[ip, 6] / Np

        particle_to_mat_kernels.vec_fill_b_v0(
            args_derham,
            eta1,
            eta2,
            eta3,
            vec,
            filling,
        )

    # -- removed omp: #$ omp end parallel


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "v", "df_inv_times_v", "filling_m", "filling_v")
def vlasov_maxwell(
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
):
    r"""
    Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p \, G^{-1}_{\mu, \nu}(\boldsymbol \eta_p) \,,
        \\[2mm]
        B_p^\mu &= w_p [DF^{-1}(\boldsymbol \eta_p) \cdot \mathbf{v}_p ]_\mu \,.

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for metric coeffs
    dfm = zeros((3, 3), dtype=float)
    df_inv = zeros((3, 3), dtype=float)
    df_inv_t = zeros((3, 3), dtype=float)
    g_inv = zeros((3, 3), dtype=float)

    # allocate for filling
    v = zeros(3, dtype=float)
    df_inv_times_v = zeros(3, dtype=float)
    filling_m = zeros((3, 3), dtype=float)
    filling_v = zeros(3, dtype=float)

    # -- removed omp: #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_times_v, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # compute shifted and stretched velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # filling functions
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
        linalg_kernels.matrix_vector(df_inv, v, df_inv_times_v)

        # filling_m = w_p * DF^{-1} * DF^{-T}
        filling_m[:, :] = markers[ip, 6] * g_inv / Np

        # filling_v = w_p * DF^{-1} * \V
        filling_v[:] = markers[ip, 6] * df_inv_times_v / Np

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_b_v1_symm(
            args_derham,
            eta1,
            eta2,
            eta3,
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

    # -- removed omp: #$ omp end parallel


@stack_array("b", "b_prod", "dfm", "df_inv", "df_inv_tg_inv", "tmp1", "tmp2")
def cc_lin_mhd_6d_1(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    boundary_cut: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} = w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) ]_{\mu, \nu}

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

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

    # allocate some temporary buffers for filling
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, b2_1, b2_2, b2_3, b, dfm, det_df, weight, df_inv, df_inv_t, g_inv, tmp1, tmp2, filling_m12, filling_m13, filling_m23)
    # -- removed omp: #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian matrix and Jacobian determinant
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # marker weight
        weight = markers[ip, 6]

        if basis_u == 0:
            # filling functions
            filling_m12 = -weight * b_prod[0, 1] * scale_mat
            filling_m13 = -weight * b_prod[0, 2] * scale_mat
            filling_m23 = -weight * b_prod[1, 2] * scale_mat

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

            filling_m12 = -weight * tmp2[0, 1] * scale_mat
            filling_m13 = -weight * tmp2[0, 2] * scale_mat
            filling_m23 = -weight * tmp2[1, 2] * scale_mat

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
            filling_m12 = -weight * b_prod[0, 1] * scale_mat / det_df**2
            filling_m13 = -weight * b_prod[0, 2] * scale_mat / det_df**2
            filling_m23 = -weight * b_prod[1, 2] * scale_mat / det_df**2

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

    # -- removed omp: #$ omp end parallel

    mat12 /= Np
    mat13 /= Np
    mat23 /= Np


@stack_array(
    "b",
    "b_prod",
    "dfm",
    "df_inv",
    "df_inv_t",
    "g_inv",
    "filling_m",
    "filling_v",
    "tmp1",
    "tmp2",
    "tmp_t",
    "tmp_v",
    "tmp_m",
    "v",
)
def cc_lin_mhd_6d_2(
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
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        A_p^{\mu, \nu} &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * G^{-1}(\eta_p) * B2_{\times}(\eta_p)^\top * G^{-1}(\eta_p) ]_{\mu, \nu}

        B_p^\mu &= w_p * [ G^{-1}(\eta_p) * B2_{\times}(\eta_p) * DF^{-1}(\eta_p) * v_p ]_\mu

    where :math:`B2_{\times} * a := B2 \times a` for :math:`a \in \mathbb R^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for magnetic field evaluation
    b = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)

    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel firstprivate(b_prod) private(ip, eta1, eta2, eta3, span1, span2, span3, b2_1, b2_2, b2_3, b, dfm, det_df, weight, v, df_inv, df_inv_t, g_inv, tmp1, tmp2, tmp_t, tmp_m, tmp_v, filling_m, filling_v)
    # -- removed omp: #$ omp for reduction ( + : mat12, mat13, mat23)
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        if basis_u == 0:
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

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

            # filling functions tmp_m = tmp2 * tmp2^T and tmp_v = tmp2 * v, where tmp2 = G^(-1) * B^x * DF^(-1)
            linalg_kernels.matrix_matrix(g_inv, b_prod, tmp1)
            linalg_kernels.matrix_matrix(tmp1, df_inv, tmp2)

            linalg_kernels.transpose(tmp2, tmp_t)

            linalg_kernels.matrix_matrix(tmp2, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp2, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

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
            # needed metric coefficients
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

            # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1) / det(DF)
            linalg_kernels.matrix_matrix(b_prod, df_inv, tmp1)

            linalg_kernels.transpose(tmp1, tmp_t)

            linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
            linalg_kernels.matrix_vector(tmp1, v, tmp_v)

            filling_m[:, :] = weight * tmp_m * scale_mat / det_df**2
            filling_v[:] = weight * tmp_v * scale_vec / det_df

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

    # -- removed omp: #$ omp end parallel

    mat11 /= Np
    mat12 /= Np
    mat13 /= Np
    mat22 /= Np
    mat23 /= Np
    mat33 /= Np

    vec1 /= Np
    vec2 /= Np
    vec3 /= Np


@stack_array("dfm", "df_t", "df_inv", "df_inv_t", "filling_m", "filling_v", "tmp1", "v", "tmp_v")
def pc_lin_mhd_6d_full(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11_11: "float[:,:,:,:,:,:]",
    mat12_11: "float[:,:,:,:,:,:]",
    mat13_11: "float[:,:,:,:,:,:]",
    mat22_11: "float[:,:,:,:,:,:]",
    mat23_11: "float[:,:,:,:,:,:]",
    mat33_11: "float[:,:,:,:,:,:]",
    mat11_12: "float[:,:,:,:,:,:]",
    mat12_12: "float[:,:,:,:,:,:]",
    mat13_12: "float[:,:,:,:,:,:]",
    mat22_12: "float[:,:,:,:,:,:]",
    mat23_12: "float[:,:,:,:,:,:]",
    mat33_12: "float[:,:,:,:,:,:]",
    mat11_13: "float[:,:,:,:,:,:]",
    mat12_13: "float[:,:,:,:,:,:]",
    mat13_13: "float[:,:,:,:,:,:]",
    mat22_13: "float[:,:,:,:,:,:]",
    mat23_13: "float[:,:,:,:,:,:]",
    mat33_13: "float[:,:,:,:,:,:]",
    mat11_22: "float[:,:,:,:,:,:]",
    mat12_22: "float[:,:,:,:,:,:]",
    mat13_22: "float[:,:,:,:,:,:]",
    mat22_22: "float[:,:,:,:,:,:]",
    mat23_22: "float[:,:,:,:,:,:]",
    mat33_22: "float[:,:,:,:,:,:]",
    mat11_23: "float[:,:,:,:,:,:]",
    mat12_23: "float[:,:,:,:,:,:]",
    mat13_23: "float[:,:,:,:,:,:]",
    mat22_23: "float[:,:,:,:,:,:]",
    mat23_23: "float[:,:,:,:,:,:]",
    mat33_23: "float[:,:,:,:,:,:]",
    mat11_33: "float[:,:,:,:,:,:]",
    mat12_33: "float[:,:,:,:,:,:]",
    mat13_33: "float[:,:,:,:,:,:]",
    mat22_33: "float[:,:,:,:,:,:]",
    mat23_33: "float[:,:,:,:,:,:]",
    mat33_33: "float[:,:,:,:,:,:]",
    vec1_1: "float[:,:,:]",
    vec2_1: "float[:,:,:]",
    vec3_1: "float[:,:,:]",
    vec1_2: "float[:,:,:]",
    vec2_2: "float[:,:,:]",
    vec3_2: "float[:,:,:]",
    vec1_3: "float[:,:,:]",
    vec2_3: "float[:,:,:]",
    vec3_3: "float[:,:,:]",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        V_{p,i} A_p^{\mu, \nu} V_{p,j} &= w_p * [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} * V_{p,i} * V_{p,j}

        V_{p,i} B_p^\mu &= w_p * [DF^{-1}(\eta_p)]_\mu * V_{p,i}

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # Avoid second computation of dfm, use linear_algebra.linalg_kernels routines to get g_inv:
        linalg_kernels.matrix_inv(dfm, df_inv)
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.transpose(df_inv, df_inv_t)

        # filling functions
        v[:] = markers[ip, 3:6]

        linalg_kernels.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        weight = markers[ip, 8]

        filling_m[:, :] = weight * tmp1 / Np * scale_mat
        filling_v[:] = weight * tmp_v / Np * scale_vec

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_pressure_full(
            args_derham,
            span1,
            span2,
            span3,
            mat11_11,
            mat12_11,
            mat13_11,
            mat22_11,
            mat23_11,
            mat33_11,
            mat11_12,
            mat12_12,
            mat13_12,
            mat22_12,
            mat23_12,
            mat33_12,
            mat11_13,
            mat12_13,
            mat13_13,
            mat22_13,
            mat23_13,
            mat33_13,
            mat11_22,
            mat12_22,
            mat13_22,
            mat22_22,
            mat23_22,
            mat33_22,
            mat11_23,
            mat12_23,
            mat13_23,
            mat22_23,
            mat23_23,
            mat33_23,
            mat11_33,
            mat12_33,
            mat13_33,
            mat22_33,
            mat23_33,
            mat33_33,
            filling_m[0, 0],
            filling_m[0, 1],
            filling_m[0, 2],
            filling_m[1, 1],
            filling_m[1, 2],
            filling_m[2, 2],
            vec1_1,
            vec2_1,
            vec3_1,
            vec1_2,
            vec2_2,
            vec3_2,
            vec1_3,
            vec2_3,
            vec3_3,
            filling_v[0],
            filling_v[1],
            filling_v[2],
            v[0],
            v[1],
            v[2],
        )


@stack_array("dfm", "df_inv_t", "df_inv", "filling_m", "filling_v", "tmp1", "v", "tmp_v")
def pc_lin_mhd_6d(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11_11: "float[:,:,:,:,:,:]",
    mat12_11: "float[:,:,:,:,:,:]",
    mat13_11: "float[:,:,:,:,:,:]",
    mat22_11: "float[:,:,:,:,:,:]",
    mat23_11: "float[:,:,:,:,:,:]",
    mat33_11: "float[:,:,:,:,:,:]",
    mat11_12: "float[:,:,:,:,:,:]",
    mat12_12: "float[:,:,:,:,:,:]",
    mat13_12: "float[:,:,:,:,:,:]",
    mat22_12: "float[:,:,:,:,:,:]",
    mat23_12: "float[:,:,:,:,:,:]",
    mat33_12: "float[:,:,:,:,:,:]",
    mat11_13: "float[:,:,:,:,:,:]",
    mat12_13: "float[:,:,:,:,:,:]",
    mat13_13: "float[:,:,:,:,:,:]",
    mat22_13: "float[:,:,:,:,:,:]",
    mat23_13: "float[:,:,:,:,:,:]",
    mat33_13: "float[:,:,:,:,:,:]",
    mat11_22: "float[:,:,:,:,:,:]",
    mat12_22: "float[:,:,:,:,:,:]",
    mat13_22: "float[:,:,:,:,:,:]",
    mat22_22: "float[:,:,:,:,:,:]",
    mat23_22: "float[:,:,:,:,:,:]",
    mat33_22: "float[:,:,:,:,:,:]",
    mat11_23: "float[:,:,:,:,:,:]",
    mat12_23: "float[:,:,:,:,:,:]",
    mat13_23: "float[:,:,:,:,:,:]",
    mat22_23: "float[:,:,:,:,:,:]",
    mat23_23: "float[:,:,:,:,:,:]",
    mat33_23: "float[:,:,:,:,:,:]",
    mat11_33: "float[:,:,:,:,:,:]",
    mat12_33: "float[:,:,:,:,:,:]",
    mat13_33: "float[:,:,:,:,:,:]",
    mat22_33: "float[:,:,:,:,:,:]",
    mat23_33: "float[:,:,:,:,:,:]",
    mat33_33: "float[:,:,:,:,:,:]",
    vec1_1: "float[:,:,:]",
    vec2_1: "float[:,:,:]",
    vec3_1: "float[:,:,:]",
    vec1_2: "float[:,:,:]",
    vec2_2: "float[:,:,:]",
    vec3_2: "float[:,:,:]",
    vec1_3: "float[:,:,:]",
    vec2_3: "float[:,:,:]",
    vec3_3: "float[:,:,:]",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
):
    r"""Accumulates into V1 with the filling functions

    .. math::

        {V_{p,i}}_\perp A_p^{\mu, \nu} {V_{p,j}}_\perp &= w_p * [ DF^{-1}(\eta_p) DF^{-\top}(\eta_p) ]_{\mu, \nu} * {V_{p,i}}_\perp * {V_{p,j}}_\perp

        {V_{p,i}}_\perp B_p^\mu &= w_p * [DF^{-1}(\eta_p)]_\mu * {V_{p,i}}_\perp

    Parameters
    ----------

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    Np = args_markers.Np

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)

    v = empty(3, dtype=float)
    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # boundary cut
        if markers[ip, 0] < boundary_cut or markers[ip, 0] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 6]
        v[:] = markers[ip, 3:6]

        # evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # Avoid second computation of dfm, use linear_algebra.linalg_kernels routines to get g_inv:
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)

        linalg_kernels.matrix_matrix(df_inv, df_inv_t, tmp1)
        linalg_kernels.matrix_vector(df_inv, v, tmp_v)

        filling_m[:, :] = weight * tmp1 * scale_mat
        filling_v[:] = weight * tmp_v * scale_vec

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_pressure(
            args_derham,
            span1,
            span2,
            span3,
            mat11_11,
            mat12_11,
            mat13_11,
            mat22_11,
            mat23_11,
            mat33_11,
            mat11_12,
            mat12_12,
            mat13_12,
            mat22_12,
            mat23_12,
            mat33_12,
            mat11_22,
            mat12_22,
            mat13_22,
            mat22_22,
            mat23_22,
            mat33_22,
            filling_m[0, 0],
            filling_m[0, 1],
            filling_m[0, 2],
            filling_m[1, 1],
            filling_m[1, 2],
            filling_m[2, 2],
            vec1_1,
            vec2_1,
            vec3_1,
            vec1_2,
            vec2_2,
            vec3_2,
            filling_v[0],
            filling_v[1],
            filling_v[2],
            v[0],
            v[1],
        )

    mat11_11 /= Np
    mat12_11 /= Np
    mat13_11 /= Np
    mat22_11 /= Np
    mat23_11 /= Np
    mat33_11 /= Np
    mat11_12 /= Np
    mat12_12 /= Np
    mat13_12 /= Np
    mat22_12 /= Np
    mat23_12 /= Np
    mat33_12 /= Np
    mat11_22 /= Np
    mat12_22 /= Np
    mat13_22 /= Np
    mat22_22 /= Np
    mat23_22 /= Np
    mat33_22 /= Np

    vec1_1 /= Np
    vec2_1 /= Np
    vec3_1 /= Np
    vec1_2 /= Np
    vec2_2 /= Np
    vec3_2 /= Np
