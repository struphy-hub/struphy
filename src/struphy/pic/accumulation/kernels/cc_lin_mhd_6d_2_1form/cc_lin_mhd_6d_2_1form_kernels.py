"Accumulation kernel for full-orbit (6D) particles, magnetic field as a 1-form."

from numpy import empty, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "b",
    "b_cart",
    "b_prod",
    "dfm",
    "df_t",
    "df_inv",
    "df_inv_t",
    "filling_m",
    "filling_v",
    "tmp1",
    "tmp_t",
    "tmp_v",
    "tmp_m",
    "v",
)
def cc_lin_mhd_6d_2_1form(
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
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    scale_vec: "float",
    boundary_cut: "float",
):
    r"""Accumulates the matrix and vector of
    :class:`~struphy.propagators.current_coupling_6d_current_1form.CurrentCoupling6DCurrent1form`, the 1-form variant
    of :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_2.cc_lin_mhd_6d_2_kernels.cc_lin_mhd_6d_2`, with the
    filling functions

    .. math::

        A_p^{\mu, \nu} = w_p\, s_\textrm{mat}\, \frac{[ T_p T_p^\top ]_{\mu, \nu}}{J_p^{2k}}\,,\qquad
        B_p^\mu = w_p\, s_\textrm{vec}\, \frac{[ T_p \mathbf v_p ]_\mu}{J_p^{k}}\,,\qquad
        T_p = \left\{
        \begin{aligned}
        &DF^{-1}\, \mathbf B_\times && \textnormal{for } \mathbf U \in H(\textnormal{curl})\,,
        \\
        &DF^\top\, \mathbf B_\times && \textnormal{for } \mathbf U \in H(\textnormal{div}),\, (H^1)^3\,,
        \end{aligned}
        \right.

    evaluated at :math:`\boldsymbol \eta_p`, with :math:`k = 1` for :math:`H(\textnormal{div})` and :math:`k = 0`
    otherwise, where :math:`J = \det DF`, :math:`\mathbf a_\times \mathbf c := \mathbf a \times \mathbf c` and
    :math:`\mathbf B = DF^{-\top} \hat{\mathbf B}^1` are the Cartesian components of the magnetic field 1-form
    :math:`\hat{\mathbf B}^1`. These equal the fillings of the 2-form kernel with
    :math:`\hat{\mathbf B}^2 = J\, DF^{-1} \mathbf B`, using
    :math:`\hat{\mathbf B}^2_\times DF^{-1} = DF^\top \mathbf B_\times` and :math:`G^{-1} DF^\top = DF^{-1}`
    (from :math:`(M \mathbf a)_\times = \det(M) M^{-\top} \mathbf a_\times M^{-1}`).

    Parameters
    ----------
        b1_1, b1_2, b1_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for magnetic field evaluation (1-form and Cartesian components)
    b = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)
    v = empty(3, dtype=float)

    tmp1 = empty((3, 3), dtype=float)
    tmp_t = empty((3, 3), dtype=float)
    tmp_m = empty((3, 3), dtype=float)

    tmp_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = args_markers.n_markers

    # fmt: off
    #$ omp parallel firstprivate(b_prod) private(b, b_cart, det_df, df_inv, df_inv_t, df_t, dfm, eta1, eta2, eta3, filling_m, filling_v, ip, span1, span2, span3, thread_derham, tmp1, tmp_m, tmp_t, tmp_v, v, weight)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for reduction(+: mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    # fmt: on
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[0, ip] == -1.0:
            continue

        # boundary cut
        if markers[0, ip] < boundary_cut or markers[0, ip] > 1.0 - boundary_cut:
            continue

        # marker positions
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        # b-field evaluation (1-form components)
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            b1_1,
            b1_2,
            b1_3,
            b,
        )

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # Cartesian components of the magnetic field B = DF^(-T) b1
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_vector(df_inv_t, b, b_cart)

        # operator Bx() as matrix
        b_prod[0, 1] = -b_cart[2]
        b_prod[0, 2] = +b_cart[1]
        b_prod[1, 0] = +b_cart[2]
        b_prod[1, 2] = -b_cart[0]
        b_prod[2, 0] = -b_cart[1]
        b_prod[2, 1] = +b_cart[0]

        # marker weight and velocity
        weight = markers[6, ip]
        v[:] = markers[3:6, ip]

        # tmp1 = T_p = DF^(-1) * B^x (Hcurl) or DF^T * B^x (H1vec, Hdiv)
        if basis_u == 1:
            linalg_kernels.matrix_matrix(df_inv, b_prod, tmp1)
        else:
            linalg_kernels.transpose(dfm, df_t)
            linalg_kernels.matrix_matrix(df_t, b_prod, tmp1)

        # filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v
        linalg_kernels.transpose(tmp1, tmp_t)

        linalg_kernels.matrix_matrix(tmp1, tmp_t, tmp_m)
        linalg_kernels.matrix_vector(tmp1, v, tmp_v)

        if basis_u == 2:
            filling_m[:, :] = weight * tmp_m * scale_mat / det_df**2
            filling_v[:] = weight * tmp_v * scale_vec / det_df
        else:
            filling_m[:, :] = weight * tmp_m * scale_mat
            filling_v[:] = weight * tmp_v * scale_vec

        # call the appropriate matvec filler
        if basis_u == 0:
            particle_to_mat_kernels.m_v_fill_v0vec_symm(
                thread_derham,
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
            particle_to_mat_kernels.m_v_fill_v1_symm(
                thread_derham,
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
            particle_to_mat_kernels.m_v_fill_v2_symm(
                thread_derham,
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

    # fmt: off
    #$ omp end parallel
    # fmt: on
