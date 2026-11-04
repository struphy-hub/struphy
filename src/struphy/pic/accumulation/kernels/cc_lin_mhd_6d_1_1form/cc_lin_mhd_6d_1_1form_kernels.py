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


@stack_array("b", "b_vec", "b_prod", "dfm", "df_inv", "df_inv_t", "g_inv")
def cc_lin_mhd_6d_1_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    basis_u: "int",
    scale_mat: "float",
    boundary_cut: "float",
):
    r"""Accumulates the antisymmetric density matrix of
    :class:`~struphy.propagators.current_coupling_6d_density_1form.CurrentCoupling6DDensity1form`, the 1-form variant
    of :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_6d_1.cc_lin_mhd_6d_1_kernels.cc_lin_mhd_6d_1`, with the
    filling functions

    .. math::

        A_p^{\mu, \nu} = - s\, w_p\, [ F_p ]_{\mu, \nu}\,,\qquad
        F_p = \left\{
        \begin{aligned}
        &\sqrt g\, \hat{\mathbf B}^v_{\times} && \textnormal{for } \mathbf U \in (H^1)^3\,,
        \\
        &\frac{1}{\sqrt g}\, \hat{\mathbf B}^1_{\times} && \textnormal{for } \mathbf U \in H(\textnormal{curl})\,,
        \\
        &\frac{1}{\sqrt g}\, \hat{\mathbf B}^v_{\times} && \textnormal{for } \mathbf U \in H(\textnormal{div})\,,
        \end{aligned}
        \right.

    evaluated at :math:`\boldsymbol \eta_p`, where :math:`\mathbf a_\times \mathbf c := \mathbf a \times \mathbf c`,
    :math:`\hat{\mathbf B}^1` is the magnetic field 1-form and :math:`\hat{\mathbf B}^v = G^{-1} \hat{\mathbf B}^1`
    its vector-field components. These equal the fillings of the 2-form kernel with
    :math:`\hat{\mathbf B}^2 = \sqrt g\, \hat{\mathbf B}^v`, using
    :math:`G^{-1} (\sqrt g\, G^{-1}\hat{\mathbf B}^1)_\times G^{-1} = \hat{\mathbf B}^1_\times / \sqrt g`
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

    # allocate for magnetic field evaluation (1-form and vector-field components)
    b = empty(3, dtype=float)
    b_vec = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # get local number of markers
    n_markers_loc = args_markers.n_markers

    # fmt: off
    #$ omp parallel firstprivate(b_prod) private(b, b_vec, det_df, df_inv, df_inv_t, dfm, eta1, eta2, eta3, fac, filling_m12, filling_m13, filling_m23, g_inv, ip, span1, span2, span3, thread_derham, weight)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for reduction ( + : mat12, mat13, mat23)
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
        weight = markers[6, ip]

        if basis_u == 1:
            # F_p = [b1]_x / det_df
            b_vec[:] = b
            fac = -weight * scale_mat / det_df
        else:
            # vector-field components b^v = G^(-1) b1
            linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
            linalg_kernels.transpose(df_inv, df_inv_t)
            linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)
            linalg_kernels.matrix_vector(g_inv, b, b_vec)

            # F_p = det_df * [b^v]_x (H1vec) or [b^v]_x / det_df (Hdiv)
            if basis_u == 0:
                fac = -weight * scale_mat * det_df
            else:
                fac = -weight * scale_mat / det_df

        # operator bx() as matrix
        b_prod[0, 1] = -b_vec[2]
        b_prod[0, 2] = +b_vec[1]
        b_prod[1, 0] = +b_vec[2]
        b_prod[1, 2] = -b_vec[0]
        b_prod[2, 0] = -b_vec[1]
        b_prod[2, 1] = +b_vec[0]

        # filling functions
        filling_m12 = fac * b_prod[0, 1]
        filling_m13 = fac * b_prod[0, 2]
        filling_m23 = fac * b_prod[1, 2]

        # call the appropriate matvec filler
        if basis_u == 0:
            particle_to_mat_kernels.mat_fill_v0vec_asym(
                thread_derham,
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
            particle_to_mat_kernels.mat_fill_v1_asym(
                thread_derham,
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
            particle_to_mat_kernels.mat_fill_v2_asym(
                thread_derham,
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

    # fmt: off
    #$ omp end parallel
    # fmt: on
