"Accumulation kernel for gyro-center (5D) particles, magnetic field as a 1-form."

from numpy import empty
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


@stack_array("dfm", "df_inv", "df_inv_t", "g_inv", "b1", "bv", "b_star", "norm_b1", "curl_norm_b")
def cc_lin_mhd_5d_D_1form(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    epsilon: float,
    ep_scale: "float",
    b1_1: "float[:,:,:]",
    b1_2: "float[:,:,:]",
    b1_3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    basis_u: "int",
):
    r"""Accumulation kernel for the propagator
    :class:`~struphy.propagators.current_coupling_5d_density_1form.CurrentCoupling5DDensity1form`, the 1-form version of
    :func:`~struphy.pic.accumulation.kernels.cc_lin_mhd_5d_D.cc_lin_mhd_5d_D_kernels.cc_lin_mhd_5d_D`.

    The magnetic field is a 1-form :math:`\hat{\mathbf B}^1`, with the vector-field (contravariant) components
    :math:`\hat{\mathbf B}^\textnormal{v} = G^{-1} \hat{\mathbf B}^1` (the 2-form is
    :math:`\hat{\mathbf B}^2 = \sqrt g\, \hat{\mathbf B}^\textnormal{v}`). With
    :math:`\hat{\mathbf B}^{*\textnormal{v}} = \hat{\mathbf B}^\textnormal{v} + \epsilon v_\parallel (\hat \nabla \times
    \hat{\mathbf b}^1_0)/\sqrt g` the antisymmetric filling functions are

    .. math::

        A_p^{\mu, \nu} = - w_p \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\epsilon}
        \left( 1-\frac{\hat{\mathbf b}^1_0 \cdot \hat{\mathbf B}^\textnormal{v}}{\hat{\mathbf b}^1_0 \cdot
        \hat{\mathbf B}^{*\textnormal{v}}} \right) \times
        \begin{cases}
            \sqrt g\, (\hat{\mathbf B}^\textnormal{v}_\times)_{\mu, \nu} & (H^1)^3\,,
            \\
            (\hat{\mathbf B}^1_\times)_{\mu, \nu} / \sqrt g & H(\textnormal{curl})\,,
            \\
            (\hat{\mathbf B}^\textnormal{v}_\times)_{\mu, \nu} / \sqrt g & H(\textnormal{div})\,,
        \end{cases}

    where :math:`\mathbf a_\times \mathbf c := \mathbf a \times \mathbf c`. The :math:`H(\textnormal{curl})` case uses
    :math:`G^{-1} (\hat{\mathbf B}^2_\times) G^{-1} = \sqrt g\, G^{-1} (G^{-1} \hat{\mathbf B}^1)_\times G^{-1} =
    (\hat{\mathbf B}^1_\times)/\sqrt g` (from :math:`M^\top (M\mathbf a)_\times M = \det(M)\, \mathbf a_\times`).

    Parameters
    ----------
        epsilon : float
            scaling factor.

        b1_1, b1_2, b1_3 : array[float]
            FE coefficients c_ijk of the magnetic field as a 1-form.

        norm_b11, norm_b12, norm_b12 : array[float]
            FE coefficients c_ijk of the unit magnetic field as a 1-form.

        curl_norm_b1, curl_norm_b2, curl_norm_b3 : array[float]
            FE coefficients c_ijk of the curl of the unit magnetic field as a 2-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for magnetic field evaluation (1-form and vector-field components)
    b1 = empty(3, dtype=float)
    bv = empty(3, dtype=float)

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # get local number of markers
    n_markers_loc = args_markers.n_markers

    # fmt: off
    #$ omp parallel private(b1, b_para, b_star, b_star_para, bv, curl_norm_b, density_const, det_df, df_inv, df_inv_t, dfm, eta1, eta2, eta3, fac, filling_m12, filling_m13, filling_m23, g_inv, ip, norm_b1, span1, span2, span3, thread_derham, v, weight)
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

        # marker positions
        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

        v = markers[3, ip]

        # b-field evaluation; 1form
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        eval_1form_spline_mpi(span1, span2, span3, thread_derham, b1_1, b1_2, b1_3, b1)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, thread_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3, thread_derham, curl_norm_b1, curl_norm_b2, curl_norm_b3, curl_norm_b)

        # evaluate Jacobian matrix, Jacobian determinant and inverse metric tensor
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # vector-field components bv = G^{-1} b1 (the 2-form is det_df * bv)
        linalg_kernels.matrix_vector(g_inv, b1, bv)

        # Bstar in vector-field components (2-form Bstar divided by det_df)
        b_star[:] = bv + epsilon * v * curl_norm_b / det_df

        # calculate b_para and b_star_para (both divided by det_df, which cancels in the ratio)
        b_para = linalg_kernels.scalar_dot(norm_b1, bv)

        b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate scaling constant
        density_const = 1 - b_para / b_star_para

        # marker weight
        weight = markers[5, ip]

        # common factor of the filling functions
        fac = -weight * density_const * ep_scale / epsilon

        if basis_u == 0:
            # filling functions: 2-form bx() = det_df * bvx()
            filling_m12 = -fac * bv[2] * det_df
            filling_m13 = +fac * bv[1] * det_df
            filling_m23 = -fac * bv[0] * det_df

            # call the appropriate matvec filler
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
            # filling functions: G^{-1} (2-form bx()) G^{-1} = b1x() / det_df
            filling_m12 = -fac * b1[2] / det_df
            filling_m13 = +fac * b1[1] / det_df
            filling_m23 = -fac * b1[0] / det_df

            # call the appropriate matvec filler
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
            # filling functions: (2-form bx()) / det_df**2 = bvx() / det_df
            filling_m12 = -fac * bv[2] / det_df
            filling_m13 = +fac * bv[1] / det_df
            filling_m23 = -fac * bv[0] / det_df

            # call the appropriate matvec filler
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
