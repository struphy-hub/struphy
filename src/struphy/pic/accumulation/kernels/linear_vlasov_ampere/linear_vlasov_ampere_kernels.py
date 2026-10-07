"Accumulation kernel for full-orbit (6D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


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

    # total number of markers (weights are w_p = delta f_p / (N * s_0))
    n_markers_tot = args_markers.Np

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
        filling_m[:, :] *= f0_values[ip] / (markers[ip, 7] * n_markers_tot)

        # filling_v = alpha^2 * kappa * w_p * DL^{-1} * v_p
        filling_v[:] = markers[ip, 6] * df_inv_v

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
