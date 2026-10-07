"Accumulation kernel for full-orbit (6D) particles."

from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


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
        filling_m[:, :] = markers[ip, 6] * g_inv

        # filling_v = w_p * DF^{-1} * \V
        filling_v[:] = markers[ip, 6] * df_inv_times_v

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
