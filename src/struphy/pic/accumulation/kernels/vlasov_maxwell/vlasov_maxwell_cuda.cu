#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

/**
 * Accumulate the matrix and vector of the full-orbit Vlasov-Maxwell (Vlasov-Ampere) coupling into V1, as in
 * vlasov_maxwell_kernels.vlasov_maxwell.
 *
 * Filling functions:
 *     A_p^{mu, nu} = w_p G^{-1}_{mu, nu}(eta_p),   G^{-1} = DF^{-1} DF^{-T},
 *     B_p^mu = w_p [DF^{-1}(eta_p) v_p]_mu;
 * one thread per marker row.
 *
 * @param args_markers Marker buffer (n_markers x n_cols, row-major); the weight is in column 6, as in pyccel.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param mat11 (1,1)-block data of the symmetric V1 -> V1 block matrix (the _data of a StencilMatrix, 6D, any
 *              strides); written with atomic additions.
 * @param mat12 (1,2)-block data; written atomically.
 * @param mat13 (1,3)-block data; written atomically.
 * @param mat22 (2,2)-block data; written atomically.
 * @param mat23 (2,3)-block data; written atomically.
 * @param mat33 (3,3)-block data; written atomically.
 * @param vec1 First component of the V1 vector data; written atomically.
 * @param vec2 Second component of the V1 vector data; written atomically.
 * @param vec3 Third component of the V1 vector data; written atomically.
 *
 * Holes (markers[ip, 0] == -1) are skipped; boundary particles are not, as in pyccel. The pyccel loop runs over all
 * rows (shape(markers)[0]), which is n_markers here. Many threads add to the same entries, so the summation order
 * (and the last bits of the result) differs from the serial pyccel loop.
 */
extern "C" __global__ void vlasov_maxwell(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                          Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                          Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                          Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate for metric coeffs (3 x 3 matrices as nine row-major entries)
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9];

    // allocate for filling
    double v[3], df_inv_times_v[3], filling_m[9], filling_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(ip, 0) == -1.) return;

    // marker positions
    double eta1 = args_markers.markers(ip, 0);
    double eta2 = args_markers.markers(ip, 1);
    double eta3 = args_markers.markers(ip, 2);

    // evaluate Jacobian, result in dfm
    struphy_cuda::df(eta1, eta2, eta3, args_domain, dfm);

    // compute shifted and stretched velocity
    v[0] = args_markers.markers(ip, 3);
    v[1] = args_markers.markers(ip, 4);
    v[2] = args_markers.markers(ip, 5);

    // filling functions
    struphy_cuda::matrix_inv(dfm, df_inv);
    struphy_cuda::transpose(df_inv, df_inv_t);
    struphy_cuda::matrix_matrix(df_inv, df_inv_t, g_inv);
    struphy_cuda::matrix_vector(df_inv, v, df_inv_times_v);

    // filling_m = w_p * DF^{-1} * DF^{-T}
    for (int k = 0; k < 9; ++k) filling_m[k] = args_markers.markers(ip, 6) * g_inv[k];

    // filling_v = w_p * DF^{-1} * \V
    for (int k = 0; k < 3; ++k) filling_v[k] = args_markers.markers(ip, 6) * df_inv_times_v[k];

    // call the appropriate matvec filler
    struphy_cuda::m_v_fill_b_v1_symm(args_derham, eta1, eta2, eta3, mat11, mat12, mat13, mat22, mat23, mat33,
                                     filling_m[0], filling_m[1], filling_m[2], filling_m[4], filling_m[5],
                                     filling_m[8], vec1, vec2, vec3, filling_v[0], filling_v[1], filling_v[2]);
}
