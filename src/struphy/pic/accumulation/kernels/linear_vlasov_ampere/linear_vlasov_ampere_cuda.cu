#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the matrix and vector of the linear Vlasov-Ampere system (delta-f) into V1, as in
 * linear_vlasov_ampere_kernels.linear_vlasov_ampere.
 *
 * Filling functions (the factors alpha^2 kappa^2 / v_th^2 and alpha^2 kappa are applied by the propagator):
 *     A_p^{mu, nu} = f_0(eta_p, v_p) / (N s_0) [DF^{-1}(eta_p) v_p]_mu [DF^{-1}(eta_p) v_p]_nu,
 *     B_p^mu = w_p [DF^{-1}(eta_p) v_p]_mu;
 * one thread per marker row.
 *
 * @param args_markers Marker buffer (n_markers x n_cols, row-major) and the total marker number Np; the weight is in
 *                     column 6 and s_0 in column 7, as in pyccel.
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
 * @param f0_values Value of f0 for each marker row (n_markers entries).
 *
 * Holes (markers[ip, 0] == -1) and boundary particles (markers[ip, -1] == -2) are skipped. The pyccel loop runs over
 * all rows (shape(markers)[0]), which is n_markers here. Many threads add to the same entries, so the summation
 * order (and the last bits of the result) differs from the serial pyccel loop.
 */
extern "C" __global__ void linear_vlasov_ampere(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                                Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                                Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                                Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                                const double* f0_values) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;

    // get number of markers
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // total number of markers (weights are w_p = delta f_p / (N * s_0))
    int n_markers_tot = args_markers.Np;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(ip, 0) == -1. || args_markers.markers(ip, args_markers.markers.shape[1] - 1) == -2.)
        return;

    // allocate for metric coeffs (3 x 3 matrices as nine row-major entries)
    double dfm[9], df_inv[9];

    // allocate for filling
    double v[3], df_inv_v[3], filling_m[9], filling_v[3];

    // marker positions
    double eta1 = args_markers.markers(ip, 0);
    double eta2 = args_markers.markers(ip, 1);
    double eta3 = args_markers.markers(ip, 2);

    // get velocity
    v[0] = args_markers.markers(ip, 3);
    v[1] = args_markers.markers(ip, 4);
    v[2] = args_markers.markers(ip, 5);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // invert Jacobian matrix
    linalg_kernels::matrix_inv(dfm, df_inv);

    // compute DF^{-1} v
    linalg_kernels::matrix_vector(df_inv, v, df_inv_v);

    // filling_m = alpha^2 * kappa^2 * f0 / (N * s_0 * v_th^2) * (DF^{-1} v_p)_mu * (DF^{-1} v_p)_nu
    linalg_kernels::outer(df_inv_v, df_inv_v, filling_m);
    double factor = f0_values[ip] / (args_markers.markers(ip, 7) * n_markers_tot);
    for (int k = 0; k < 9; ++k) filling_m[k] *= factor;

    // filling_v = alpha^2 * kappa * w_p * DL^{-1} * v_p
    for (int k = 0; k < 3; ++k) filling_v[k] = args_markers.markers(ip, 6) * df_inv_v[k];

    // call the appropriate matvec filler
    particle_to_mat_kernels::m_v_fill_b_v1_symm(args_derham, eta1, eta2, eta3, mat11, mat12, mat13, mat22, mat23,
                                                mat33, filling_m[0], filling_m[1], filling_m[2], filling_m[4],
                                                filling_m[5], filling_m[8], vec1, vec2, vec3, filling_v[0],
                                                filling_v[1], filling_v[2]);
}
