#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the full pressure-coupling matrices and vectors of PressureCoupling6D into V1, as in
 * pc_lin_mhd_6d_full_kernels.pc_lin_mhd_6d_full.
 *
 * Filling functions, with V_p = v_p:
 *     V_p,a A_p^(mu, nu) V_p,b = w_p s [DF^{-1} DF^{-T}]_(mu, nu) V_p,a V_p,b,
 *     V_p,a B_p^mu = w_p s [DF^{-1} v_p]_mu V_p,a,
 * with s = ep_scale; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param mat11_11 ... mat33_11 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_x v_x.
 * @param mat11_12 ... mat33_12 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_x v_y.
 * @param mat11_13 ... mat33_13 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_x v_z.
 * @param mat11_22 ... mat33_22 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_y v_y.
 * @param mat11_23 ... mat33_23 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_y v_z.
 * @param mat11_33 ... mat33_33 Blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of the V1 -> V1 matrix of v_z v_z.
 * @param vec1_1, vec2_1, vec3_1 Components of the V1 vector of v_x.
 * @param vec1_2, vec2_2, vec3_2 Components of the V1 vector of v_y.
 * @param vec1_3, vec2_3, vec3_3 Components of the V1 vector of v_z.
 * @param ep_scale Scaling factor of the matrices and vectors.
 *
 * The kernel writes all matrices and vectors, with atomic additions.
 * Holes (markers[0, ip] == -1) are skipped. Many threads add to the same entries, so the summation order (and the
 * last bits of the result) differs from the pyccel loop. The 36 matrix views make the parameter list about 4.8 KB,
 * more than the 4 KB of older CUDA versions (CUDA 12.1 and newer allow 32 KB on Volta and newer GPUs).
 */
extern "C" __global__ void pc_lin_mhd_6d_full(
    MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain, Array6D<double> mat11_11,
    Array6D<double> mat12_11, Array6D<double> mat13_11, Array6D<double> mat22_11, Array6D<double> mat23_11,
    Array6D<double> mat33_11, Array6D<double> mat11_12, Array6D<double> mat12_12, Array6D<double> mat13_12,
    Array6D<double> mat22_12, Array6D<double> mat23_12, Array6D<double> mat33_12, Array6D<double> mat11_13,
    Array6D<double> mat12_13, Array6D<double> mat13_13, Array6D<double> mat22_13, Array6D<double> mat23_13,
    Array6D<double> mat33_13, Array6D<double> mat11_22, Array6D<double> mat12_22, Array6D<double> mat13_22,
    Array6D<double> mat22_22, Array6D<double> mat23_22, Array6D<double> mat33_22, Array6D<double> mat11_23,
    Array6D<double> mat12_23, Array6D<double> mat13_23, Array6D<double> mat22_23, Array6D<double> mat23_23,
    Array6D<double> mat33_23, Array6D<double> mat11_33, Array6D<double> mat12_33, Array6D<double> mat13_33,
    Array6D<double> mat22_33, Array6D<double> mat23_33, Array6D<double> mat33_33, Array3D<double> vec1_1,
    Array3D<double> vec2_1, Array3D<double> vec3_1, Array3D<double> vec1_2, Array3D<double> vec2_2,
    Array3D<double> vec3_2, Array3D<double> vec1_3, Array3D<double> vec2_3, Array3D<double> vec3_3, double ep_scale) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate for metric coeffs (3 x 3 matrices as nine row-major entries)
    double dfm[9], df_inv[9], df_inv_t[9];

    // allocate for filling
    double filling_m[9], filling_v[3], tmp1[9], v[3], tmp_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // marker positions
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs (pyccel also computes DF^T, which this kernel does not use)
    linalg_kernels::matrix_inv(dfm, df_inv);
    linalg_kernels::transpose(df_inv, df_inv_t);

    // filling functions
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    linalg_kernels::matrix_matrix(df_inv, df_inv_t, tmp1);
    linalg_kernels::matrix_vector(df_inv, v, tmp_v);

    double weight = args_markers.markers(6, ip);

    for (int j = 0; j < 9; ++j) filling_m[j] = weight * tmp1[j] * ep_scale;
    for (int j = 0; j < 3; ++j) filling_v[j] = weight * tmp_v[j] * ep_scale;

    // call the appropriate matvec filler
    particle_to_mat_kernels::m_v_fill_v1_pressure_full(
        args_derham, scratch, span1, span2, span3, mat11_11, mat12_11, mat13_11, mat22_11, mat23_11, mat33_11, mat11_12,
        mat12_12, mat13_12, mat22_12, mat23_12, mat33_12, mat11_13, mat12_13, mat13_13, mat22_13, mat23_13, mat33_13,
        mat11_22, mat12_22, mat13_22, mat22_22, mat23_22, mat33_22, mat11_23, mat12_23, mat13_23, mat22_23, mat23_23,
        mat33_23, mat11_33, mat12_33, mat13_33, mat22_33, mat23_33, mat33_33, filling_m[0], filling_m[1], filling_m[2],
        filling_m[4], filling_m[5], filling_m[8], vec1_1, vec2_1, vec3_1, vec1_2, vec2_2, vec3_2, vec1_3, vec2_3,
        vec3_3, filling_v[0], filling_v[1], filling_v[2], v[0], v[1], v[2]);
}
