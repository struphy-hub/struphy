#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the vectors of PressureCoupling6D into V1 and store the symmetric filling F_p of its matrices per marker,
 * as in pc_lin_mhd_6d_mf_kernels.pc_lin_mhd_6d_mf (the fillings of pc_lin_mhd_6d (n_v = 2) or pc_lin_mhd_6d_full
 * (n_v = 3), for marker_operator_pressure_dot).
 *
 * With B_p = w_p s DF^{-1} v_p, vector a gets v_p,a B_p (a <= n_v), and F_p = w_p s G^{-1}, with s = ep_scale; one
 * thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param vec1_1 First component of the V1 vector of v_x; written with atomic additions.
 * @param vec2_1 Second component of the vector of v_x; written atomically.
 * @param vec3_1 Third component of the vector of v_x; written atomically.
 * @param vec1_2 First component of the V1 vector of v_y; written atomically.
 * @param vec2_2 Second component of the vector of v_y; written atomically.
 * @param vec3_2 Third component of the vector of v_y; written atomically.
 * @param vec1_3 First component of the V1 vector of v_z; written atomically if n_v == 3.
 * @param vec2_3 Second component of the vector of v_z; written atomically if n_v == 3.
 * @param vec3_3 Third component of the vector of v_z; written atomically if n_v == 3.
 * @param ep_scale Scaling factor of the matrices and vectors.
 * @param n_v Number of velocity components used: 2 (perpendicular) or 3 (full).
 * @param column First of the six marker columns that receive F_11, F_12, F_13, F_22, F_23, F_33.
 *
 * Holes (markers[0, ip] == -1) are skipped. The summation order of the vectors differs from the pyccel loop.
 */
extern "C" __global__ void pc_lin_mhd_6d_mf(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                            Array3D<double> vec1_1, Array3D<double> vec2_1, Array3D<double> vec3_1,
                                            Array3D<double> vec1_2, Array3D<double> vec2_2, Array3D<double> vec3_2,
                                            Array3D<double> vec1_3, Array3D<double> vec2_3, Array3D<double> vec3_3,
                                            double ep_scale, int n_v, int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // metric coefficients (3 x 3 matrices as nine row-major entries) and velocity
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9], v[3], tmp_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);
    double weight = args_markers.markers(6, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);
    linalg_kernels::matrix_inv(dfm, df_inv);
    linalg_kernels::transpose(df_inv, df_inv_t);
    linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);
    linalg_kernels::matrix_vector(df_inv, v, tmp_v);

    double fac = weight * ep_scale;
    args_markers.markers(column, ip) = fac * g_inv[0];
    args_markers.markers(column + 1, ip) = fac * g_inv[1];
    args_markers.markers(column + 2, ip) = fac * g_inv[2];
    args_markers.markers(column + 3, ip) = fac * g_inv[4];
    args_markers.markers(column + 4, ip) = fac * g_inv[5];
    args_markers.markers(column + 5, ip) = fac * g_inv[8];

    particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_1, vec2_1, vec3_1,
                                         fac * tmp_v[0] * v[0], fac * tmp_v[1] * v[0], fac * tmp_v[2] * v[0]);
    particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_2, vec2_2, vec3_2,
                                         fac * tmp_v[0] * v[1], fac * tmp_v[1] * v[1], fac * tmp_v[2] * v[1]);
    if (n_v == 3)
        particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_3, vec2_3, vec3_3,
                                             fac * tmp_v[0] * v[2], fac * tmp_v[1] * v[2], fac * tmp_v[2] * v[2]);
}
