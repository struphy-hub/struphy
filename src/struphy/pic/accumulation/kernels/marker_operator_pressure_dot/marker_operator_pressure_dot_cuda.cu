#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Apply the 3 x 3 block particle matrix of the pressure coupling (MarkerOperator with n_blocks=3), as in
 * marker_operator_pressure_dot_kernels.marker_operator_pressure_dot.
 *
 * Deposits B_p,a = v_p,a F_p sum_b v_p,b S_h,b(eta_p) into block a, where the symmetric F_p has the entries f11, f12,
 * f13, f22, f23, f33 in the marker columns column to column + 5 and S_h,b is the 1-form spline function of s_b; only
 * the first n_v velocity components are used; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker).
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments (unused; part of the common accumulation signature).
 * @param vec1_1 First component of output block 1; written with atomic additions.
 * @param vec2_1 Second component of output block 1; written atomically.
 * @param vec3_1 Third component of output block 1; written atomically.
 * @param vec1_2 First component of output block 2; written atomically.
 * @param vec2_2 Second component of output block 2; written atomically.
 * @param vec3_2 Third component of output block 2; written atomically.
 * @param vec1_3 First component of output block 3; written atomically if n_v == 3.
 * @param vec2_3 Second component of output block 3; written atomically if n_v == 3.
 * @param vec3_3 Third component of output block 3; written atomically if n_v == 3.
 * @param s1_1 Coefficients of the first component of the 1-form s_1 (with ghost regions).
 * @param s2_1 Coefficients of the second component of s_1.
 * @param s3_1 Coefficients of the third component of s_1.
 * @param s1_2 Coefficients of the first component of the 1-form s_2.
 * @param s2_2 Coefficients of the second component of s_2.
 * @param s3_2 Coefficients of the third component of s_2.
 * @param s1_3 Coefficients of the first component of the 1-form s_3 (read only if n_v == 3).
 * @param s2_3 Coefficients of the second component of s_3.
 * @param s3_3 Coefficients of the third component of s_3.
 * @param n_v Number of velocity components used: 2 (v_perp = (v_1, v_2, 0)) or 3 (v).
 * @param column First of the six marker columns holding f11, f12, f13, f22, f23, f33.
 *
 * Holes (markers[0, ip] == -1) are skipped.
 */
extern "C" __global__ void marker_operator_pressure_dot(
    MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain, Array3D<double> vec1_1,
    Array3D<double> vec2_1, Array3D<double> vec3_1, Array3D<double> vec1_2, Array3D<double> vec2_2,
    Array3D<double> vec3_2, Array3D<double> vec1_3, Array3D<double> vec2_3, Array3D<double> vec3_3,
    Array3D<double> s1_1, Array3D<double> s2_1, Array3D<double> s3_1, Array3D<double> s1_2, Array3D<double> s2_2,
    Array3D<double> s3_2, Array3D<double> s1_3, Array3D<double> s2_3, Array3D<double> s3_3, int n_v, int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    double v[3], s_h[3], s_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(args_markers.markers(0, ip), args_markers.markers(1, ip),
                                     args_markers.markers(2, ip), args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);
    if (n_v == 2) v[2] = 0.0;

    // s_v = sum_b v_b S_{h,b}
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, s1_1, s2_1, s3_1, s_h);
    for (int j = 0; j < 3; ++j) s_v[j] = v[0] * s_h[j];
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, s1_2, s2_2, s3_2, s_h);
    for (int j = 0; j < 3; ++j) s_v[j] += v[1] * s_h[j];
    if (n_v == 3) {
        evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, s1_3, s2_3, s3_3, s_h);
        for (int j = 0; j < 3; ++j) s_v[j] += v[2] * s_h[j];
    }

    double f11 = args_markers.markers(column, ip);
    double f12 = args_markers.markers(column + 1, ip);
    double f13 = args_markers.markers(column + 2, ip);
    double f22 = args_markers.markers(column + 3, ip);
    double f23 = args_markers.markers(column + 4, ip);
    double f33 = args_markers.markers(column + 5, ip);

    double y1 = f11 * s_v[0] + f12 * s_v[1] + f13 * s_v[2];
    double y2 = f12 * s_v[0] + f22 * s_v[1] + f23 * s_v[2];
    double y3 = f13 * s_v[0] + f23 * s_v[1] + f33 * s_v[2];

    particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_1, vec2_1, vec3_1, v[0] * y1,
                                         v[0] * y2, v[0] * y3);
    particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_2, vec2_2, vec3_2, v[1] * y1,
                                         v[1] * y2, v[1] * y3);
    if (n_v == 3)
        particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1_3, vec2_3, vec3_3,
                                             v[2] * y1, v[2] * y2, v[2] * y3);
}
