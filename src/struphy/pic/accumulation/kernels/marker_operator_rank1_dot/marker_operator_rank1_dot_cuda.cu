#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Apply a particle matrix with rank-one fillings F_p = c_p e_p e_p^T (MarkerOperator), as in
 * marker_operator_rank1_dot_kernels.marker_operator_rank1_dot.
 *
 * Deposits B_p = c_p (e_p . S_h(eta_p)) e_p, where e_p and c_p are read from the marker columns column to
 * column + 3 and S_h is the spline function of s; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker).
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments (unused; part of the common accumulation signature).
 * @param vec1 First component of the output vector data; written with atomic additions.
 * @param vec2 Second component of the output vector data; written with atomic additions.
 * @param vec3 Third component of the output vector data; written with atomic additions.
 * @param s1 Coefficients of the first component of s (with ghost regions).
 * @param s2 Coefficients of the second component of s.
 * @param s3 Coefficients of the third component of s.
 * @param basis_u Space of s and of the output: 0 (H1vec), 1 (H(curl)) or 2 (H(div)); other values deposit nothing.
 * @param column First of the four marker columns holding e_p (three entries) and c_p.
 *
 * Holes (markers[0, ip] == -1) are skipped.
 */
extern "C" __global__ void marker_operator_rank1_dot(MarkerArgs args_markers, DerhamArgs args_derham,
                                                     DomainArgs args_domain, Array3D<double> vec1,
                                                     Array3D<double> vec2, Array3D<double> vec3, Array3D<double> s1,
                                                     Array3D<double> s2, Array3D<double> s3, int basis_u,
                                                     int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    double s_h[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(args_markers.markers(0, ip), args_markers.markers(1, ip),
                                     args_markers.markers(2, ip), args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    double e1 = args_markers.markers(column, ip);
    double e2 = args_markers.markers(column + 1, ip);
    double e3 = args_markers.markers(column + 2, ip);

    if (basis_u == 0)
        evaluation_kernels_3d::eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, scratch, s1, s2, s3, s_h);
    else if (basis_u == 1)
        evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, s1, s2, s3, s_h);
    else if (basis_u == 2)
        evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, s1, s2, s3, s_h);
    else
        return;  // CUDA-only: pyccel computes fill from uninitialized s_h here and deposits nothing

    double fill = args_markers.markers(column + 3, ip) * (e1 * s_h[0] + e2 * s_h[1] + e3 * s_h[2]);

    if (basis_u == 0)
        particle_to_mat_kernels::vec_fill_v0vec(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                                fill * e1, fill * e2, fill * e3);
    else if (basis_u == 1)
        particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3, fill * e1,
                                             fill * e2, fill * e3);
    else if (basis_u == 2)
        particle_to_mat_kernels::vec_fill_v2(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3, fill * e1,
                                             fill * e2, fill * e3);
}
