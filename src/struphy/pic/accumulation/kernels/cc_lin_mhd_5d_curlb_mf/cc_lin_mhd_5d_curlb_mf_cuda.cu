#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulation of CurrentCoupling5DCurlb (matrix-free), as in cc_lin_mhd_5d_curlb_mf_kernels.cc_lin_mhd_5d_curlb_mf.
 *
 * Deposits B_p = c_p e_p and stores e_p, c_p in the marker columns column to column + 3, where
 * c_p = w_p v_parallel^2 ep_scale and e_p = B^2 x (curl b_0)^2 / B*_parallel (/ sqrt(g) for basis_u = 2);
 * one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 5.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param vec1 First component of the vector data (V2 or H1vec); written with atomic additions.
 * @param vec2 Second component of the vector data; written with atomic additions.
 * @param vec3 Third component of the vector data; written with atomic additions.
 * @param epsilon Scaling parameter epsilon.
 * @param ep_scale Scaling factor of the coupling.
 * @param b1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param norm_b11 Coefficients of the first component of the unit vector b_0 as a 1-form.
 * @param norm_b12 Coefficients of the second component of b_0 as a 1-form.
 * @param norm_b13 Coefficients of the third component of b_0 as a 1-form.
 * @param curl_norm_b1 Coefficients of the first component of curl b_0 as a 2-form.
 * @param curl_norm_b2 Coefficients of the second component of curl b_0 as a 2-form.
 * @param curl_norm_b3 Coefficients of the third component of curl b_0 as a 2-form.
 * @param basis_u Space of U: 0 (H1vec) or 2 (H(div)); other values store the columns but deposit nothing.
 * @param column First of the four marker columns e_p, c_p are written to.
 *
 * Holes (markers[0, ip] == -1) are skipped.
 */
extern "C" __global__ void cc_lin_mhd_5d_curlb_mf(MarkerArgs args_markers, DerhamArgs args_derham,
                                                  DomainArgs args_domain, Array3D<double> vec1, Array3D<double> vec2,
                                                  Array3D<double> vec3, double epsilon, double ep_scale,
                                                  Array3D<double> b1, Array3D<double> b2, Array3D<double> b3,
                                                  Array3D<double> norm_b11, Array3D<double> norm_b12,
                                                  Array3D<double> norm_b13, Array3D<double> curl_norm_b1,
                                                  Array3D<double> curl_norm_b2, Array3D<double> curl_norm_b3,
                                                  int basis_u, int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    double b[3], bfull_star[3], norm_b1[3], curl_norm_b[3], dfm[9], e[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    double weight = args_markers.markers(5, ip);
    double v = args_markers.markers(3, ip);

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);
    double det_df = linalg_kernels::det(dfm);

    // total magnetic field (2-form), unit vector b0 (1-form) and its curl (2-form)
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b1, b2, b3, b);
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, norm_b11, norm_b12,
                                                 norm_b13, norm_b1);
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_norm_b1,
                                                 curl_norm_b2, curl_norm_b3, curl_norm_b);

    for (int j = 0; j < 3; ++j) bfull_star[j] = b[j] + curl_norm_b[j] * v * epsilon;
    double abs_b_star_para = linalg_kernels::scalar_dot(norm_b1, bfull_star);

    // e = B x curl(b0) / B*_par (/ sqrt(g) for 2-forms)
    linalg_kernels::cross(b, curl_norm_b, e);
    for (int j = 0; j < 3; ++j) e[j] = e[j] / abs_b_star_para;
    if (basis_u == 2) {
        for (int j = 0; j < 3; ++j) e[j] = e[j] / det_df;
    }

    double c = weight * v * v * ep_scale;

    args_markers.markers(column, ip) = e[0];
    args_markers.markers(column + 1, ip) = e[1];
    args_markers.markers(column + 2, ip) = e[2];
    args_markers.markers(column + 3, ip) = c;

    if (basis_u == 0)
        particle_to_mat_kernels::vec_fill_v0vec(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3, c * e[0],
                                                c * e[1], c * e[2]);
    else if (basis_u == 2)
        particle_to_mat_kernels::vec_fill_v2(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3, c * e[0],
                                             c * e[1], c * e[2]);
}
