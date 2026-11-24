#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Velocity update of CurrentCoupling5DCurlb with U in H(div), as in push_gc_cc_J1_Hdiv_kernels.push_gc_cc_J1_Hdiv.
 *
 * v_parallel += dt v_parallel (B^2 x U^2 / sqrt(g)) . (curl b_0)^2 / B*_parallel; one thread per marker.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param epsilon Scaling parameter epsilon.
 * @param b1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param norm_b11 Coefficients of the first component of the unit vector b_0 as a 1-form.
 * @param norm_b12 Coefficients of the second component of b_0 as a 1-form.
 * @param norm_b13 Coefficients of the third component of b_0 as a 1-form.
 * @param curl_norm_b1 Coefficients of the first component of curl b_0 as a 2-form.
 * @param curl_norm_b2 Coefficients of the second component of curl b_0 as a 2-form.
 * @param curl_norm_b3 Coefficients of the third component of curl b_0 as a 2-form.
 * @param u1 Coefficients of the first component of U as a 2-form.
 * @param u2 Coefficients of the second component of U as a 2-form.
 * @param u3 Coefficients of the third component of U as a 2-form.
 *
 * Holes (markers[0, ip] == -1) are skipped, as in pyccel.
 */
extern "C" __global__ void push_gc_cc_J1_Hdiv(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                              DerhamArgs args_derham, double epsilon, Array3D<double> b1,
                                              Array3D<double> b2, Array3D<double> b3, Array3D<double> norm_b11,
                                              Array3D<double> norm_b12, Array3D<double> norm_b13,
                                              Array3D<double> curl_norm_b1, Array3D<double> curl_norm_b2,
                                              Array3D<double> curl_norm_b3, Array3D<double> u1, Array3D<double> u2,
                                              Array3D<double> u3) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate metric coeffs
    double dfm[9];

    // containers for fields
    double e[3], u[3], b[3], b_star[3], norm_b1[3], curl_norm_b[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);
    double v = args_markers.markers(3, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs
    double det_df = linalg_kernels::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b1, b2, b3, b);

    // u; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, u1, u2, u3, u);

    // norm_b1; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, norm_b11, norm_b12,
                                                 norm_b13, norm_b1);

    // curl_norm_b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_norm_b1,
                                                 curl_norm_b2, curl_norm_b3, curl_norm_b);

    // b_star; 2form
    for (int j = 0; j < 3; ++j) b_star[j] = b[j] + curl_norm_b[j] * v * epsilon;

    // calculate 3form abs_b_star_para
    double abs_b_star_para = linalg_kernels::scalar_dot(norm_b1, b_star);

    // transform u into H1vec
    for (int j = 0; j < 3; ++j) u[j] = u[j] / det_df;

    // electric field E(1) = B(2) X U(0)
    linalg_kernels::cross(b, u, e);

    // curl_norm_b dot electric field
    double temp = linalg_kernels::scalar_dot(e, curl_norm_b);

    args_markers.markers(3, ip) += temp / abs_b_star_para * v * dt;
}
