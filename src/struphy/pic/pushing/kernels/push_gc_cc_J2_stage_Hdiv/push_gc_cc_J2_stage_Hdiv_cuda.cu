#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * One stage of an explicit Runge-Kutta position push of CurrentCoupling5DGradB with U in H(div), as in
 * push_gc_cc_J2_stage_Hdiv_kernels.push_gc_cc_J2_stage_Hdiv.
 *
 * d eta_p / dt = -(b_0 x)(B x) U^2 / (sqrt(g) B*_parallel); one thread per marker.
 *
 * @param dt Time step.
 * @param stage Index of the current stage, 0 <= stage < n_stages.
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
 * @param a Butcher coefficients a_stage (n_stages entries).
 * @param b Butcher weights b; n_stages is its length.
 * @param c Butcher nodes c (unused, as in the pyccel kernel).
 *
 * Holes (markers[0, ip] == -1) and refilled markers (first_pusher_idx == -1) are skipped. The kinetic boundary
 * conditions are applied after the update.
 */
extern "C" __global__ void push_gc_cc_J2_stage_Hdiv(
    double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham, double epsilon,
    Array3D<double> b1, Array3D<double> b2, Array3D<double> b3, Array3D<double> norm_b11, Array3D<double> norm_b12,
    Array3D<double> norm_b13, Array3D<double> curl_norm_b1, Array3D<double> curl_norm_b2,
    Array3D<double> curl_norm_b3, Array3D<double> u1, Array3D<double> u2, Array3D<double> u3, Array1D<double> a,
    Array1D<double> b, Array1D<double> c) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate metric coeffs
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9];

    // containers for fields
    double tmp[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double norm_b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double e[3], u[3], bb[3], b_star[3], norm_b1[3], curl_norm_b[3];

    // get marker arguments
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_free_idx = args_markers.first_free_idx;

    // get number of stages
    int n_stages = b.shape[0];

    double last = stage == n_stages - 1 ? 1.0 : 0.0;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // if particle is refilled
    if (args_markers.markers(first_pusher_idx, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);
    double v = args_markers.markers(3, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs (g_inv is not used, as in the pyccel kernel)
    double det_df = linalg_kernels::det(dfm);
    linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
    linalg_kernels::transpose(df_inv, df_inv_t);
    linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b1, b2, b3, bb);

    // u; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, u1, u2, u3, u);

    // norm_b1; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, norm_b11, norm_b12,
                                                 norm_b13, norm_b1);

    // curl_norm_b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_norm_b1,
                                                 curl_norm_b2, curl_norm_b3, curl_norm_b);

    // operator bx() as matrix (row-major)
    b_prod[1] = -bb[2];
    b_prod[2] = +bb[1];
    b_prod[3] = +bb[2];
    b_prod[5] = -bb[0];
    b_prod[6] = -bb[1];
    b_prod[7] = +bb[0];

    norm_b_prod[1] = -norm_b1[2];
    norm_b_prod[2] = +norm_b1[1];
    norm_b_prod[3] = +norm_b1[2];
    norm_b_prod[5] = -norm_b1[0];
    norm_b_prod[6] = -norm_b1[1];
    norm_b_prod[7] = +norm_b1[0];

    // b_star; 2form
    for (int j = 0; j < 3; ++j) b_star[j] = bb[j] + curl_norm_b[j] * v * epsilon;

    // calculate abs_b_star_para
    double abs_b_star_para = linalg_kernels::scalar_dot(norm_b1, b_star);

    linalg_kernels::matrix_matrix(norm_b_prod, b_prod, tmp);
    linalg_kernels::matrix_vector(tmp, u, e);

    for (int j = 0; j < 3; ++j) {
        e[j] /= abs_b_star_para;
        e[j] /= det_df;
    }

    for (int j = 0; j < 3; ++j) {
        // accumulation for last stage
        args_markers.markers(first_free_idx + j, ip) -= dt * b(stage) * e[j];
        // update positions for intermediate stages or last stage
        args_markers.markers(j, ip) = args_markers.markers(first_pusher_idx + j, ip) - dt * a(stage) * e[j] +
                                      last * args_markers.markers(first_free_idx + j, ip);
    }

    // apply kinetic boundary conditions
    pusher_utilities_kernels::apply_kinetic_bc_marker(ip, args_markers, args_domain, false);
}
