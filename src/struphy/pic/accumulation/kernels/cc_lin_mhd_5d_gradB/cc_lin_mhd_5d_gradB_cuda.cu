#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Vector accumulation of CurrentCoupling5DGradB, as in cc_lin_mhd_5d_gradB_kernels.cc_lin_mhd_5d_gradB.
 *
 * B_p = w_p mu_p / B*_parallel (B x)(b_0 x) (grad B_parallel + grad B_parallel,eq) ep_scale (/ sqrt(g) for
 * basis_u = 2); one thread per marker.
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
 * @param grad_PB1 Coefficients of the first component of grad B_parallel as a 1-form.
 * @param grad_PB2 Coefficients of the second component of grad B_parallel as a 1-form.
 * @param grad_PB3 Coefficients of the third component of grad B_parallel as a 1-form.
 * @param grad_PBeq1 Coefficients of the first component of the equilibrium grad B_parallel as a 1-form.
 * @param grad_PBeq2 Coefficients of the second component of the equilibrium grad B_parallel as a 1-form.
 * @param grad_PBeq3 Coefficients of the third component of the equilibrium grad B_parallel as a 1-form.
 * @param basis_u Space of U: 0 (H1vec) or 2 (H(div)); other values deposit nothing.
 *
 * Holes (markers[0, ip] == -1) and refilled markers (first_pusher_idx == -1) are skipped.
 */
extern "C" __global__ void cc_lin_mhd_5d_gradB(
    MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain, Array3D<double> vec1,
    Array3D<double> vec2, Array3D<double> vec3, double epsilon, double ep_scale, Array3D<double> b1,
    Array3D<double> b2, Array3D<double> b3, Array3D<double> norm_b11, Array3D<double> norm_b12,
    Array3D<double> norm_b13, Array3D<double> curl_norm_b1, Array3D<double> curl_norm_b2,
    Array3D<double> curl_norm_b3, Array3D<double> grad_PB1, Array3D<double> grad_PB2, Array3D<double> grad_PB3,
    Array3D<double> grad_PBeq1, Array3D<double> grad_PBeq2, Array3D<double> grad_PBeq3, int basis_u) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    int first_pusher_idx = args_markers.first_pusher_idx;
    int mu_idx = args_markers.mu_idx;

    // allocate for magnetic field evaluation
    double b[3], b_star[3], curl_norm_b[3], norm_b1[3], grad_PB[3], grad_PBeq[3];
    double b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double norm_b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

    // allocate for metric coeffs (g_inv is not used, as in the pyccel kernel)
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9];

    // allocate for filling
    double filling_v[3], tmp[9], tmp_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // if particle is refilled
    if (args_markers.markers(first_pusher_idx, ip) == -1.0) return;

    // marker positions
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // marker weight and velocity
    double weight = args_markers.markers(5, ip);
    double v = args_markers.markers(3, ip);
    double mu = args_markers.markers(mu_idx, ip);

    // b-field evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // needed metric coefficients
    linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
    linalg_kernels::transpose(df_inv, df_inv_t);
    linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);

    // b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b1, b2, b3, b);

    // norm_b1; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, norm_b11, norm_b12,
                                                 norm_b13, norm_b1);

    // curl_norm_b; 2form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_norm_b1,
                                                 curl_norm_b2, curl_norm_b3, curl_norm_b);

    // grad_PB; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, grad_PB1, grad_PB2,
                                                 grad_PB3, grad_PB);

    // grad_PBeq; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, grad_PBeq1, grad_PBeq2,
                                                 grad_PBeq3, grad_PBeq);

    // b_star; 2form transformed into H1vec
    for (int j = 0; j < 3; ++j) b_star[j] = b[j] + curl_norm_b[j] * v * epsilon;

    // calculate abs_b_star_para
    double abs_b_star_para = linalg_kernels::scalar_dot(norm_b1, b_star);

    // operator bx() as matrix (row-major)
    b_prod[1] = -b[2];
    b_prod[2] = +b[1];
    b_prod[3] = +b[2];
    b_prod[5] = -b[0];
    b_prod[6] = -b[1];
    b_prod[7] = +b[0];

    norm_b_prod[1] = -norm_b1[2];
    norm_b_prod[2] = +norm_b1[1];
    norm_b_prod[3] = +norm_b1[2];
    norm_b_prod[5] = -norm_b1[0];
    norm_b_prod[6] = -norm_b1[1];
    norm_b_prod[7] = +norm_b1[0];

    // full gradient of the parallel magnetic field (perturbation + equilibrium)
    for (int j = 0; j < 3; ++j) grad_PB[j] += grad_PBeq[j];

    if (basis_u == 0) {
        linalg_kernels::matrix_matrix(b_prod, norm_b_prod, tmp);
        linalg_kernels::matrix_vector(tmp, grad_PB, tmp_v);

        for (int j = 0; j < 3; ++j) filling_v[j] = weight * tmp_v[j] * mu / abs_b_star_para * ep_scale;

        // call the appropriate matvec filler
        particle_to_mat_kernels::vec_fill_v0vec(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                                filling_v[0], filling_v[1], filling_v[2]);
    } else if (basis_u == 2) {
        linalg_kernels::matrix_matrix(b_prod, norm_b_prod, tmp);
        linalg_kernels::matrix_vector(tmp, grad_PB, tmp_v);

        for (int j = 0; j < 3; ++j)
            filling_v[j] = weight * tmp_v[j] * mu / abs_b_star_para / det_df * ep_scale;

        // call the appropriate matvec filler
        particle_to_mat_kernels::vec_fill_v2(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                             filling_v[0], filling_v[1], filling_v[2]);
    }
}
