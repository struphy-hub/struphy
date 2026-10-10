#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Evaluate B*_parallel as a 3-form, as in bstar_parallel_3form_kernels.bstar_parallel_3form.
 *
 * B^{*3}_parallel(Z_p) = sqrt(g) (B . b + epsilon v_parallel,p ((curl b_0) . b_0)(eta_p)) at the weighted average
 * Z_{p,i} = alpha_i Z^{n+1,k}_{p,i} + (1 - alpha_i) Z^n_{p,i} (i = 1, ..., 4); one thread per marker row.
 *
 * @param alpha Weights of the average (at least four entries): alpha[0:3] for eta, alpha[3] for v_parallel.
 * @param output_indices Marker column of the result in output_indices[0]; -1 skips the output.
 * @param args_markers Marker buffer (n_markers x n_cols, row-major) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param epsilon Scaling parameter epsilon in front of v_parallel.
 * @param B_dot_b_coeffs Coefficients of the 0-form B . b.
 * @param curl_unit_b_dot_b0 Coefficients of the 0-form (curl b_0) . b_0.
 *
 * Holes (markers[ip, 0] == -1) are skipped, as in pyccel (which does not use valid_mks here).
 */
extern "C" __global__ void bstar_parallel_3form(const double* alpha, const long long* output_indices,
                                                MarkerArgs args_markers, DomainArgs args_domain,
                                                DerhamArgs args_derham, double epsilon,
                                                Array3D<double> B_dot_b_coeffs, Array3D<double> curl_unit_b_dot_b0) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate stack arrays
    double eta_k[3], eta_n[3], eta[3], dfm[9];

    // get marker arguments
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_shift_idx = args_markers.first_shift_idx;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(ip, 0) == -1.0) return;

    for (int j = 0; j < 3; ++j) {
        eta_k[j] = args_markers.markers(ip, j) + args_markers.markers(ip, first_shift_idx + j);
        eta_n[j] = args_markers.markers(ip, first_pusher_idx + j);

        eta[j] = alpha[j] * eta_k[j] + (1.0 - alpha[j]) * eta_n[j];
        // numpy.mod(eta, 1.0) (Fortran MODULO): the result has the sign of the divisor
        eta[j] = eta[j] - floor(eta[j]);
    }

    double v_k = args_markers.markers(ip, 3);
    double v_n = args_markers.markers(ip, first_pusher_idx + 3);
    double v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n;

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta[0], eta[1], eta[2], args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta[0], eta[1], eta[2], args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // compute B*_parallel
    double B_dot_b =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, B_dot_b_coeffs);

    double b_star_parallel =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_unit_b_dot_b0);

    b_star_parallel *= epsilon * v;
    b_star_parallel += B_dot_b;
    b_star_parallel *= det_df;

    if (output_indices[0] >= 0) args_markers.markers(ip, output_indices[0]) = b_star_parallel;
}
