#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * One stage of an explicit Runge-Kutta position push of PushEtaPC with U in H(curl), as in
 * push_pc_eta_stage_Hcurl_kernels.push_pc_eta_stage_Hcurl.
 *
 * Solves d eta_p / dt = DF^{-1}(eta_p) v_p + G^{-1}(eta_p) U^1(eta_p) with constant velocity; one thread per marker.
 *
 * @param dt Time step.
 * @param stage Index of the current stage, 0 <= stage < n_stages.
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param u_1 Coefficients of the first component of U as a 1-form.
 * @param u_2 Coefficients of the second component of U as a 1-form.
 * @param u_3 Coefficients of the third component of U as a 1-form.
 * @param use_perp_model Whether the third component of U is set to zero.
 * @param a Butcher coefficients a_stage (n_stages entries).
 * @param b Butcher weights b; n_stages is its length.
 * @param c Butcher nodes c (unused, as in the pyccel kernel).
 *
 * Holes (markers[0, ip] == -1) are skipped, as in pyccel. The kinetic boundary conditions are applied after the update.
 */
extern "C" __global__ void push_pc_eta_stage_Hcurl(double dt, int stage, MarkerArgs args_markers,
                                                   DomainArgs args_domain, DerhamArgs args_derham, Array3D<double> u_1,
                                                   Array3D<double> u_2, Array3D<double> u_3, bool use_perp_model,
                                                   Array1D<double> a, Array1D<double> b, Array1D<double> c) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate metric coeffs
    double dfm[9], dfinv[9], dfinv_t[9], ginv[9];

    // marker velocity, U-field and intermediate stages in RK4
    double v[3], u[3], k[3], k_v[3], k_u[3];

    // get marker arguments
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_free_idx = args_markers.first_free_idx;

    // get number of stages
    int n_stages = b.shape[0];

    double last = stage == n_stages - 1 ? 1.0 : 0.0;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double e1 = args_markers.markers(0, ip);
    double e2 = args_markers.markers(1, ip);
    double e3 = args_markers.markers(2, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    // ----------------- stage n in Runge-Kutta method -------------------
    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(e1, e2, e3, args_domain, dfm);

    // metric coeffs
    linalg_kernels::matrix_inv(dfm, dfinv);
    linalg_kernels::transpose(dfinv, dfinv_t);
    linalg_kernels::matrix_matrix(dfinv, dfinv_t, ginv);

    // pull-back of velocity
    linalg_kernels::matrix_vector(dfinv, v, k_v);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(e1, e2, e3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // U-field
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, u_1, u_2, u_3, u);

    if (use_perp_model) u[2] = 0.0;

    // transform to vector field
    linalg_kernels::matrix_vector(ginv, u, k_u);

    // sum contribs
    for (int j = 0; j < 3; ++j) k[j] = k_v[j] + k_u[j];

    for (int j = 0; j < 3; ++j) {
        // accum k
        args_markers.markers(first_free_idx + j, ip) += dt * b(stage) * k[j];
        // update markers for the next stage
        args_markers.markers(j, ip) = args_markers.markers(first_pusher_idx + j, ip) + dt * k[j] * a(stage) +
                                      last * args_markers.markers(first_free_idx + j, ip);
    }

    // apply kinetic boundary conditions
    pusher_utilities_kernels::apply_kinetic_bc_marker(ip, args_markers, args_domain, false);
}
