#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * One stage of an explicit Runge-Kutta position push of PushEtaPC with U in (H^1)^3, as in
 * push_pc_eta_stage_H1vec_kernels.push_pc_eta_stage_H1vec.
 *
 * Solves d eta_p / dt = DF^{-1}(eta_p) v_p + U(eta_p) with constant velocity; one thread per marker.
 *
 * @param dt Time step.
 * @param stage Index of the current stage, 0 <= stage < n_stages.
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param u_1 Coefficients of the first component of U as a vector field in (H^1)^3.
 * @param u_2 Coefficients of the second component of U as a vector field.
 * @param u_3 Coefficients of the third component of U as a vector field.
 * @param use_perp_model Whether the third component of U is set to zero.
 * @param a Butcher coefficients a_stage (n_stages entries).
 * @param b Butcher weights b; n_stages is its length.
 * @param c Butcher nodes c (unused, as in the pyccel kernel).
 *
 * Holes and markers reflected in an earlier stage (first_pusher_idx == -1) and boundary particles (last column
 * == -2) are skipped, as in pyccel. The kinetic boundary conditions are applied after the update.
 */
extern "C" __global__ void push_pc_eta_stage_H1vec(double dt, int stage, MarkerArgs args_markers,
                                                   DomainArgs args_domain, DerhamArgs args_derham, Array3D<double> u_1,
                                                   Array3D<double> u_2, Array3D<double> u_3, bool use_perp_model,
                                                   Array1D<double> a, Array1D<double> b, Array1D<double> c) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate metric coeffs
    double dfm[9], dfinv[9];

    // marker velocity, U-field and intermediate stages in RK4
    double v[3], u[3], k[3], k_v[3];

    // get marker arguments
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_free_idx = args_markers.first_free_idx;

    // get number of stages
    int n_stages = b.shape[0];

    double last = stage == n_stages - 1 ? 1.0 : 0.0;

    // skip holes, markers reflected in an earlier stage (their start position is overwritten) and boundary particles
    if (args_markers.markers(first_pusher_idx, ip) == -1.0 ||
        args_markers.markers(args_markers.markers.shape[0] - 1, ip) == -2.0)
        return;

    double e1 = args_markers.markers(0, ip);
    double e2 = args_markers.markers(1, ip);
    double e3 = args_markers.markers(2, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    // ----------------- stage n in Runge-Kutta method -------------------
    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(e1, e2, e3, args_domain, dfm);

    // metric coeffs (pyccel also computes DF^{-T} and G^{-1}, which this kernel does not use)
    linalg_kernels::matrix_inv(dfm, dfinv);

    // pull-back of velocity
    linalg_kernels::matrix_vector(dfinv, v, k_v);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(e1, e2, e3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // U-field
    evaluation_kernels_3d::eval_vectorfield_spline_mpi(span1, span2, span3, args_derham, scratch, u_1, u_2, u_3, u);

    if (use_perp_model) u[2] = 0.0;

    // sum contribs
    for (int j = 0; j < 3; ++j) k[j] = k_v[j] + u[j];

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
