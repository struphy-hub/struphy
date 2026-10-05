#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

/**
 * One stage of an explicit Runge-Kutta position push, as in push_eta_stage_kernels.push_eta_stage.
 *
 * Solves d eta_p / dt = DF^{-1}(eta_p) v_p with constant velocity, one thread per marker row.
 *
 * @param dt Time step.
 * @param stage Index of the current stage, 0 <= stage < n_stages.
 * @param args_markers Marker buffer (n_markers x n_cols, row-major) and column indices.
 * @param args_domain Mapping arguments; only Cuboid (kind_map == 10) is supported.
 * @param a Butcher coefficients a_stage (n_stages entries).
 * @param b Butcher weights b (n_stages entries).
 * @param c Butcher nodes c (n_stages entries; unused, as in the pyccel kernel).
 * @param n_stages Number of stages; a raw device pointer carries no length.
 *
 * Holes (first_init_idx == -1) and boundary particles (last column == -2) are skipped.
 * The kinetic boundary conditions are applied per marker after the update.
 */
extern "C" __global__ void push_eta_stage(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                          const double* a, const double* b, const double* c, int n_stages) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    int first_init_idx = args_markers.first_init_idx;
    int first_free_idx = args_markers.first_free_idx;
    double last = stage == n_stages - 1 ? 1. : 0.;

    // check if marker is a hole or a boundary particle
    if (MARKER(args_markers, ip, first_init_idx) == -1. || MARKER(args_markers, ip, args_markers.n_cols - 1) == -2.)
        return;

    double e1 = MARKER(args_markers, ip, 0);
    double e2 = MARKER(args_markers, ip, 1);
    double e3 = MARKER(args_markers, ip, 2);
    double v[3], dfm[9], dfinv[9], k[3];
    for (int j = 0; j < 3; ++j) v[j] = MARKER(args_markers, ip, 3 + j);

    // evaluate Jacobian, result in dfm
    struphy_cuda::df(e1, e2, e3, args_domain, dfm);

    // evaluate inverse Jacobian matrix
    struphy_cuda::matrix_inv(dfm, dfinv);

    // pull-back of velocity
    struphy_cuda::matrix_vector(dfinv, v, k);

    for (int j = 0; j < 3; ++j) {
        // accumulation for last stage
        MARKER(args_markers, ip, first_free_idx + j) += dt * b[stage] * k[j];
        // update positions for intermediate stages or last stage
        MARKER(args_markers, ip, j) = MARKER(args_markers, ip, first_init_idx + j) + dt * a[stage] * k[j] +
                                      last * MARKER(args_markers, ip, first_free_idx + j);
    }

    // apply kinetic boundary conditions
    struphy_cuda::apply_kinetic_bc_marker(ip, args_markers, args_domain, false);
}
