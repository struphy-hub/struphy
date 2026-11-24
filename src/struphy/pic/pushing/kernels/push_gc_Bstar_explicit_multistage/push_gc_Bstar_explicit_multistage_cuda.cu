#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * One stage of an explicit Runge-Kutta push along B*, as in
 * push_gc_Bstar_explicit_multistage_kernels.push_gc_Bstar_explicit_multistage.
 *
 * d eta_p / dt = v_parallel B*^2 / (sqrt(g) B*_parallel) and d v_parallel / dt = B*^2 . E*^1 / (epsilon sqrt(g)
 * B*_parallel), with E* = -grad phi - epsilon mu grad |B| and B* = B + epsilon v_parallel curl b_0; one thread per
 * marker.
 *
 * @param dt Time step.
 * @param stage Index of the current stage, 0 <= stage < n_stages.
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param epsilon Scaling parameter epsilon.
 * @param grad_b_full_1 Coefficients of the first component of grad |B| as a 1-form.
 * @param grad_b_full_2 Coefficients of the second component of grad |B| as a 1-form.
 * @param grad_b_full_3 Coefficients of the third component of grad |B| as a 1-form.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param curl_unit_b2_1 Coefficients of the first component of curl b_0 as a 2-form.
 * @param curl_unit_b2_2 Coefficients of the second component of curl b_0 as a 2-form.
 * @param curl_unit_b2_3 Coefficients of the third component of curl b_0 as a 2-form.
 * @param B_dot_b_coeffs Coefficients of the 0-form B . b.
 * @param curl_unit_b_dot_b0 Coefficients of the 0-form (curl b_0) . b_0.
 * @param e_field_1 Coefficients of the first component of the electric field as a 1-form.
 * @param e_field_2 Coefficients of the second component of the electric field as a 1-form.
 * @param e_field_3 Coefficients of the third component of the electric field as a 1-form.
 * @param evaluate_e_field Whether the electric field is added to E*.
 * @param a Butcher coefficients a_stage (n_stages entries).
 * @param b Butcher weights b; n_stages is its length.
 * @param c Butcher nodes c (unused, as in the pyccel kernel).
 *
 * Holes (first_pusher_idx == -1) are skipped. The kinetic boundary conditions are applied after the update.
 */
extern "C" __global__ void push_gc_Bstar_explicit_multistage(
    double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham, double epsilon,
    Array3D<double> grad_b_full_1, Array3D<double> grad_b_full_2, Array3D<double> grad_b_full_3,
    Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3, Array3D<double> curl_unit_b2_1,
    Array3D<double> curl_unit_b2_2, Array3D<double> curl_unit_b2_3, Array3D<double> B_dot_b_coeffs,
    Array3D<double> curl_unit_b_dot_b0, Array3D<double> e_field_1, Array3D<double> e_field_2,
    Array3D<double> e_field_3, bool evaluate_e_field, Array1D<double> a, Array1D<double> b, Array1D<double> c) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate metric coeffs
    double dfm[9];

    // containers for fields
    double e_star[3], e_field[3] = {0.0, 0.0, 0.0}, b2[3], b_star[3];

    // intermediate k-vector
    double k[3];

    // get marker arguments
    int mu_idx = args_markers.mu_idx;
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_free_idx = args_markers.first_free_idx;

    // get number of stages
    int n_stages = b.shape[0];

    double last = stage == n_stages - 1 ? 1.0 : 0.0;

    // check if marker is a hole
    if (args_markers.markers(first_pusher_idx, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);
    double v = args_markers.markers(3, ip);
    double mu = args_markers.markers(mu_idx, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // compute E*
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, grad_b_full_1,
                                                 grad_b_full_2, grad_b_full_3, e_star);

    for (int j = 0; j < 3; ++j) e_star[j] *= -epsilon * mu;

    if (evaluate_e_field) {
        evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, e_field_1, e_field_2,
                                                     e_field_3, e_field);
        for (int j = 0; j < 3; ++j) e_star[j] += e_field[j];
    }

    // compute B*
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b2);

    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_unit_b2_1,
                                                 curl_unit_b2_2, curl_unit_b2_3, b_star);

    for (int j = 0; j < 3; ++j) {
        b_star[j] *= epsilon * v;
        b_star[j] += b2[j];
    }

    // compute B*_parallel
    double B_dot_b =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, B_dot_b_coeffs);

    double b_star_parallel =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_unit_b_dot_b0);

    b_star_parallel *= epsilon * v;
    b_star_parallel += B_dot_b;
    b_star_parallel *= det_df;

    // calculate k for eta
    for (int j = 0; j < 3; ++j) k[j] = b_star[j] / b_star_parallel * v;

    // calculate k_v for v
    double k_v = linalg_kernels::scalar_dot(b_star, e_star);
    k_v /= b_star_parallel * epsilon;

    // accumulation for last stage and update of positions (regular at a pole)
    pusher_utilities_kernels::update_position_stage(ip, k, dt, a(stage), b(stage), last, args_markers);

    // same for v_parallel
    args_markers.markers(first_free_idx + 3, ip) += dt * b(stage) * k_v;
    args_markers.markers(3, ip) = args_markers.markers(first_pusher_idx + 3, ip) + dt * a(stage) * k_v +
                                  last * args_markers.markers(first_free_idx + 3, ip);

    // apply kinetic boundary conditions
    pusher_utilities_kernels::apply_kinetic_bc_marker(ip, args_markers, args_domain, false);
}
