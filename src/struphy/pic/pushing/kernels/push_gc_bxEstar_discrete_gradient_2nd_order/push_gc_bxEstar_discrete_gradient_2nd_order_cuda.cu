#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * One Picard iteration of the 2nd-order (Gonzalez) discrete-gradient push of eta for the b x E* drift, as in
 * push_gc_bxEstar_discrete_gradient_2nd_order_kernels.push_gc_bxEstar_discrete_gradient_2nd_order.
 *
 * eta^{n+1,k+1} = eta^n + dt b_0^1 x grad_I / (sqrt(g) B*_parallel), evaluated at the mid-point (eta^{n+1,k} + eta^n)/2,
 * where grad_I is the discrete gradient of H from the Hamiltonians at n and (n+1, k) (marker columns first_free_idx
 * and first_free_idx + 1, written by the init and evaluation kernels); one thread per marker.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param epsilon Scaling parameter epsilon.
 * @param unit_b1_1 Coefficients of the first component of the unit vector b_0 as a 1-form.
 * @param unit_b1_2 Coefficients of the second component of b_0 as a 1-form.
 * @param unit_b1_3 Coefficients of the third component of b_0 as a 1-form.
 * @param grad_b_full_1 Coefficients of the first component of grad |B| as a 1-form.
 * @param grad_b_full_2 Coefficients of the second component of grad |B| as a 1-form.
 * @param grad_b_full_3 Coefficients of the third component of grad |B| as a 1-form.
 * @param B_dot_b_coeffs Coefficients of the 0-form B . b.
 * @param curl_unit_b_dot_b0 Coefficients of the 0-form (curl b_0) . b_0.
 * @param e_field_1 Coefficients of the first component of the electric field as a 1-form.
 * @param e_field_2 Coefficients of the second component of the electric field as a 1-form.
 * @param e_field_3 Coefficients of the third component of the electric field as a 1-form.
 * @param evaluate_e_field Whether the electric field is added to grad H.
 *
 * Converged markers and holes (first_pusher_idx == -1) are skipped. The residual |eta^{n+1,k+1} - eta^{n+1,k}| is
 * written to residual_idx and the kinetic boundary conditions are applied after the update.
 */
extern "C" __global__ void push_gc_bxEstar_discrete_gradient_2nd_order(
    double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham, double epsilon,
    Array3D<double> unit_b1_1, Array3D<double> unit_b1_2, Array3D<double> unit_b1_3, Array3D<double> grad_b_full_1,
    Array3D<double> grad_b_full_2, Array3D<double> grad_b_full_3, Array3D<double> B_dot_b_coeffs,
    Array3D<double> curl_unit_b_dot_b0, Array3D<double> e_field_1, Array3D<double> e_field_2,
    Array3D<double> e_field_3, bool evaluate_e_field) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate metric coeffs and stack arrays
    double dfm[9];
    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    double grad_H[3], grad_I[3];
    double unit_b1[3], e_field[3] = {0.0, 0.0, 0.0}, Exb[3];

    // intermediate k-vector
    double k[3];

    // get marker arguments
    int mu_idx = args_markers.mu_idx;
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_shift_idx = args_markers.first_shift_idx;
    int residual_idx = args_markers.residual_idx;
    int first_free_idx = args_markers.first_free_idx;

    // check if marker is converged or a hole
    if (args_markers.markers(first_pusher_idx, ip) == -1.0) return;

    for (int j = 0; j < 3; ++j) {
        eta_k[j] = args_markers.markers(j, ip) + args_markers.markers(first_shift_idx + j, ip);
        eta_n[j] = args_markers.markers(first_pusher_idx + j, ip);

        eta_mid[j] = (eta_k[j] + eta_n[j]) / 2.0;
        // numpy.mod(eta_mid, 1.0) (Fortran MODULO): the result has the sign of the divisor
        eta_mid[j] = eta_mid[j] - floor(eta_mid[j]);
        eta_diff[j] = eta_k[j] - eta_n[j];
    }

    double v = args_markers.markers(3, ip);
    double mu = args_markers.markers(mu_idx, ip);

    // Hamiltonian at n (from init_kernel)
    double H_n = args_markers.markers(first_free_idx, ip);

    // Hamiltonian at (n+1, k) (from eval_kernel)
    double H_k = args_markers.markers(first_free_idx + 1, ip);

    // mid-point evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta_mid[0], eta_mid[1], eta_mid[2], args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // mid-point spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, unit_b1_1, unit_b1_2,
                                                 unit_b1_3, unit_b1);

    // compute grad_H at n + 1/2
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, grad_b_full_1,
                                                 grad_b_full_2, grad_b_full_3, grad_H);

    for (int j = 0; j < 3; ++j) grad_H[j] *= epsilon * mu;

    if (evaluate_e_field) {
        evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, e_field_1, e_field_2,
                                                     e_field_3, e_field);

        for (int j = 0; j < 3; ++j) {
            e_field[j] *= -1.0;
            grad_H[j] += e_field[j];
        }
    }

    // compute grad_I
    double dZ_dot_grad_H = linalg_kernels::scalar_dot(eta_diff, grad_H);
    double dZ_squared = linalg_kernels::scalar_dot(eta_diff, eta_diff);

    if (dZ_squared == 0.0) {
        for (int j = 0; j < 3; ++j) grad_I[j] = grad_H[j];
    } else {
        for (int j = 0; j < 3; ++j) grad_I[j] = grad_H[j] + eta_diff[j] * (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
    }

    // compute B*_parallel
    double B_dot_b =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, B_dot_b_coeffs);

    double b_star_parallel =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_unit_b_dot_b0);

    b_star_parallel *= epsilon * v;
    b_star_parallel += B_dot_b;
    b_star_parallel *= det_df;

    // calculate k
    linalg_kernels::cross(unit_b1, grad_I, Exb);

    for (int j = 0; j < 3; ++j) k[j] = Exb[j] / b_star_parallel;

    // accumulation for last stage
    for (int j = 0; j < 3; ++j) args_markers.markers(j, ip) = eta_n[j] + dt * k[j];

    // residual
    double d1 = args_markers.markers(0, ip) - eta_k[0];
    double d2 = args_markers.markers(1, ip) - eta_k[1];
    double d3 = args_markers.markers(2, ip) - eta_k[2];
    args_markers.markers(residual_idx, ip) = sqrt(d1 * d1 + d2 * d2 + d3 * d3);

    // apply kinetic boundary conditions
    pusher_utilities_kernels::apply_kinetic_bc_marker(ip, args_markers, args_domain, false);
}
