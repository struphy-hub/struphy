#include "struphy/bsplines/evaluation_kernels_3d.cuh"

using namespace struphy_cuda;

/**
 * Evaluate the drift-kinetic Hamiltonian, as in driftkinetic_hamiltonian_kernels.driftkinetic_hamiltonian.
 *
 * H(Z_p) = epsilon v_parallel^2 / 2 + epsilon mu_p |B|(eta_p) + phi(eta_p) at the weighted average
 * Z_{p,i} = alpha_i Z^{n+1,k}_{p,i} + (1 - alpha_i) Z^n_{p,i} (i = 1, ..., 4); one thread per marker.
 *
 * @param alpha Weights of the average (at least four entries): alpha[0:3] for eta, alpha[3] for v_parallel.
 * @param output_indices Marker column of the result in output_indices[0]; -1 skips the output.
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments (unused; part of the common evaluation signature).
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param epsilon Scaling parameter epsilon.
 * @param B_dot_b_coeffs Coefficients of the 0-form B . b (= |B|).
 * @param phi_coeffs Coefficients of the 0-form electrostatic potential phi.
 * @param evaluate_e_field Whether phi is evaluated (otherwise phi = 0).
 *
 * Holes (markers[0, ip] == -1) are skipped, as in pyccel.
 */
extern "C" __global__ void driftkinetic_hamiltonian(const double* alpha, const long long* output_indices,
                                                    MarkerArgs args_markers, DomainArgs args_domain,
                                                    DerhamArgs args_derham, double epsilon,
                                                    Array3D<double> B_dot_b_coeffs, Array3D<double> phi_coeffs,
                                                    bool evaluate_e_field) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // allocate stack arrays
    double eta_k[3], eta_n[3], eta[3];

    // get marker arguments
    int mu_idx = args_markers.mu_idx;
    int first_pusher_idx = args_markers.first_pusher_idx;
    int first_shift_idx = args_markers.first_shift_idx;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    for (int j = 0; j < 3; ++j) {
        eta_k[j] = args_markers.markers(j, ip) + args_markers.markers(first_shift_idx + j, ip);
        eta_n[j] = args_markers.markers(first_pusher_idx + j, ip);

        eta[j] = alpha[j] * eta_k[j] + (1.0 - alpha[j]) * eta_n[j];
        // numpy.mod(eta, 1.0) (Fortran MODULO): the result has the sign of the divisor
        eta[j] = eta[j] - floor(eta[j]);
    }

    double v_k = args_markers.markers(3, ip);
    double v_n = args_markers.markers(first_pusher_idx + 3, ip);
    double v = alpha[3] * v_k + (1.0 - alpha[3]) * v_n;

    double mu = args_markers.markers(mu_idx, ip);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta[0], eta[1], eta[2], args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    double phi;
    if (evaluate_e_field)
        phi = evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, phi_coeffs);
    else
        phi = 0.0;

    double B_dot_b =
        evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, B_dot_b_coeffs);

    // save
    if (output_indices[0] >= 0)
        args_markers.markers(output_indices[0], ip) = epsilon * v * v / 2.0 + epsilon * mu * B_dot_b + phi;
}
