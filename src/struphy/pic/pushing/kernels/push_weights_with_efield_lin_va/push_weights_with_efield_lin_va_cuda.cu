#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

/**
 * Weight update of the linear Vlasov-Ampere system (delta-f), as in
 * push_weights_with_efield_lin_va_kernels.push_weights_with_efield_lin_va.
 *
 * w_p += kappa * dt / (2 * N * s_0 * v_th^2) * f_0 * (DF^{-1} v_p) . e(eta_p), with e the 1-form spline field
 * (the sum of the old and new electric field); one thread per marker row.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_markers x n_cols, row-major), total marker number Np and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param e1_1 Coefficients of the first component of the electric 1-form.
 * @param e1_2 Coefficients of the second component of the electric 1-form.
 * @param e1_3 Coefficients of the third component of the electric 1-form.
 * @param f0_values Value of f0 for each marker row (n_markers entries).
 * @param kappa Coupling strength between particles and fields.
 * @param vth Thermal velocity.
 *
 * Holes and boundary particles are skipped.
 */
extern "C" __global__ void push_weights_with_efield_lin_va(double dt, int stage, MarkerArgs args_markers,
                                                           DomainArgs args_domain, DerhamArgs args_derham,
                                                           Array3D<double> e1_1, Array3D<double> e1_2,
                                                           Array3D<double> e1_3, const double* f0_values,
                                                           double kappa, double vth) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // total number of markers (weights are w_p = delta f_p / (N * s_0))
    int n_markers_tot = args_markers.Np;

    if (args_markers.markers(ip, 0) == -1. || args_markers.markers(ip, args_markers.markers.shape[1] - 1) == -2.) return;

    // position
    double eta1 = args_markers.markers(ip, 0);
    double eta2 = args_markers.markers(ip, 1);
    double eta3 = args_markers.markers(ip, 2);
    double v[3], dfm[9], df_inv[9], df_inv_v[3], e_vec[3];

    // get velocity
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(ip, 3 + j);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    struphy_cuda::SplineScratch scratch;
    struphy_cuda::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // Compute Jacobian matrix
    struphy_cuda::df(eta1, eta2, eta3, args_domain, dfm);

    // invert Jacobian matrix
    struphy_cuda::matrix_inv(dfm, df_inv);

    // compute DF^{-1} v
    struphy_cuda::matrix_vector(df_inv, v, df_inv_v);

    // E-field (1-form)
    struphy_cuda::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, e1_1, e1_2, e1_3, e_vec);

    // w_{n+1} = w_n + kappa * dt / (2 * N * s_0 * v_th^2) * f_0 * ( DF^{-1} v_p ) \cdot ( e_{n+1} + e_n )
    double update = (df_inv_v[0] * e_vec[0] + df_inv_v[1] * e_vec[1] + df_inv_v[2] * e_vec[2]) * f0_values[ip] *
                    kappa * dt / (2 * n_markers_tot * args_markers.markers(ip, 7) * (vth * vth));
    args_markers.markers(ip, 6) += update;
}
