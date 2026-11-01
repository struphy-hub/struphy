#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Evaluate mu_p (eta_p - eta_p^old) . (grad |B_0| + grad B_parallel)(eta_mid) per marker, as in
 * eval_gradB_ediff_kernels.eval_gradB_ediff, with eta_mid = mod((eta_p + eta_p^old) / 2, 1).
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices; eta_p^old is
 * stored at first_pusher_idx. One thread per marker.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param gradB1 Coefficients of the first component of grad |B_0| as a 1-form.
 * @param gradB2 Coefficients of the second component of grad |B_0| as a 1-form.
 * @param gradB3 Coefficients of the third component of grad |B_0| as a 1-form.
 * @param grad_PB_b1 Coefficients of the first component of grad B_parallel as a 1-form.
 * @param grad_PB_b2 Coefficients of the second component of grad B_parallel as a 1-form.
 * @param grad_PB_b3 Coefficients of the third component of grad B_parallel as a 1-form.
 * @param idx Marker column of the result.
 *
 * Holes (markers[0, ip] == -1) are skipped. The metric (g and its inverse) is evaluated but not used, as in the
 * pyccel kernel.
 */
extern "C" __global__ void eval_gradB_ediff(MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham,
                                            Array3D<double> gradB1, Array3D<double> gradB2, Array3D<double> gradB3,
                                            Array3D<double> grad_PB_b1, Array3D<double> grad_PB_b2,
                                            Array3D<double> grad_PB_b3, int idx) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    Array2D<double>& markers = args_markers.markers;
    int mu_idx = args_markers.mu_idx;
    int first_pusher_idx = args_markers.first_pusher_idx;

    // allocate metric coeffs
    double dfm[9], df_t[9], g[9], g_inv[9];

    // allocate for magnetic field evaluation
    double gradB[3], grad_PB_b[3], tmp[3], eta_mid[3], eta_diff[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (markers(0, ip) == -1.0) return;

    // marker positions, mid point; numpy.mod(eta, 1.0) (Fortran MODULO): the result has the sign of the divisor
    for (int j = 0; j < 3; ++j) {
        eta_mid[j] = (markers(j, ip) + markers(first_pusher_idx + j, ip)) / 2.0;
        eta_mid[j] = eta_mid[j] - floor(eta_mid[j]);
        eta_diff[j] = markers(j, ip) - markers(first_pusher_idx + j, ip);
    }

    double mu = markers(mu_idx, ip);

    // b-field evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta_mid[0], eta_mid[1], eta_mid[2], args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta_mid[0], eta_mid[1], eta_mid[2], args_domain, dfm);

    linalg_kernels::transpose(dfm, df_t);
    linalg_kernels::matrix_matrix(df_t, dfm, g);
    linalg_kernels::matrix_inv(g, g_inv);

    // gradB; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, gradB1, gradB2, gradB3,
                                                 gradB);

    // grad_PB_b; 1form
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, grad_PB_b1, grad_PB_b2,
                                                 grad_PB_b3, grad_PB_b);

    for (int j = 0; j < 3; ++j) tmp[j] = gradB[j] + grad_PB_b[j];

    markers(idx, ip) = linalg_kernels::scalar_dot(eta_diff, tmp);
    markers(idx, ip) *= mu;
}
