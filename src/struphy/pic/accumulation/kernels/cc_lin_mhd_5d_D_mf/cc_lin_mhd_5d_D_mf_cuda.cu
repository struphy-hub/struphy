#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Store the upper entries of the antisymmetric filling matrix F_p of the CurrentCoupling5DDensity matrix per marker,
 * as in cc_lin_mhd_5d_D_mf_kernels.cc_lin_mhd_5d_D_mf (matrix-free variant of cc_lin_mhd_5d_D).
 *
 * F_p = fac (B x) for basis_u = 0, fac (B x) / det(DF)^2 for basis_u = 2 and fac G^{-1} (B x) G^{-1} for
 * basis_u = 1, with fac = -w_p (1 - B_parallel / B*_parallel) ep_scale / epsilon; (F_12, F_13, F_23) are written to
 * the marker columns column, column + 1, column + 2. One thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 5.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param epsilon Scaling parameter epsilon.
 * @param ep_scale Scaling factor of the coupling.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param norm_b11 Coefficients of the first component of the unit vector b_0 as a 1-form.
 * @param norm_b12 Coefficients of the second component of b_0 as a 1-form.
 * @param norm_b13 Coefficients of the third component of b_0 as a 1-form.
 * @param curl_norm_b1 Coefficients of the first component of curl b_0 as a 2-form.
 * @param curl_norm_b2 Coefficients of the second component of curl b_0 as a 2-form.
 * @param curl_norm_b3 Coefficients of the third component of curl b_0 as a 2-form.
 * @param basis_u Space of U: 0 (H1vec), 1 (H(curl)) or 2 (H(div)).
 * @param column First of the three marker columns the filling is written to.
 *
 * Holes (markers[0, ip] == -1) are skipped.
 */
extern "C" __global__ void cc_lin_mhd_5d_D_mf(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                              double epsilon, double ep_scale, Array3D<double> b2_1,
                                              Array3D<double> b2_2, Array3D<double> b2_3, Array3D<double> norm_b11,
                                              Array3D<double> norm_b12, Array3D<double> norm_b13,
                                              Array3D<double> curl_norm_b1, Array3D<double> curl_norm_b2,
                                              Array3D<double> curl_norm_b3, int basis_u, int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    double b[3], b_star[3], norm_b1[3], curl_norm_b[3];
    double b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9], tmp1[9], tmp2[9];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    double v = args_markers.markers(3, ip);

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b);
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, norm_b11, norm_b12,
                                                 norm_b13, norm_b1);
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, curl_norm_b1,
                                                 curl_norm_b2, curl_norm_b3, curl_norm_b);

    // operator bx() as matrix (row-major)
    b_prod[1] = -b[2];
    b_prod[2] = +b[1];
    b_prod[3] = +b[2];
    b_prod[5] = -b[0];
    b_prod[6] = -b[1];
    b_prod[7] = +b[0];

    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);
    double det_df = linalg_kernels::det(dfm);

    for (int j = 0; j < 3; ++j) b_star[j] = b[j] + epsilon * v * curl_norm_b[j];
    double b_para = linalg_kernels::scalar_dot(norm_b1, b);
    double b_star_para = linalg_kernels::scalar_dot(norm_b1, b_star);

    // factor of the filling matrix
    double fac = -args_markers.markers(5, ip) * (1 - b_para / b_star_para) * ep_scale / epsilon;

    if (basis_u == 1) {
        linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);
        linalg_kernels::matrix_matrix(g_inv, b_prod, tmp1);
        linalg_kernels::matrix_matrix(tmp1, g_inv, tmp2);
        args_markers.markers(column, ip) = fac * tmp2[1];
        args_markers.markers(column + 1, ip) = fac * tmp2[2];
        args_markers.markers(column + 2, ip) = fac * tmp2[5];
    } else {
        if (basis_u == 2) fac = fac / (det_df * det_df);
        args_markers.markers(column, ip) = fac * b_prod[1];
        args_markers.markers(column + 1, ip) = fac * b_prod[2];
        args_markers.markers(column + 2, ip) = fac * b_prod[5];
    }
}
