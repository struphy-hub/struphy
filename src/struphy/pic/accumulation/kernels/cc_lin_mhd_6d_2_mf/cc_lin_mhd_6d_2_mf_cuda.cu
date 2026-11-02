#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the vector of CurrentCoupling6DCurrent and store the symmetric filling F_p of its matrix per marker, as in
 * cc_lin_mhd_6d_2_mf_kernels.cc_lin_mhd_6d_2_mf (the fillings of cc_lin_mhd_6d_2, for marker_operator_symm_dot).
 *
 * With T_p = G^{-1} B2_x DF^{-1} (basis_u = 1) or B2_x DF^{-1} (otherwise), F_p = w_p s_mat T_p T_p^T / det(DF)^{2k}
 * and B_p = w_p s_vec T_p v_p / det(DF)^k, with k = 1 for basis_u = 2 and k = 0 otherwise; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param vec1 First component of the u-space vector data; written with atomic additions.
 * @param vec2 Second component of the vector data; written atomically.
 * @param vec3 Third component of the vector data; written atomically.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param basis_u Space of u: 0 (H1vec), 1 (H(curl)) or 2 (H(div)); other values store F_p but deposit nothing.
 * @param scale_mat Scaling factor of the matrix.
 * @param scale_vec Scaling factor of the vector.
 * @param boundary_cut Markers with eta1 < boundary_cut or eta1 > 1 - boundary_cut get zero fillings.
 * @param column First of the six marker columns that receive F_11, F_12, F_13, F_22, F_23, F_33.
 *
 * Holes (markers[0, ip] == -1) are skipped. The summation order of the vector differs from the pyccel loop.
 */
extern "C" __global__ void cc_lin_mhd_6d_2_mf(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                              Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                              Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3,
                                              int basis_u, double scale_mat, double scale_vec, double boundary_cut,
                                              int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // magnetic field, velocity and metric coefficients (3 x 3 matrices as nine row-major entries)
    double b[3], b_prod[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, v[3];
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9], tmp1[9], tmp2[9], tmp_t[9], tmp_m[9], tmp_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // boundary cut
    if (args_markers.markers(0, ip) < boundary_cut || args_markers.markers(0, ip) > 1.0 - boundary_cut) {
        for (int j = 0; j < 6; ++j) args_markers.markers(column + j, ip) = 0.0;
        return;
    }

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b);

    // operator bx() as matrix
    b_prod[1] = -b[2];
    b_prod[2] = +b[1];
    b_prod[3] = +b[2];
    b_prod[5] = -b[0];
    b_prod[6] = -b[1];
    b_prod[7] = +b[0];

    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);
    double det_df = linalg_kernels::det(dfm);

    double weight = args_markers.markers(6, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);

    // tmp2 = T_p
    if (basis_u == 1) {
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);
        linalg_kernels::matrix_matrix(g_inv, b_prod, tmp1);
        linalg_kernels::matrix_matrix(tmp1, df_inv, tmp2);
    } else {
        linalg_kernels::matrix_matrix(b_prod, df_inv, tmp2);
    }

    linalg_kernels::transpose(tmp2, tmp_t);
    linalg_kernels::matrix_matrix(tmp2, tmp_t, tmp_m);
    linalg_kernels::matrix_vector(tmp2, v, tmp_v);

    double fac_m = weight * scale_mat;
    double fac_v = weight * scale_vec;
    if (basis_u == 2) {
        fac_m = fac_m / (det_df * det_df);
        fac_v = fac_v / det_df;
    }

    args_markers.markers(column, ip) = fac_m * tmp_m[0];
    args_markers.markers(column + 1, ip) = fac_m * tmp_m[1];
    args_markers.markers(column + 2, ip) = fac_m * tmp_m[2];
    args_markers.markers(column + 3, ip) = fac_m * tmp_m[4];
    args_markers.markers(column + 4, ip) = fac_m * tmp_m[5];
    args_markers.markers(column + 5, ip) = fac_m * tmp_m[8];

    if (basis_u == 0)
        particle_to_mat_kernels::vec_fill_v0vec(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                                fac_v * tmp_v[0], fac_v * tmp_v[1], fac_v * tmp_v[2]);
    else if (basis_u == 1)
        particle_to_mat_kernels::vec_fill_v1(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                             fac_v * tmp_v[0], fac_v * tmp_v[1], fac_v * tmp_v[2]);
    else if (basis_u == 2)
        particle_to_mat_kernels::vec_fill_v2(args_derham, scratch, span1, span2, span3, vec1, vec2, vec3,
                                             fac_v * tmp_v[0], fac_v * tmp_v[1], fac_v * tmp_v[2]);
}
