#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the symmetric matrix and the vector of CurrentCoupling6DCurrent, as in
 * cc_lin_mhd_6d_2_kernels.cc_lin_mhd_6d_2.
 *
 * With T_p = B2_x DF^{-1} (basis_u = 0), G^{-1} B2_x DF^{-1} (basis_u = 1) or B2_x DF^{-1} (basis_u = 2), where
 * B2_x a = B2 x a, the filling functions are
 *     A_p = w_p s_mat T_p T_p^T / det(DF)^{2k},   B_p = w_p s_vec T_p v_p / det(DF)^k,
 * with k = 1 for basis_u = 2 and k = 0 otherwise; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param mat11 (1,1)-block data of the symmetric u-space block matrix; written with atomic additions.
 * @param mat12 (1,2)-block data; written atomically.
 * @param mat13 (1,3)-block data; written atomically.
 * @param mat22 (2,2)-block data; written atomically.
 * @param mat23 (2,3)-block data; written atomically.
 * @param mat33 (3,3)-block data; written atomically.
 * @param vec1 First component of the u-space vector data; written atomically.
 * @param vec2 Second component of the vector data; written atomically.
 * @param vec3 Third component of the vector data; written atomically.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param basis_u Space of u: 0 (H1vec), 1 (H(curl)) or 2 (H(div)); other values accumulate nothing.
 * @param scale_mat Scaling factor of the matrix.
 * @param scale_vec Scaling factor of the vector.
 * @param boundary_cut Markers with eta1 < boundary_cut or eta1 > 1 - boundary_cut are not accumulated.
 *
 * Holes (markers[0, ip] == -1) are skipped. Many threads add to the same entries, so the summation order (and the
 * last bits of the result) differs from the pyccel loop.
 */
extern "C" __global__ void cc_lin_mhd_6d_2(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                           Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                           Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                           Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                           Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3,
                                           int basis_u, double scale_mat, double scale_vec, double boundary_cut) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate for magnetic field evaluation (3 x 3 matrices as nine row-major entries)
    double b[3], b_prod[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // allocate for metric coeffs
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9];

    // allocate for filling
    double filling_m[9], filling_v[3], v[3];
    double tmp1[9], tmp2[9], tmp_t[9], tmp_m[9], tmp_v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // boundary cut
    if (args_markers.markers(0, ip) < boundary_cut || args_markers.markers(0, ip) > 1.0 - boundary_cut) return;

    // marker positions
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // b-field evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
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

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // marker weight and velocity
    double weight = args_markers.markers(6, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    if (basis_u == 0) {
        // needed metric coefficients
        linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);

        // filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1)
        linalg_kernels::matrix_matrix(b_prod, df_inv, tmp1);

        linalg_kernels::transpose(tmp1, tmp_t);

        linalg_kernels::matrix_matrix(tmp1, tmp_t, tmp_m);
        linalg_kernels::matrix_vector(tmp1, v, tmp_v);

        for (int j = 0; j < 9; ++j) filling_m[j] = weight * tmp_m[j] * scale_mat;
        for (int j = 0; j < 3; ++j) filling_v[j] = weight * tmp_v[j] * scale_vec;

        // call the appropriate matvec filler
        particle_to_mat_kernels::m_v_fill_v0vec_symm(args_derham, scratch, span1, span2, span3, mat11, mat12, mat13,
                                                     mat22, mat23, mat33, filling_m[0], filling_m[1], filling_m[2],
                                                     filling_m[4], filling_m[5], filling_m[8], vec1, vec2, vec3,
                                                     filling_v[0], filling_v[1], filling_v[2]);
    } else if (basis_u == 1) {
        // needed metric coefficients
        linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);

        // filling functions tmp_m = tmp2 * tmp2^T and tmp_v = tmp2 * v, where tmp2 = G^(-1) * B^x * DF^(-1)
        linalg_kernels::matrix_matrix(g_inv, b_prod, tmp1);
        linalg_kernels::matrix_matrix(tmp1, df_inv, tmp2);

        linalg_kernels::transpose(tmp2, tmp_t);

        linalg_kernels::matrix_matrix(tmp2, tmp_t, tmp_m);
        linalg_kernels::matrix_vector(tmp2, v, tmp_v);

        for (int j = 0; j < 9; ++j) filling_m[j] = weight * tmp_m[j] * scale_mat;
        for (int j = 0; j < 3; ++j) filling_v[j] = weight * tmp_v[j] * scale_vec;

        // call the appropriate matvec filler
        particle_to_mat_kernels::m_v_fill_v1_symm(args_derham, scratch, span1, span2, span3, mat11, mat12, mat13, mat22,
                                                  mat23, mat33, filling_m[0], filling_m[1], filling_m[2], filling_m[4],
                                                  filling_m[5], filling_m[8], vec1, vec2, vec3, filling_v[0],
                                                  filling_v[1], filling_v[2]);
    } else if (basis_u == 2) {
        // needed metric coefficients
        linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);

        // filling functions tmp_m = tmp1 * tmp1^T and tmp_v = tmp1 * v, where tmp1 = B^x * DF^(-1) / det(DF)
        linalg_kernels::matrix_matrix(b_prod, df_inv, tmp1);

        linalg_kernels::transpose(tmp1, tmp_t);

        linalg_kernels::matrix_matrix(tmp1, tmp_t, tmp_m);
        linalg_kernels::matrix_vector(tmp1, v, tmp_v);

        for (int j = 0; j < 9; ++j) filling_m[j] = weight * tmp_m[j] * scale_mat / (det_df * det_df);
        for (int j = 0; j < 3; ++j) filling_v[j] = weight * tmp_v[j] * scale_vec / det_df;

        // call the appropriate matvec filler
        particle_to_mat_kernels::m_v_fill_v2_symm(args_derham, scratch, span1, span2, span3, mat11, mat12, mat13, mat22,
                                                  mat23, mat33, filling_m[0], filling_m[1], filling_m[2], filling_m[4],
                                                  filling_m[5], filling_m[8], vec1, vec2, vec3, filling_v[0],
                                                  filling_v[1], filling_v[2]);
    }
}
