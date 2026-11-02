#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the antisymmetric density matrix of CurrentCoupling6DDensity, as in
 * cc_lin_mhd_6d_1_kernels.cc_lin_mhd_6d_1.
 *
 * Filling functions (upper blocks (1,2), (1,3), (2,3) only):
 *     basis_u = 0: A_p = -w_p s B2_x,
 *     basis_u = 1: A_p = -w_p s G^{-1} B2_x G^{-1},
 *     basis_u = 2: A_p = -w_p s B2_x / det(DF)^2,
 * where B2_x a = B2 x a and s = scale_mat; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param mat12 (1,2)-block data of the u-space block matrix; written with atomic additions.
 * @param mat13 (1,3)-block data; written atomically.
 * @param mat23 (2,3)-block data; written atomically.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param basis_u Space of u: 0 (H1vec), 1 (H(curl)) or 2 (H(div)); other values accumulate nothing.
 * @param scale_mat Scaling factor of the matrix.
 * @param boundary_cut Markers with eta1 < boundary_cut or eta1 > 1 - boundary_cut are not accumulated.
 *
 * Holes (markers[0, ip] == -1) are skipped. Many threads add to the same entries, so the summation order (and the
 * last bits of the result) differs from the pyccel loop.
 */
extern "C" __global__ void cc_lin_mhd_6d_1(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                           Array6D<double> mat12, Array6D<double> mat13, Array6D<double> mat23,
                                           Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3,
                                           int basis_u, double scale_mat, double boundary_cut) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate for magnetic field evaluation (3 x 3 matrices as nine row-major entries)
    double b[3], b_prod[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // allocate for metric coefficients
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9];

    // allocate some temporary buffers for filling
    double tmp1[9], tmp2[9];

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

    // evaluate Jacobian matrix and Jacobian determinant
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    double det_df = linalg_kernels::det(dfm);

    // marker weight
    double weight = args_markers.markers(6, ip);

    if (basis_u == 0) {
        // filling functions
        double filling_m12 = -weight * b_prod[1] * scale_mat;
        double filling_m13 = -weight * b_prod[2] * scale_mat;
        double filling_m23 = -weight * b_prod[5] * scale_mat;

        // call the appropriate matvec filler
        particle_to_mat_kernels::mat_fill_v0vec_asym(args_derham, scratch, span1, span2, span3, mat12, mat13, mat23,
                                                     filling_m12, filling_m13, filling_m23);
    } else if (basis_u == 1) {
        // filling functions
        linalg_kernels::matrix_inv_with_det(dfm, det_df, df_inv);
        linalg_kernels::transpose(df_inv, df_inv_t);
        linalg_kernels::matrix_matrix(df_inv, df_inv_t, g_inv);
        linalg_kernels::matrix_matrix(g_inv, b_prod, tmp1);
        linalg_kernels::matrix_matrix(tmp1, g_inv, tmp2);

        double filling_m12 = -weight * tmp2[1] * scale_mat;
        double filling_m13 = -weight * tmp2[2] * scale_mat;
        double filling_m23 = -weight * tmp2[5] * scale_mat;

        // call the appropriate matvec filler
        particle_to_mat_kernels::mat_fill_v1_asym(args_derham, scratch, span1, span2, span3, mat12, mat13, mat23,
                                                  filling_m12, filling_m13, filling_m23);
    } else if (basis_u == 2) {
        // filling functions
        double filling_m12 = -weight * b_prod[1] * scale_mat / (det_df * det_df);
        double filling_m13 = -weight * b_prod[2] * scale_mat / (det_df * det_df);
        double filling_m23 = -weight * b_prod[5] * scale_mat / (det_df * det_df);

        // call the appropriate matvec filler
        particle_to_mat_kernels::mat_fill_v2_asym(args_derham, scratch, span1, span2, span3, mat12, mat13, mat23,
                                                  filling_m12, filling_m13, filling_m23);
    }
}
