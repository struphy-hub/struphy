#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Store the upper entries (F_12, F_13, F_23) of the antisymmetric filling F_p of the density matrix of
 * CurrentCoupling6DDensity per marker, as in cc_lin_mhd_6d_1_mf_kernels.cc_lin_mhd_6d_1_mf (the filling of
 * cc_lin_mhd_6d_1, for marker_operator_asym_dot); one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 6.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param basis_u Space of u: 0 (H1vec), 1 (H(curl)) or 2 (H(div)).
 * @param scale_mat Scaling factor of the matrix.
 * @param boundary_cut Markers with eta1 < boundary_cut or eta1 > 1 - boundary_cut get zero fillings.
 * @param column First of the three marker columns that receive F_12, F_13, F_23.
 *
 * Holes (markers[0, ip] == -1) are skipped. Each thread writes only its own marker, so no atomics are needed.
 */
extern "C" __global__ void cc_lin_mhd_6d_1_mf(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain,
                                              Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3,
                                              int basis_u, double scale_mat, double boundary_cut, int column) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // magnetic field and metric coefficients (3 x 3 matrices as nine row-major entries)
    double b[3], b_prod[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dfm[9], df_inv[9], df_inv_t[9], g_inv[9], tmp1[9], tmp2[9];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // boundary cut
    if (args_markers.markers(0, ip) < boundary_cut || args_markers.markers(0, ip) > 1.0 - boundary_cut) {
        args_markers.markers(column, ip) = 0.0;
        args_markers.markers(column + 1, ip) = 0.0;
        args_markers.markers(column + 2, ip) = 0.0;
        return;
    }

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    evaluation_kernels_3d::eval_2form_spline_mpi(scratch.span1, scratch.span2, scratch.span3, args_derham, scratch,
                                                 b2_1, b2_2, b2_3, b);

    // operator bx() as matrix
    b_prod[1] = -b[2];
    b_prod[2] = +b[1];
    b_prod[3] = +b[2];
    b_prod[5] = -b[0];
    b_prod[6] = -b[1];
    b_prod[7] = +b[0];

    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);
    double det_df = linalg_kernels::det(dfm);

    // factor of the filling matrix
    double fac = -args_markers.markers(6, ip) * scale_mat;

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
