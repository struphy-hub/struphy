#pragma once
// Device versions of geometry/evaluation_kernels.py: the kind_map switch for the mapping F and its Jacobian DF,
// and the metric chain built on DF (det_df, df_inv, g, g_inv, select_metric_coeff), for every spline mapping
// (kind_map 0-2, from geometry/spline_mappings_kernels.cuh) and analytic mapping (kind_map 10-12, 20-22, 30-32).
// Matrices are stored as nine row-major entries (m[3 * i + j] is pyccel's m[i, j]). The temporary arrays
// tmp0..tmp3 of the pyccel helpers are kept as arguments (per-thread local arrays of the caller).
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/geometry/spline_mappings_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/geometry/domains/cuboid/cuboid_cuda.cuh"
#include "struphy/geometry/domains/orthogonal/orthogonal_cuda.cuh"
#include "struphy/geometry/domains/colella/colella_cuda.cuh"
#include "struphy/geometry/domains/hollow_cylinder/hollow_cylinder_cuda.cuh"
#include "struphy/geometry/domains/powered_elliptic_cylinder/powered_elliptic_cylinder_cuda.cuh"
#include "struphy/geometry/domains/hollow_torus/hollow_torus_cuda.cuh"
#include "struphy/geometry/domains/shafranov_shift_cylinder/shafranov_shift_cylinder_cuda.cuh"
#include "struphy/geometry/domains/shafranov_sqrt_cylinder/shafranov_sqrt_cylinder_cuda.cuh"
#include "struphy/geometry/domains/shafranov_dshaped_cylinder/shafranov_dshaped_cylinder_cuda.cuh"
namespace struphy_cuda::evaluation_kernels {
/**
 * Point-wise evaluation of (x, y, z) = F(eta1, eta2, eta3), as in geometry.evaluation_kernels.f.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier (kind_map), parameters (params) and, for spline mappings, knots, indices and
 *             control points.
 * @param f_out Output buffer for the three physical coordinates.
 *
 * Every spline (kind_map 0-2) and analytic mapping is supported; unknown identifiers trap.
 */
__device__ inline void f(double eta1, double eta2, double eta3, const DomainArgs& args, double* f_out) {
    switch (args.kind_map) {
        case 0:
            spline_mappings_kernels::spline_3d(eta1, eta2, eta3, args.degree, args.ind1, args.ind2, args.ind3, args,
                                               f_out);
            return;
        case 1:
            spline_mappings_kernels::spline_2d_straight(eta1, eta2, eta3, args.degree, args.ind1, args.ind2, args,
                                                        args.params[0], f_out);
            return;
        case 2:
            spline_mappings_kernels::spline_2d_torus(eta1, eta2, eta3, args.degree, args.ind1, args.ind2, args,
                                                     args.params[0], f_out);
            return;
        case 10:
            cuboid_kernels::cuboid(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2], args.params[3],
                                   args.params[4], args.params[5], f_out);
            return;
        case 11:
            orthogonal_kernels::orthogonal(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2],
                                           args.params[3], f_out);
            return;
        case 12:
            colella_kernels::colella(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2], args.params[3],
                                     f_out);
            return;
        case 20:
            hollow_cylinder_kernels::hollow_cyl(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2],
                                                args.params[3], f_out);
            return;
        case 21:
            powered_elliptic_cylinder_kernels::powered_ellipse(eta1, eta2, eta3, args.params[0], args.params[1],
                                                               args.params[2], args.params[3], f_out);
            return;
        case 22:
            hollow_torus_kernels::hollow_torus(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2],
                                               args.params[3], args.params[4], args.params[5], f_out);
            return;
        case 30:
            shafranov_shift_cylinder_kernels::shafranov_shift(eta1, eta2, eta3, args.params[0], args.params[1],
                                                              args.params[2], args.params[3], f_out);
            return;
        case 31:
            shafranov_sqrt_cylinder_kernels::shafranov_sqrt(eta1, eta2, eta3, args.params[0], args.params[1],
                                                            args.params[2], args.params[3], f_out);
            return;
        case 32:
            shafranov_dshaped_cylinder_kernels::shafranov_dshaped(eta1, eta2, eta3, args.params[0], args.params[1],
                                                                  args.params[2], args.params[3], args.params[4],
                                                                  args.params[5], args.params[6], f_out);
            return;
        default:
            asm("trap;");  // unknown kind_map
    }
}

/**
 * Point-wise evaluation of the Jacobian matrix DF = (dF_i/deta_j), as in geometry.evaluation_kernels.df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier (kind_map), parameters (params) and, for spline mappings, knots, indices and
 *             control points.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 *
 * Every spline (kind_map 0-2) and analytic mapping is supported; unknown identifiers trap.
 */
__device__ inline void df(double eta1, double eta2, double eta3, const DomainArgs& args, double* df_out) {
    switch (args.kind_map) {
        case 0:
            spline_mappings_kernels::spline_3d_df(eta1, eta2, eta3, args.degree, args.ind1, args.ind2, args.ind3, args,
                                                  df_out);
            return;
        case 1:
            spline_mappings_kernels::spline_2d_straight_df(eta1, eta2, args.degree, args.ind1, args.ind2, args,
                                                           args.params[0], df_out);
            return;
        case 2:
            spline_mappings_kernels::spline_2d_torus_df(eta1, eta2, eta3, args.degree, args.ind1, args.ind2, args,
                                                        args.params[0], df_out);
            return;
        case 10:
            cuboid_kernels::cuboid_df(args.params[0], args.params[1], args.params[2], args.params[3], args.params[4],
                                      args.params[5], df_out);
            return;
        case 11:
            orthogonal_kernels::orthogonal_df(eta1, eta2, args.params[0], args.params[1], args.params[2],
                                              args.params[3], df_out);
            return;
        case 12:
            colella_kernels::colella_df(eta1, eta2, args.params[0], args.params[1], args.params[2], args.params[3],
                                        df_out);
            return;
        case 20:
            hollow_cylinder_kernels::hollow_cyl_df(eta1, eta2, args.params[0], args.params[1], args.params[2],
                                                   args.params[3], df_out);
            return;
        case 21:
            powered_elliptic_cylinder_kernels::powered_ellipse_df(eta1, eta2, eta3, args.params[0], args.params[1],
                                                                  args.params[2], args.params[3], df_out);
            return;
        case 22:
            hollow_torus_kernels::hollow_torus_df(eta1, eta2, eta3, args.params[0], args.params[1], args.params[2],
                                                  args.params[3], args.params[4], args.params[5], df_out);
            return;
        case 30:
            shafranov_shift_cylinder_kernels::shafranov_shift_df(eta1, eta2, eta3, args.params[0], args.params[1],
                                                                 args.params[2], args.params[3], df_out);
            return;
        case 31:
            shafranov_sqrt_cylinder_kernels::shafranov_sqrt_df(eta1, eta2, eta3, args.params[0], args.params[1],
                                                               args.params[2], args.params[3], df_out);
            return;
        case 32:
            shafranov_dshaped_cylinder_kernels::shafranov_dshaped_df(eta1, eta2, eta3, args.params[0], args.params[1],
                                                                     args.params[2], args.params[3], args.params[4],
                                                                     args.params[5], args.params[6], df_out);
            return;
        default:
            asm("trap;");  // unknown kind_map
    }
}

/**
 * Point-wise evaluation of the Jacobian determinant det(DF), as in geometry.evaluation_kernels.det_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier and parameters.
 * @param tmp1 Temporary 3x3 matrix (nine entries), overwritten with DF.
 * @return detdf, the Jacobian determinant.
 */
__device__ inline double det_df(double eta1, double eta2, double eta3, const DomainArgs& args, double* tmp1) {
    df(eta1, eta2, eta3, args, tmp1);
    double detdf = linalg_kernels::det(tmp1);
    return detdf;
}

/**
 * Point-wise evaluation of the inverse Jacobian matrix DF^(-1), as in geometry.evaluation_kernels.df_inv.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier and parameters.
 * @param tmp1 Temporary 3x3 matrix (nine entries), overwritten with DF.
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 * @param dfinv_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void df_inv(double eta1, double eta2, double eta3, const DomainArgs& args, double* tmp1,
                              bool avoid_round_off, double* dfinv_out) {
    df(eta1, eta2, eta3, args, tmp1);
    linalg_kernels::matrix_inv(tmp1, dfinv_out);

    // set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if (avoid_round_off) {
        switch (args.kind_map) {
            case 1:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 2:
                dfinv_out[8] = 0.0;  // [2, 2]
                break;
            case 10:
                dfinv_out[1] = 0.0;  // [0, 1]
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[3] = 0.0;  // [1, 0]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 11:
                dfinv_out[1] = 0.0;  // [0, 1]
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[3] = 0.0;  // [1, 0]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 12:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 20:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 21:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 22:
                dfinv_out[8] = 0.0;  // [2, 2]
                break;
            case 30:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 31:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
            case 32:
                dfinv_out[2] = 0.0;  // [0, 2]
                dfinv_out[5] = 0.0;  // [1, 2]
                dfinv_out[6] = 0.0;  // [2, 0]
                dfinv_out[7] = 0.0;  // [2, 1]
                break;
        }
    }
}

/**
 * Point-wise evaluation of the metric tensor G = DF^T DF, as in geometry.evaluation_kernels.g.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier and parameters.
 * @param tmp1 Temporary 3x3 matrix (nine entries), overwritten with DF.
 * @param tmp2 Temporary 3x3 matrix (nine entries), overwritten with DF^T.
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 * @param g_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void g(double eta1, double eta2, double eta3, const DomainArgs& args, double* tmp1, double* tmp2,
                         bool avoid_round_off, double* g_out) {
    df(eta1, eta2, eta3, args, tmp1);
    linalg_kernels::transpose(tmp1, tmp2);
    linalg_kernels::matrix_matrix(tmp2, tmp1, g_out);

    // set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if (avoid_round_off) {
        switch (args.kind_map) {
            case 1:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 2:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 10:
                g_out[1] = 0.0;  // [0, 1]
                g_out[2] = 0.0;  // [0, 2]
                g_out[3] = 0.0;  // [1, 0]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 11:
                g_out[1] = 0.0;  // [0, 1]
                g_out[2] = 0.0;  // [0, 2]
                g_out[3] = 0.0;  // [1, 0]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 12:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 20:
                g_out[1] = 0.0;  // [0, 1]
                g_out[2] = 0.0;  // [0, 2]
                g_out[3] = 0.0;  // [1, 0]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 21:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 22:
                // straight field line coordinates
                if (args.params[3] == 1.0) {
                    g_out[2] = 0.0;  // [0, 2]
                    g_out[5] = 0.0;  // [1, 2]
                    g_out[6] = 0.0;  // [2, 0]
                    g_out[7] = 0.0;  // [2, 1]
                }
                // equal angle coordinates
                else {
                    g_out[1] = 0.0;  // [0, 1]
                    g_out[2] = 0.0;  // [0, 2]
                    g_out[3] = 0.0;  // [1, 0]
                    g_out[5] = 0.0;  // [1, 2]
                    g_out[6] = 0.0;  // [2, 0]
                    g_out[7] = 0.0;  // [2, 1]
                }
                break;
            case 30:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 31:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
            case 32:
                g_out[2] = 0.0;  // [0, 2]
                g_out[5] = 0.0;  // [1, 2]
                g_out[6] = 0.0;  // [2, 0]
                g_out[7] = 0.0;  // [2, 1]
                break;
        }
    }
}

/**
 * Point-wise evaluation of the inverse metric tensor G^(-1), as in geometry.evaluation_kernels.g_inv.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args Mapping identifier and parameters.
 * @param tmp1 Temporary 3x3 matrix (nine entries), overwritten with DF.
 * @param tmp2 Temporary 3x3 matrix (nine entries), overwritten with DF^T.
 * @param tmp3 Temporary 3x3 matrix (nine entries), overwritten with G.
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 * @param ginv_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void g_inv(double eta1, double eta2, double eta3, const DomainArgs& args, double* tmp1, double* tmp2,
                             double* tmp3, bool avoid_round_off, double* ginv_out) {
    g(eta1, eta2, eta3, args, tmp1, tmp2, avoid_round_off, tmp3);
    linalg_kernels::matrix_inv(tmp3, ginv_out);

    // set known (analytical) zero components manually to zero to avoid round-off error remainders!
    if (avoid_round_off) {
        switch (args.kind_map) {
            case 1:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 2:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 10:
                ginv_out[1] = 0.0;  // [0, 1]
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[3] = 0.0;  // [1, 0]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 11:
                ginv_out[1] = 0.0;  // [0, 1]
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[3] = 0.0;  // [1, 0]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 12:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 20:
                ginv_out[1] = 0.0;  // [0, 1]
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[3] = 0.0;  // [1, 0]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 21:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 22:
                // straight field line coordinates
                if (args.params[3] == 1.0) {
                    ginv_out[2] = 0.0;  // [0, 2]
                    ginv_out[5] = 0.0;  // [1, 2]
                    ginv_out[6] = 0.0;  // [2, 0]
                    ginv_out[7] = 0.0;  // [2, 1]
                }
                // equal angle coordinates
                else {
                    ginv_out[1] = 0.0;  // [0, 1]
                    ginv_out[2] = 0.0;  // [0, 2]
                    ginv_out[3] = 0.0;  // [1, 0]
                    ginv_out[5] = 0.0;  // [1, 2]
                    ginv_out[6] = 0.0;  // [2, 0]
                    ginv_out[7] = 0.0;  // [2, 1]
                }
                break;
            case 30:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 31:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
            case 32:
                ginv_out[2] = 0.0;  // [0, 2]
                ginv_out[5] = 0.0;  // [1, 2]
                ginv_out[6] = 0.0;  // [2, 0]
                ginv_out[7] = 0.0;  // [2, 1]
                break;
        }
    }
}

/**
 * Point-wise evaluation of one metric coefficient, as in geometry.evaluation_kernels.select_metric_coeff.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param kind_coeff Which coefficient: -1 identity, 0 mapping F, 1 DF, 2 det(DF), 3 DF^(-1), 4 G, 5 G^(-1).
 * @param args Mapping identifier and parameters.
 * @param tmp0 Temporary vector (three entries).
 * @param tmp1 Temporary 3x3 matrix (nine entries).
 * @param tmp2 Temporary 3x3 matrix (nine entries).
 * @param tmp3 Temporary 3x3 matrix (nine entries).
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 * @param out Output 3x3 matrix stored as nine row-major entries. Vectors (kind_coeff -1 and 0) go into the
 *            first column, the determinant (kind_coeff 2) into out[0, 0]; other entries are left unchanged.
 */
__device__ inline void select_metric_coeff(double eta1, double eta2, double eta3, int kind_coeff,
                                           const DomainArgs& args, double* tmp0, double* tmp1, double* tmp2,
                                           double* tmp3, bool avoid_round_off, double* out) {
    // identity map
    if (kind_coeff == -1) {
        out[0] = eta1;
        out[3] = eta2;
        out[6] = eta3;
    }
    // mapping F
    else if (kind_coeff == 0) {
        f(eta1, eta2, eta3, args, tmp0);
        out[0] = tmp0[0];
        out[3] = tmp0[1];
        out[6] = tmp0[2];
    }
    // Jacobian matrix DF
    else if (kind_coeff == 1) {
        df(eta1, eta2, eta3, args, out);
    }
    // Jacobian determinant det(dF)
    else if (kind_coeff == 2) {
        out[0] = det_df(eta1, eta2, eta3, args, tmp1);
    }
    // inverse Jacobian matrix DF^(-1)
    else if (kind_coeff == 3) {
        df_inv(eta1, eta2, eta3, args, tmp1, avoid_round_off, out);
    }
    // metric tensor G = DF^T * DF
    else if (kind_coeff == 4) {
        g(eta1, eta2, eta3, args, tmp1, tmp2, avoid_round_off, out);
    }
    // inverse metric tensor G^(-1) = DF^(-1) * DF^(-T)
    else if (kind_coeff == 5) {
        g_inv(eta1, eta2, eta3, args, tmp1, tmp2, tmp3, avoid_round_off, out);
    }
}
}  // namespace struphy_cuda::evaluation_kernels
