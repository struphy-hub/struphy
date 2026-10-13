#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::powered_elliptic_cylinder_kernels {
/**
 * Evaluate the powered elliptic cylinder mapping, as in powered_elliptic_cylinder_kernels.powered_ellipse.
 *
 * F_x = rx eta1^s cos(2 pi eta2), F_y = ry eta1^s sin(2 pi eta2), F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param rx Axis length in x-direction.
 * @param ry Axis length in y-direction.
 * @param lz Length in z-direction.
 * @param s Power of eta1.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void powered_ellipse(double eta1, double eta2, double eta3, double rx, double ry, double lz,
                                       double s, double* f_out) {
    f_out[0] = pow(eta1, s) * rx * cos(2 * pi * eta2);
    f_out[1] = pow(eta1, s) * ry * sin(2 * pi * eta2);
    f_out[2] = eta3 * lz;
}

/**
 * Evaluate the Jacobian of the powered elliptic cylinder mapping, as in
 * powered_elliptic_cylinder_kernels.powered_ellipse_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis (unused, as in pyccel).
 * @param rx Axis length in x-direction.
 * @param ry Axis length in y-direction.
 * @param lz Length in z-direction.
 * @param s Power of eta1.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void powered_ellipse_df(double eta1, double eta2, double eta3, double rx, double ry, double lz,
                                          double s, double* df_out) {
    df_out[0] = s * pow(eta1, s - 1) * rx * cos(2 * pi * eta2);
    df_out[1] = -2 * pi * pow(eta1, s) * rx * sin(2 * pi * eta2);
    df_out[2] = 0.0;
    df_out[3] = s * pow(eta1, s - 1) * ry * sin(2 * pi * eta2);
    df_out[4] = 2 * pi * pow(eta1, s) * ry * cos(2 * pi * eta2);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::powered_elliptic_cylinder_kernels
