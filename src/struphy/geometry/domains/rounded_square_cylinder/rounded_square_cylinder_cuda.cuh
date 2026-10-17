#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::rounded_square_cylinder_kernels {
/**
 * Evaluate the rounded square cylinder mapping, as in rounded_square_cylinder_kernels.rounded_square.
 *
 * rho = sqrt(2 / (1 + sqrt(1 + 4 eps cos(8 pi eta2)))), F_x = a eta1 rho cos(2 pi eta2),
 * F_y = a eta1 rho sin(2 pi eta2), F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param a Scale of the cross-section.
 * @param eps Squareness of the cross-section.
 * @param lz Length in z-direction.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void rounded_square(double eta1, double eta2, double eta3, double a, double eps, double lz,
                                      double* f_out) {
    double q = sqrt(1.0 + 4.0 * eps * cos(8 * pi * eta2));
    double rho = sqrt(2.0 / (1.0 + q));
    f_out[0] = a * eta1 * rho * cos(2 * pi * eta2);
    f_out[1] = a * eta1 * rho * sin(2 * pi * eta2);
    f_out[2] = lz * eta3;
}

/**
 * Evaluate the Jacobian of the rounded square cylinder mapping, as in rounded_square_cylinder_kernels.rounded_square_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param a Scale of the cross-section.
 * @param eps Squareness of the cross-section.
 * @param lz Length in z-direction.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void rounded_square_df(double eta1, double eta2, double a, double eps, double lz, double* df_out) {
    double q = sqrt(1.0 + 4.0 * eps * cos(8 * pi * eta2));
    double rho = sqrt(2.0 / (1.0 + q));
    double drho = 16 * pi * eps * sin(8 * pi * eta2) / (rho * (1.0 + q) * (1.0 + q) * q);
    double radius = a * eta1 * rho;
    double dr1 = a * rho;
    double dr2 = a * eta1 * drho;
    double c = cos(2 * pi * eta2);
    double s = sin(2 * pi * eta2);
    df_out[0] = dr1 * c;
    df_out[1] = dr2 * c - 2 * pi * radius * s;
    df_out[2] = 0.0;
    df_out[3] = dr1 * s;
    df_out[4] = dr2 * s + 2 * pi * radius * c;
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::rounded_square_cylinder_kernels
