#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::shafranov_shift_cylinder_kernels {
/**
 * Evaluate the Shafranov shift mapping, as in shafranov_shift_cylinder_kernels.shafranov_shift.
 *
 * F_x = rx eta1 cos(2 pi eta2) + (1 - eta1^2) rx de, F_y = ry eta1 sin(2 pi eta2), F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param rx Axis length in x-direction.
 * @param ry Axis length in y-direction.
 * @param lz Length in z-direction.
 * @param de Shift factor, should be in [0, 0.1].
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void shafranov_shift(double eta1, double eta2, double eta3, double rx, double ry, double lz,
                                       double de, double* f_out) {
    f_out[0] = (eta1 * rx) * cos(2 * pi * eta2) + (1 - pow(eta1, 2)) * rx * de;
    f_out[1] = (eta1 * ry) * sin(2 * pi * eta2);
    f_out[2] = eta3 * lz;
}

/**
 * Evaluate the Jacobian of the Shafranov shift mapping, as in shafranov_shift_cylinder_kernels.shafranov_shift_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis (unused, as in pyccel).
 * @param rx Axis length in x-direction.
 * @param ry Axis length in y-direction.
 * @param lz Length in z-direction.
 * @param de Shift factor, should be in [0, 0.1].
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void shafranov_shift_df(double eta1, double eta2, double eta3, double rx, double ry, double lz,
                                          double de, double* df_out) {
    df_out[0] = rx * cos(2 * pi * eta2) - 2 * eta1 * rx * de;
    df_out[1] = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2);
    df_out[2] = 0.0;
    df_out[3] = ry * sin(2 * pi * eta2);
    df_out[4] = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::shafranov_shift_cylinder_kernels
