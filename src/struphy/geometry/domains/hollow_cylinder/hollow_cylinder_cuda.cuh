#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::hollow_cylinder_kernels {
/**
 * Evaluate the hollow cylinder mapping, as in hollow_cylinder_kernels.hollow_cyl.
 *
 * F_x = (a1 + (a2 - a1) eta1) cos(2 pi eta2 / poc), F_y = (a1 + (a2 - a1) eta1) sin(2 pi eta2 / poc),
 * F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param a1 Inner radius.
 * @param a2 Outer radius.
 * @param lz Length in z-direction.
 * @param poc Periodicity in the second direction.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void hollow_cyl(double eta1, double eta2, double eta3, double a1, double a2, double lz, double poc,
                                  double* f_out) {
    double da = a2 - a1;
    f_out[0] = (a1 + eta1 * da) * cos(2 * pi * eta2 / poc);
    f_out[1] = (a1 + eta1 * da) * sin(2 * pi * eta2 / poc);
    f_out[2] = lz * eta3;
}

/**
 * Evaluate the Jacobian of the hollow cylinder mapping, as in hollow_cylinder_kernels.hollow_cyl_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param a1 Inner radius.
 * @param a2 Outer radius.
 * @param lz Length in z-direction.
 * @param poc Periodicity in the second direction.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void hollow_cyl_df(double eta1, double eta2, double a1, double a2, double lz, double poc,
                                     double* df_out) {
    double da = a2 - a1;
    df_out[0] = da * cos(2 * pi * eta2 / poc);
    df_out[1] = -2 * pi / poc * (a1 + eta1 * da) * sin(2 * pi * eta2 / poc);
    df_out[2] = 0.0;
    df_out[3] = da * sin(2 * pi * eta2 / poc);
    df_out[4] = 2 * pi / poc * (a1 + eta1 * da) * cos(2 * pi * eta2 / poc);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::hollow_cylinder_kernels
