#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::colella_kernels {
/**
 * Evaluate the Colella mapping, as in colella_kernels.colella.
 *
 * F_x = lx (eta1 + alpha sin(2 pi eta1) sin(2 pi eta2)), F_y = ly (eta2 + alpha sin(2 pi eta1) sin(2 pi eta2)),
 * F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param lx Length in x-direction.
 * @param ly Length in y-direction.
 * @param alpha Distortion factor.
 * @param lz Length in z-direction.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void colella(double eta1, double eta2, double eta3, double lx, double ly, double alpha, double lz,
                               double* f_out) {
    f_out[0] = lx * (eta1 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2));
    f_out[1] = ly * (eta2 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2));
    f_out[2] = lz * eta3;
}

/**
 * Evaluate the Jacobian of the Colella mapping, as in colella_kernels.colella_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param lx Length in x-direction.
 * @param ly Length in y-direction.
 * @param alpha Distortion factor.
 * @param lz Length in z-direction.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void colella_df(double eta1, double eta2, double lx, double ly, double alpha, double lz,
                                  double* df_out) {
    df_out[0] = lx * (1 + alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi);
    df_out[1] = lx * alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi;
    df_out[2] = 0.0;
    df_out[3] = ly * alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi;
    df_out[4] = ly * (1 + alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::colella_kernels
