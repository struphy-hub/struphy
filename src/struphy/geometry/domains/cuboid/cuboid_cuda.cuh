#pragma once
namespace struphy_cuda {
/**
 * Evaluate the cuboid mapping, as in cuboid_kernels.cuboid.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param l1 Left boundary of the first axis.
 * @param r1 Right boundary of the first axis.
 * @param l2 Left boundary of the second axis.
 * @param r2 Right boundary of the second axis.
 * @param l3 Left boundary of the third axis.
 * @param r3 Right boundary of the third axis.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void cuboid(double eta1, double eta2, double eta3,
                             double l1, double r1, double l2, double r2, double l3, double r3,
                             double* f_out) {
    f_out[0] = l1 + (r1 - l1) * eta1;
    f_out[1] = l2 + (r2 - l2) * eta2;
    f_out[2] = l3 + (r3 - l3) * eta3;
}

/**
 * Evaluate the constant Jacobian, as in cuboid_kernels.cuboid_df.
 *
 * @param l1 Left boundary of the first axis.
 * @param r1 Right boundary of the first axis.
 * @param l2 Left boundary of the second axis.
 * @param r2 Right boundary of the second axis.
 * @param l3 Left boundary of the third axis.
 * @param r3 Right boundary of the third axis.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void cuboid_df(double l1, double r1, double l2, double r2, double l3, double r3,
                                double* df_out) {
    df_out[0] = r1 - l1;
    df_out[1] = 0.;
    df_out[2] = 0.;
    df_out[3] = 0.;
    df_out[4] = r2 - l2;
    df_out[5] = 0.;
    df_out[6] = 0.;
    df_out[7] = 0.;
    df_out[8] = r3 - l3;
}
}
