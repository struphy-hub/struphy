#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::perturbed_disk_cylinder_kernels {
/**
 * Evaluate the perturbed disk cylinder mapping, as in perturbed_disk_cylinder_kernels.perturbed_disk.
 *
 * R = r1 eta1 (1 + eps eta1^q cos(2 pi k eta2)), F_x = R cos(2 pi eta2), F_y = R sin(2 pi eta2), F_z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param r1 Radius of the unperturbed disk.
 * @param eps Amplitude of the perturbation.
 * @param k Number of petals.
 * @param lz Length in z-direction.
 * @param q Power of eta1 in the perturbation.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void perturbed_disk(double eta1, double eta2, double eta3, double r1, double eps, double k,
                                      double lz, double q, double* f_out) {
    double radius = r1 * eta1 * (1.0 + eps * pow(eta1, q) * cos(2 * pi * k * eta2));
    f_out[0] = radius * cos(2 * pi * eta2);
    f_out[1] = radius * sin(2 * pi * eta2);
    f_out[2] = lz * eta3;
}

/**
 * Evaluate the Jacobian of the perturbed disk cylinder mapping, as in perturbed_disk_cylinder_kernels.perturbed_disk_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param r1 Radius of the unperturbed disk.
 * @param eps Amplitude of the perturbation.
 * @param k Number of petals.
 * @param lz Length in z-direction.
 * @param q Power of eta1 in the perturbation.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void perturbed_disk_df(double eta1, double eta2, double r1, double eps, double k, double lz,
                                         double q, double* df_out) {
    double ck = cos(2 * pi * k * eta2);
    double sk = sin(2 * pi * k * eta2);
    double radius = r1 * eta1 * (1.0 + eps * pow(eta1, q) * ck);
    double dr1 = r1 * (1.0 + (q + 1.0) * eps * pow(eta1, q) * ck);
    double dr2 = -2 * pi * k * r1 * eps * pow(eta1, q + 1.0) * sk;
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
}  // namespace struphy_cuda::perturbed_disk_cylinder_kernels
