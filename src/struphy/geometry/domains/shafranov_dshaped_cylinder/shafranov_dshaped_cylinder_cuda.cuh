#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda::shafranov_dshaped_cylinder_kernels {
/**
 * Evaluate the D-shaped Shafranov mapping, as in shafranov_dshaped_cylinder_kernels.shafranov_dshaped.
 *
 * x = r0 (1 + (1 - eta1^2) dx + eta1 eg cos(2 pi eta2 + arcsin(dg) eta1 sin(2 pi eta2))),
 * y = r0 ((1 - eta1^2) dy + eta1 eg kg sin(2 pi eta2)), z = lz eta3.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param r0 Base radius.
 * @param lz Length in z-direction.
 * @param dx Shafranov shift in x-direction.
 * @param dy Shafranov shift in y-direction.
 * @param dg Triangularity delta = sin(alpha).
 * @param eg Inverse aspect ratio epsilon.
 * @param kg Ellipticity kappa.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void shafranov_dshaped(double eta1, double eta2, double eta3, double r0, double lz, double dx,
                                         double dy, double dg, double eg, double kg, double* f_out) {
    f_out[0] =
        r0 * (1 + (1 - pow(eta1, 2)) * dx + eg * eta1 * cos(2 * pi * eta2 + asin(dg) * eta1 * sin(2 * pi * eta2)));
    f_out[1] = r0 * ((1 - pow(eta1, 2)) * dy + eg * kg * eta1 * sin(2 * pi * eta2));
    f_out[2] = eta3 * lz;
}

/**
 * Evaluate the Jacobian of the D-shaped Shafranov mapping, as in
 * shafranov_dshaped_cylinder_kernels.shafranov_dshaped_df.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis (unused, as in pyccel).
 * @param r0 Base radius.
 * @param lz Length in z-direction.
 * @param dx Shafranov shift in x-direction.
 * @param dy Shafranov shift in y-direction.
 * @param dg Triangularity delta = sin(alpha).
 * @param eg Inverse aspect ratio epsilon.
 * @param kg Ellipticity kappa.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void shafranov_dshaped_df(double eta1, double eta2, double eta3, double r0, double lz, double dx,
                                            double dy, double dg, double eg, double kg, double* df_out) {
    df_out[0] =
        r0 * (-2 * dx * eta1 -
              eg * eta1 * sin(2 * pi * eta2) * asin(dg) * sin(eta1 * sin(2 * pi * eta2) * asin(dg) + 2 * pi * eta2) +
              eg * cos(eta1 * sin(2 * pi * eta2) * asin(dg) + 2 * pi * eta2));
    df_out[1] = -r0 * eg * eta1 * (2 * pi * eta1 * cos(2 * pi * eta2) * asin(dg) + 2 * pi) *
                sin(eta1 * sin(2 * pi * eta2) * asin(dg) + 2 * pi * eta2);
    df_out[2] = 0.0;
    df_out[3] = r0 * (-2 * dy * eta1 + eg * kg * sin(2 * pi * eta2));
    df_out[4] = 2 * pi * r0 * eg * eta1 * kg * cos(2 * pi * eta2);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}
}  // namespace struphy_cuda::shafranov_dshaped_cylinder_kernels
