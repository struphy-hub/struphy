#pragma once
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda {
/**
 * Evaluate the hollow torus mapping, as in hollow_torus_kernels.hollow_torus.
 *
 * F_x = (r cos(theta) + r0) cos(2 pi eta3 / tor_period), F_y = -(r cos(theta) + r0) sin(2 pi eta3 / tor_period),
 * F_z = r sin(theta), with r = a1 + (a2 - a1) eta1 and theta = 2 pi eta2 / pol_period (equal angle) or the
 * straight field line angle theta = 2 arctan(sqrt((1 + r/r0) / (1 - r/r0)) tan(pi eta2)).
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param a1 Inner radius.
 * @param a2 Outer radius.
 * @param r0 Major radius.
 * @param sfl Straight field line angle if 1.0, equal angle otherwise.
 * @param pol_period Poloidal periodicity (equal angle only).
 * @param tor_period Toroidal periodicity.
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void hollow_torus(double eta1, double eta2, double eta3, double a1, double a2, double r0,
                                    double sfl, double pol_period, double tor_period, double* f_out) {
    // straight field lines coordinates
    if (sfl == 1.0) {
        double da = a2 - a1;
        double r = a1 + eta1 * da;
        double theta = 2 * atan(sqrt((1 + r / r0) / (1 - r / r0)) * tan(pi * eta2));
        f_out[0] = (r * cos(theta) + r0) * cos(2 * pi * eta3 / tor_period);
        f_out[1] = (r * cos(theta) + r0) * (-1) * sin(2 * pi * eta3 / tor_period);
        f_out[2] = r * sin(theta);
    }
    // equal angle coordinates
    else {
        double da = a2 - a1;
        f_out[0] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * cos(2 * pi * eta3 / tor_period);
        f_out[1] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * (-1) * sin(2 * pi * eta3 / tor_period);
        f_out[2] = (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period);
    }
}

/**
 * Evaluate the Jacobian of the hollow torus mapping, as in hollow_torus_kernels.hollow_torus_df.
 *
 * The local names (da, r, eps, eps_p, tpe, tpe_p, g, g_p, theta, dtheta_deta1, dtheta_deta2) are those of pyccel.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param a1 Inner radius.
 * @param a2 Outer radius.
 * @param r0 Major radius.
 * @param sfl Straight field line angle if 1.0, equal angle otherwise.
 * @param pol_period Poloidal periodicity (equal angle only).
 * @param tor_period Toroidal periodicity.
 * @param df_out Output 3x3 matrix stored as nine row-major entries.
 */
__device__ inline void hollow_torus_df(double eta1, double eta2, double eta3, double a1, double a2, double r0,
                                       double sfl, double pol_period, double tor_period, double* df_out) {
    // straight field lines coordinates
    if (sfl == 1.0) {
        double da = a2 - a1;
        double r = a1 + da * eta1;
        double eps = r / r0;
        double eps_p = da / r0;
        double tpe = tan(pi * eta2);
        double tpe_p = pi / pow(cos(pi * eta2), 2);
        double g = sqrt((1 + eps) / (1 - eps));
        double g_p = 1 / (2 * g) * (eps_p * (1 - eps) + (1 + eps) * eps_p) / pow(1 - eps, 2);
        double theta = 2 * atan(g * tpe);
        double dtheta_deta1 = 2 / (1 + pow(g * tpe, 2)) * g_p * tpe;
        double dtheta_deta2 = 2 / (1 + pow(g * tpe, 2)) * g * tpe_p;
        df_out[0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * cos(2 * pi * eta3 / tor_period);
        df_out[1] = -r * sin(theta) * dtheta_deta2 * cos(2 * pi * eta3 / tor_period);
        df_out[2] = -2 * pi / tor_period * (r * cos(theta) + r0) * sin(2 * pi * eta3 / tor_period);
        df_out[3] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * (-1) * sin(2 * pi * eta3 / tor_period);
        df_out[4] = -r * sin(theta) * dtheta_deta2 * (-1) * sin(2 * pi * eta3 / tor_period);
        df_out[5] = 2 * pi / tor_period * (r * cos(theta) + r0) * (-1) * cos(2 * pi * eta3 / tor_period);
        df_out[6] = da * sin(theta) + r * cos(theta) * dtheta_deta1;
        df_out[7] = r * cos(theta) * dtheta_deta2;
        df_out[8] = 0.0;
    }
    // equal angle coordinates
    else {
        double da = a2 - a1;
        df_out[0] = da * cos(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period);
        df_out[1] =
            -2 * pi / pol_period * (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period);
        df_out[2] = -2 * pi / tor_period * ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) *
                    sin(2 * pi * eta3 / tor_period);
        df_out[3] = da * cos(2 * pi * eta2 / pol_period) * (-1) * sin(2 * pi * eta3 / tor_period);
        df_out[4] = -2 * pi / pol_period * (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period) * (-1) *
                    sin(2 * pi * eta3 / tor_period);
        df_out[5] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * (-1) *
                    cos(2 * pi * eta3 / tor_period) * 2 * pi / tor_period;
        df_out[6] = da * sin(2 * pi * eta2 / pol_period);
        df_out[7] = (a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) * 2 * pi / pol_period;
        df_out[8] = 0.0;
    }
}
}
