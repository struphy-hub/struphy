#pragma once
#include "cunumpy/array_view.cuh"
namespace struphy_cuda::evaluation_kernels_2d {
/**
 * Sum the non-zero contributions of a 2d spline, as in evaluation_kernels_2d.evaluation_kernel_2d.
 *
 * @param p1 Degree of the univariate splines along the first axis.
 * @param p2 Degree of the univariate splines along the second axis.
 * @param basis1 The p1 + 1 non-zero basis values along the first axis.
 * @param basis2 The p2 + 1 non-zero basis values along the second axis.
 * @param ind1 Global indices of the p1 + 1 non-vanishing splines along the first axis.
 * @param ind2 Global indices of the p2 + 1 non-vanishing splines along the second axis.
 * @param coeff The spline coefficients c_ij (any strides).
 * @return spline_value, the value of the tensor-product spline.
 */
__device__ inline double evaluation_kernel_2d(int p1, int p2, const double* basis1, const double* basis2,
                                              Array1D<long long> ind1, Array1D<long long> ind2,
                                              Array2D<double> coeff) {
    double spline_value = 0.;
    for (int il1 = 0; il1 <= p1; ++il1) {
        long long i1 = ind1(il1);
        for (int il2 = 0; il2 <= p2; ++il2) {
            long long i2 = ind2(il2);
            spline_value += coeff(i1, i2) * basis1[il1] * basis2[il2];
        }
    }
    return spline_value;
}
}  // namespace struphy_cuda::evaluation_kernels_2d
