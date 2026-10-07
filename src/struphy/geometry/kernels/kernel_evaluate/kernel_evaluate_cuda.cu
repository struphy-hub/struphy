#include "struphy/geometry/evaluation_kernels.cuh"

// Same arguments, in the same order, as the pyccel kernel in kernel_evaluate_kernels.py. mat_f has five
// dimensions, more than cunumpy's array views (Array1D-Array4D), so it is passed as a C-contiguous pointer.

/**
 * Evaluation of metric coefficients on a 3d grid of points, as in kernel_evaluate_kernels.kernel_evaluate.
 *
 * One thread per grid point (i1, i2, i3), n1 * n2 * n3 threads (launched with n_threads). The point is
 * (eta1[i1, i2 * s, i3 * s], eta2[i1 * s, i2, i3 * s], eta3[i1 * s, i2 * s, i3]) with s = 0 for a sparse meshgrid
 * and s = 1 otherwise.
 *
 * @param eta1 First coordinates, shape (n1, n2, n3) or (n1, 1, 1) for a sparse meshgrid, any strides.
 * @param eta2 Second coordinates, shape (n1, n2, n3) or (1, n2, 1).
 * @param eta3 Third coordinates, shape (n1, n2, n3) or (1, 1, n3).
 * @param kind_coeff Which coefficient: -1 identity, 0 mapping F, 1 DF, 2 det(DF), 3 DF^(-1), 4 G, 5 G^(-1).
 * @param args Mapping arguments; every spline and analytic mapping.
 * @param mat_f Output, C-contiguous of shape (n1, n2, n3, 3, 3); entries a coefficient does not write keep their
 *        value.
 * @param is_sparse_meshgrid Whether the points come from a sparse meshgrid.
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 */
extern "C" __global__ void kernel_evaluate(Array3D<double> eta1, Array3D<double> eta2, Array3D<double> eta3,
                                           int kind_coeff, DomainArgs args, double* mat_f, bool is_sparse_meshgrid,
                                           bool avoid_round_off) {
    long long n1 = eta1.shape[0];
    long long n2 = eta2.shape[1];
    long long n3 = eta3.shape[2];

    // CUDA-only: flat index of this thread, split into the pyccel loop variables i1, i2, i3
    long long i123 = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i123 >= n1 * n2 * n3) return;
    long long i3 = i123 % n3;
    long long i2 = (i123 / n3) % n2;
    long long i1 = i123 / (n2 * n3);

    double tmp0[3], tmp1[9], tmp2[9], tmp3[9], out[9];

    int sparse_factor = is_sparse_meshgrid ? 0 : 1;

    double e1 = eta1(i1, i2 * sparse_factor, i3 * sparse_factor);
    double e2 = eta2(i1 * sparse_factor, i2, i3 * sparse_factor);
    double e3 = eta3(i1 * sparse_factor, i2 * sparse_factor, i3);

    // CUDA-only: the 3x3 block mat_f[i1, i2, i3, :, :] of the C-contiguous output
    double* mat_f_point = mat_f + 9 * i123;

    for (int k = 0; k < 9; ++k) out[k] = mat_f_point[k];

    struphy_cuda::select_metric_coeff(e1, e2, e3, kind_coeff, args, tmp0, tmp1, tmp2, tmp3, avoid_round_off, out);

    for (int k = 0; k < 9; ++k) mat_f_point[k] = out[k];
}
