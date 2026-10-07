#include "struphy/geometry/transform_kernels.cuh"

// Same arguments, in the same order, as the pyccel kernel in kernel_pullpush_kernels.py.

/**
 * Pull-backs, push-forwards and transformations on a 3d grid of points, as in kernel_pullpush_kernels.kernel_pullpush.
 *
 * One thread per grid point (i1, i2, i3), n1 * n2 * n3 threads (launched with n_threads). The point is
 * (eta1[i1, i2 * s, i3 * s], eta2[i1 * s, i2, i3 * s], eta3[i1 * s, i2 * s, i3]) with s = 0 for a sparse meshgrid
 * and s = 1 otherwise.
 *
 * @param a Values to transform, shape (n1, n2, n3, n_comp), n_comp 1 (scalar kinds) or 3 (vector kinds), any strides.
 * @param eta1 First coordinates, shape (n1, n2, n3) or (n1, 1, 1) for a sparse meshgrid, any strides.
 * @param eta2 Second coordinates, shape (n1, n2, n3) or (1, n2, 1).
 * @param eta3 Third coordinates, shape (n1, n2, n3) or (1, 1, n3).
 * @param kind_transform Which general transformation: 0 pull, 1 push, otherwise tran.
 * @param kind_fun Which detailed transformation (see pull, push, tran in transform_kernels.cuh).
 * @param args_domain Mapping arguments; every analytic mapping (spline mappings trap until CUDA strategy PR 19).
 * @param is_sparse_meshgrid Whether the points come from a sparse meshgrid.
 * @param out Output values, shape (n1, n2, n3, 3), any strides; entries a transformation does not write keep their
 *        value.
 */
extern "C" __global__ void kernel_pullpush(Array4D<double> a, Array3D<double> eta1, Array3D<double> eta2,
                                           Array3D<double> eta3, int kind_transform, int kind_fun,
                                           DomainArgs args_domain, bool is_sparse_meshgrid, Array4D<double> out) {
    long long n1 = eta1.shape[0];
    long long n2 = eta2.shape[1];
    long long n3 = eta3.shape[2];

    // CUDA-only: flat index of this thread, split into the pyccel loop variables i1, i2, i3
    long long i123 = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i123 >= n1 * n2 * n3) return;
    long long i3 = i123 % n3;
    long long i2 = (i123 / n3) % n2;
    long long i1 = i123 / (n2 * n3);

    // CUDA-only: per-thread copies of the pyccel stack arrays tmp1 (length a.shape[3]) and tmp2 (out.shape[3]),
    // at most three entries each
    double tmp1[3], tmp2[3];
    long long n_tmp1 = a.shape[3] < 3 ? a.shape[3] : 3;
    long long n_tmp2 = out.shape[3] < 3 ? out.shape[3] : 3;

    int sparse_factor = is_sparse_meshgrid ? 0 : 1;

    double e1 = eta1(i1, i2 * sparse_factor, i3 * sparse_factor);
    double e2 = eta2(i1 * sparse_factor, i2, i3 * sparse_factor);
    double e3 = eta3(i1 * sparse_factor, i2 * sparse_factor, i3);

    for (long long k = 0; k < n_tmp1; ++k) tmp1[k] = a(i1, i2, i3, k);
    for (long long k = 0; k < n_tmp2; ++k) tmp2[k] = out(i1, i2, i3, k);

    if (kind_transform == 0)
        struphy_cuda::pull(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);
    else if (kind_transform == 1)
        struphy_cuda::push(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);
    else
        struphy_cuda::tran(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);

    for (long long k = 0; k < n_tmp2; ++k) out(i1, i2, i3, k) = tmp2[k];
}
