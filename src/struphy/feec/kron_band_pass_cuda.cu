/**
 * One pass of a Kronecker product of 1D banded matrices (struphy.feec.kronecker_mass._band_pass):
 *
 *     out[..., i, ...] = sum_d A[i, d] * inp[..., i + offset + d, ...]    along `axis`,
 *
 * with the other two axes unchanged (out and inp have the same extent along them). One thread per entry of `out`,
 * the last axis fastest, so that neighbouring threads read neighbouring entries of `inp` along every axis. `out` may
 * be a strided view (the owned rows of a stencil vector's data).
 *
 * @param A Banded 1D matrix, shape (out.shape[axis], n_diagonals).
 * @param inp Input array (3D).
 * @param out Output array (3D, any strides); written, not added to.
 * @param axis Axis of the pass (0, 1 or 2).
 * @param offset Index in `inp` of diagonal 0 of row 0 along `axis`.
 */
#include "cunumpy/array_view.cuh"

extern "C" __global__ void kron_band_pass(Array2D<double> A, Array3D<double> inp, Array3D<double> out, int axis,
                                          long long offset) {
    const long long n1 = out.shape[1], n2 = out.shape[2];
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= out.shape[0] * n1 * n2) return;
    const long long i2 = t % n2;
    const long long i1 = (t / n2) % n1;
    const long long i0 = t / (n1 * n2);
    const long long i = axis == 0 ? i0 : (axis == 1 ? i1 : i2);

    double v = 0.0;
    for (long long d = 0; d < A.shape[1]; ++d) {
        const long long j = i + offset + d;
        const double x = axis == 0 ? inp(j, i1, i2) : (axis == 1 ? inp(i0, j, i2) : inp(i0, i1, j));
        v += A(i, d) * x;
    }
    out(i0, i1, i2) = v;
}
