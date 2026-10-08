#include "struphy/bsplines/evaluation_kernels_3d.cuh"

using namespace struphy_cuda;

// Same arguments, in the same order, as the pyccel kernel in eval_spline_mpi_tensor_product_fixed_kernels.py. Array
// views carry dimensions and strides, so no CUDA-only lengths are needed.

/**
 * Tensor-product evaluation of a distributed spline on a fixed grid, as in
 * eval_spline_mpi_tensor_product_fixed_kernels.eval_spline_mpi_tensor_product_fixed.
 *
 * One thread per entry (i, j, k) of values. The knot spans and the non-zero basis values at the grid points are
 * pre-evaluated (SplineFunction.eval_tp_fixed_loc, Derham.prepare_eval_tp_fixed), so the thread only sums the
 * contributions of the local coefficients.
 *
 * @param span1s Knot span indices of the grid points along the first axis (ni entries).
 * @param span2s Knot span indices along the second axis (nj entries).
 * @param span3s Knot span indices along the third axis (nk entries).
 * @param b1s Values of the non-zero basis functions along the first axis, shape (ni, pn[0] - kind[0] + 1).
 * @param b2s Values of the non-zero basis functions along the second axis, shape (nj, pn[1] - kind[1] + 1).
 * @param b3s Values of the non-zero basis functions along the third axis, shape (nk, pn[2] - kind[2] + 1).
 * @param _data Spline coefficients of the current process, any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param starts Start indices of the splines on the current process (three entries).
 * @param values Output spline values, shape (ni, nj, nk).
 *
 * Pyccel copies the rows b1s[i, :], b2s[j, :], b3s[k, :] into b1, b2, b3 before calling eval_spline_mpi_kernel; here
 * b1, b2, b3 are thread-local arrays of MAX_SPLINE_DEGREE + 1 entries, since the rows of a view may be strided.
 */
extern "C" __global__ void eval_spline_mpi_tensor_product_fixed(Array1D<long long> span1s, Array1D<long long> span2s,
                                                                Array1D<long long> span3s, Array2D<double> b1s,
                                                                Array2D<double> b2s, Array2D<double> b3s,
                                                                Array3D<double> _data, const long long* kind,
                                                                const long long* pn, const long long* starts,
                                                                Array3D<double> values) {
    // CUDA-only: flat index of this thread, split into the pyccel loop variables i, j, k
    long long ijk = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long ni = span1s.shape[0];
    long long nj = span2s.shape[0];
    long long nk = span3s.shape[0];
    if (ijk >= ni * nj * nk) return;
    long long k = ijk % nk;
    long long j = (ijk / nk) % nj;
    long long i = ijk / (nj * nk);

    double b1[bsplines_kernels::MAX_SPLINE_DEGREE + 1], b2[bsplines_kernels::MAX_SPLINE_DEGREE + 1],
        b3[bsplines_kernels::MAX_SPLINE_DEGREE + 1];
    for (int il = 0; il <= pn[0] - kind[0]; ++il) b1[il] = b1s(i, il);
    for (int il = 0; il <= pn[1] - kind[1]; ++il) b2[il] = b2s(j, il);
    for (int il = 0; il <= pn[2] - kind[2]; ++il) b3[il] = b3s(k, il);

    values(i, j, k) =
        evaluation_kernels_3d::eval_spline_mpi_kernel(pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3,
                                                      span1s(i), span2s(j), span3s(k), _data, starts);
}
