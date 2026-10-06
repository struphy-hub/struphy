#include "struphy/bsplines/evaluation_kernels_3d.cuh"

// Same arguments, in the same order, as the pyccel kernel in eval_spline_mpi_matrix_kernels.py. Array views carry
// dimensions and strides, so no CUDA-only lengths are needed.

/**
 * 3d array evaluation of a distributed spline, as in evaluation_kernels_3d.eval_spline_mpi_matrix.
 *
 * One thread per entry (i, j, k) of values; the point is (eta1[i, j, k], eta2[i, j, k], eta3[i, j, k]).
 *
 * @param eta1 First coordinates of the points (any strides); -1 flags a point not on the process domain.
 * @param eta2 Second coordinates of the points, same shape as eta1.
 * @param eta3 Third coordinates of the points, same shape as eta1.
 * @param _data Spline coefficients of the current process, any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param tn1 Knot vector of V0 along the first axis.
 * @param tn2 Knot vector of V0 along the second axis.
 * @param tn3 Knot vector of V0 along the third axis.
 * @param starts Start indices of the splines on the current process (three entries).
 * @param values Output values, same shape as eta1; entries of flagged points are left unchanged.
 */
extern "C" __global__ void eval_spline_mpi_matrix(Array3D<double> eta1, Array3D<double> eta2, Array3D<double> eta3,
                                                  Array3D<double> _data, const long long* kind, long long* pn,
                                                  Array1D<double> tn1, Array1D<double> tn2, Array1D<double> tn3,
                                                  long long* starts, Array3D<double> values) {
    // CUDA-only: flat index of this thread, split into the pyccel loop variables i, j, k
    long long ijk = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ijk >= values.shape[0] * values.shape[1] * values.shape[2]) return;
    long long k = ijk % values.shape[2];
    long long j = (ijk / values.shape[2]) % values.shape[1];
    long long i = ijk / (values.shape[1] * values.shape[2]);

    // point not in process domain
    if (eta1(i, j, k) == -1. || eta2(i, j, k) == -1. || eta3(i, j, k) == -1.) return;

    values(i, j, k) = eval_spline_mpi(eta1(i, j, k), eta2(i, j, k), eta3(i, j, k), _data, kind, pn, tn1, tn2, tn3,
                                      starts);
}
