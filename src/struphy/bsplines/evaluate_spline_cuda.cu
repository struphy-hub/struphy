#include "struphy/bsplines/evaluation_kernels_3d.cuh"

// The kernels below take the same arguments, in the same order, as their pyccel versions in
// evaluation_kernels_3d.py. Array views carry dimensions and strides, so no CUDA-only lengths are needed.

/**
 * Point-wise evaluation of a distributed tensor-product spline, as in evaluation_kernels_3d.eval_spline_mpi.
 *
 * @param eta1 Evaluation point along the first axis.
 * @param eta2 Evaluation point along the second axis.
 * @param eta3 Evaluation point along the third axis.
 * @param _data Spline coefficients of the current process (the _data of a StencilVector), any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param tn1 Knot vector of V0 along the first axis, any stride.
 * @param tn2 Knot vector of V0 along the second axis.
 * @param tn3 Knot vector of V0 along the third axis.
 * @param starts Start indices of the splines on the current process (three entries).
 * @return value, the value of the spline at (eta1, eta2, eta3).
 *
 * Pyccel allocates bn1, ..., bd3 per call; here they are fields of the thread-local SplineScratch.
 */
__device__ inline double eval_spline_mpi(double eta1, double eta2, double eta3, Array3D<double> _data,
                                         const long long* kind, const long long* pn, Array1D<double> tn1,
                                         Array1D<double> tn2, Array1D<double> tn3, const long long* starts) {
    struphy_cuda::SplineScratch scratch;

    // get spline values at eta
    scratch.span1 = struphy_cuda::find_span(tn1, pn[0], eta1);
    scratch.span2 = struphy_cuda::find_span(tn2, pn[1], eta2);
    scratch.span3 = struphy_cuda::find_span(tn3, pn[2], eta3);
    struphy_cuda::b_d_splines_slim(tn1, pn[0], eta1, scratch.span1, scratch.bn1, scratch.bd1);
    struphy_cuda::b_d_splines_slim(tn2, pn[1], eta2, scratch.span2, scratch.bn2, scratch.bd2);
    struphy_cuda::b_d_splines_slim(tn3, pn[2], eta3, scratch.span3, scratch.bn3, scratch.bd3);

    const double* b1 = kind[0] == 0 ? scratch.bn1 : scratch.bd1;
    const double* b2 = kind[1] == 0 ? scratch.bn2 : scratch.bd2;
    const double* b3 = kind[2] == 0 ? scratch.bn3 : scratch.bd3;

    double value = struphy_cuda::eval_spline_mpi_kernel(pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3,
                                                        scratch.span1, scratch.span2, scratch.span3, _data, starts);
    return value;
}

/**
 * Flat (marker) evaluation of a distributed spline, as in evaluation_kernels_3d.eval_spline_mpi_markers.
 *
 * One thread per marker row.
 *
 * @param markers Marker coordinates (Np x 3 or wider, any strides); rows flagged with -1 in the first
 *        column are not on the process domain and are skipped.
 * @param _data Spline coefficients of the current process, any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param tn1 Knot vector of V0 along the first axis.
 * @param tn2 Knot vector of V0 along the second axis.
 * @param tn3 Knot vector of V0 along the third axis.
 * @param starts Start indices of the splines on the current process (three entries).
 * @param values Output values S_p = S(*markers[p, :]), one per row; skipped rows are left unchanged.
 */
extern "C" __global__ void eval_spline_mpi_markers(Array2D<double> markers, Array3D<double> _data,
                                                   const long long* kind, long long* pn, Array1D<double> tn1,
                                                   Array1D<double> tn2, Array1D<double> tn3, long long* starts,
                                                   Array1D<double> values) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long Np = markers.shape[0];
    if (ip >= Np) return;

    // point not in process domain
    if (markers(ip, 0) == -1.) return;

    values(ip) = eval_spline_mpi(markers(ip, 0), markers(ip, 1), markers(ip, 2), _data, kind, pn, tn1, tn2, tn3,
                                 starts);
}

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

/**
 * Sparse meshgrid evaluation of a distributed spline, as in evaluation_kernels_3d.eval_spline_mpi_sparse_meshgrid.
 *
 * One thread per entry (i, j, k) of values; the point is (eta1[i, 0, 0], eta2[0, j, 0], eta3[0, 0, k]).
 *
 * @param eta1 First coordinates, shape (n1, 1, 1), any strides; -1 flags a point not on the process domain.
 * @param eta2 Second coordinates, shape (1, n2, 1).
 * @param eta3 Third coordinates, shape (1, 1, n3).
 * @param _data Spline coefficients of the current process, any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param tn1 Knot vector of V0 along the first axis.
 * @param tn2 Knot vector of V0 along the second axis.
 * @param tn3 Knot vector of V0 along the third axis.
 * @param starts Start indices of the splines on the current process (three entries).
 * @param values Output values, shape (n1, n2, n3); entries of flagged points are left unchanged.
 */
extern "C" __global__ void eval_spline_mpi_sparse_meshgrid(Array3D<double> eta1, Array3D<double> eta2,
                                                           Array3D<double> eta3, Array3D<double> _data,
                                                           const long long* kind, long long* pn, Array1D<double> tn1,
                                                           Array1D<double> tn2, Array1D<double> tn3,
                                                           long long* starts, Array3D<double> values) {
    // CUDA-only: flat index of this thread, split into the pyccel loop variables i, j, k
    long long ijk = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ijk >= values.shape[0] * values.shape[1] * values.shape[2]) return;
    long long k = ijk % values.shape[2];
    long long j = (ijk / values.shape[2]) % values.shape[1];
    long long i = ijk / (values.shape[1] * values.shape[2]);

    // point not in process domain
    if (eta1(i, 0, 0) == -1. || eta2(0, j, 0) == -1. || eta3(0, 0, k) == -1.) return;

    values(i, j, k) = eval_spline_mpi(eta1(i, 0, 0), eta2(0, j, 0), eta3(0, 0, k), _data, kind, pn, tn1, tn2, tn3,
                                      starts);
}
