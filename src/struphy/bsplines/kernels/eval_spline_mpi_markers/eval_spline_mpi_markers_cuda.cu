#include "struphy/bsplines/evaluation_kernels_3d.cuh"

using namespace struphy_cuda;

// Same arguments, in the same order, as the pyccel kernel in eval_spline_mpi_markers_kernels.py. Array views carry
// dimensions and strides, so no CUDA-only lengths are needed.

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

    values(ip) = evaluation_kernels_3d::eval_spline_mpi(markers(ip, 0), markers(ip, 1), markers(ip, 2), _data, kind, pn,
                                                        tn1, tn2, tn3, starts);
}
