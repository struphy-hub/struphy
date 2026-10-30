#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/kernel_arguments/spline_args.cuh"

using namespace struphy_cuda;

// Same arguments, in the same order, as the pyccel kernel in eval_spline_mpi_markers_kernels.py. Array views carry
// dimensions and strides, so no CUDA-only lengths are needed.

/**
 * Flat (marker) evaluation of a distributed spline, as in evaluation_kernels_3d.eval_spline_mpi_markers.
 *
 * One thread per marker.
 *
 * @param markers Marker coordinates (3 or more x Np, one column per marker, any strides); columns flagged with -1
 *        in the first row are not on the process domain and are skipped.
 * @param _data Spline coefficients of the current process, any strides.
 * @param args_spline Kind of 1d basis (0 = N-spline, 1 = D-spline), spline degrees (1 to 8) and knot vectors of V0
 *        in each direction, and start indices of the splines on the current process.
 * @param values Output values S_p = S(*markers[:, p]), one per marker; skipped markers are left unchanged.
 */
extern "C" __global__ void eval_spline_mpi_markers(Array2D<double> markers, Array3D<double> _data,
                                                   SplineArgs args_spline, Array1D<double> values) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long Np = markers.shape[1];
    if (ip >= Np) return;

    // point not in process domain
    if (markers(0, ip) == -1.) return;

    values(ip) = evaluation_kernels_3d::eval_spline_mpi(markers(0, ip), markers(1, ip), markers(2, ip), _data,
                                                        args_spline);
}
