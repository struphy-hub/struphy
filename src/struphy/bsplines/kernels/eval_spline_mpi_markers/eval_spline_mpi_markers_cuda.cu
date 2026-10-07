#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/kernel_arguments/spline_args.cuh"

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
 * @param args_spline Kind of 1d basis (0 = N-spline, 1 = D-spline), spline degrees (1 to 8) and knot vectors of V0
 *        in each direction, and start indices of the splines on the current process.
 * @param values Output values S_p = S(*markers[p, :]), one per row; skipped rows are left unchanged.
 */
extern "C" __global__ void eval_spline_mpi_markers(Array2D<double> markers, Array3D<double> _data,
                                                   SplineArgs args_spline, Array1D<double> values) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long Np = markers.shape[0];
    if (ip >= Np) return;

    // point not in process domain
    if (markers(ip, 0) == -1.) return;

    values(ip) = eval_spline_mpi(markers(ip, 0), markers(ip, 1), markers(ip, 2), _data, args_spline.kind,
                                 args_spline.pn, args_spline.tn1, args_spline.tn2, args_spline.tn3, args_spline.starts);
}
