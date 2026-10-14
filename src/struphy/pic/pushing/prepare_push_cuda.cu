/**
 * Prepare the marker buffer for a push, as the three column-slice assignments at the start of Pusher._push:
 *
 *     markers[:, first_pusher_idx:first_shift_idx] = markers[:, :n_init]
 *     markers[:, first_shift_idx:residual_idx] = 0.0
 *     markers[:, residual_idx:-2] = 0.0
 *
 * A 2D launch, n_threads = (n_rows, n_cols - 2 - first_pusher_idx): x is the row, y the column within the range
 * [first_pusher_idx, n_cols - 2), so the whole range is written in a single pass. The buffer is addressed through
 * a strided view: on the CuPy backend it is column-major (Particles allocates it in Fortran order), so neighbouring
 * threads (rows) of one column write neighbouring addresses.
 *
 * @param markers Marker buffer (n_rows x n_cols, any strides), every row including holes.
 * @param first_pusher_idx First column of the saved initial phase space coordinates.
 * @param n_init Number of saved coordinates, 3 + vdim (first_shift_idx = first_pusher_idx + n_init).
 */
#include "cunumpy/array_view.cuh"

extern "C" __global__ void prepare_push(Array2D<double> markers, int first_pusher_idx, int n_init) {
    const long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long k = (long long)blockDim.y * blockIdx.y + threadIdx.y;
    if (ip >= markers.shape[0] || k >= markers.shape[1] - 2 - first_pusher_idx) return;
    markers(ip, first_pusher_idx + k) = k < n_init ? markers(ip, k) : 0.0;
}
