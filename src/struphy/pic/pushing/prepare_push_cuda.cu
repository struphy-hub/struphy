#include "cunumpy/array_view.cuh"

/**
 * Prepare the marker buffer for a push, as the three column-slice assignments at the start of Pusher._push:
 *
 *     markers[:, first_pusher_idx:first_shift_idx] = markers[:, :n_init]
 *     markers[:, first_shift_idx:residual_idx] = 0.0
 *     markers[:, residual_idx:-2] = 0.0
 *
 * One thread per entry of the column range [first_pusher_idx, n_cols - 2) of every row, numbered column by
 * column. In the column-major marker buffer (struphy.pic.marker_layout) neighbouring threads then read and write
 * neighbouring rows of the same column, and the whole range is written in a single launch (instead of one per
 * slice). The buffer is an Array2D view: a raw pointer argument would be made C-contiguous (copied) by the launch.
 *
 * @param markers Marker buffer (n_rows x n_cols, column-major), every row including holes.
 * @param first_pusher_idx First column of the saved initial phase space coordinates.
 * @param n_init Number of saved coordinates, 3 + vdim (first_shift_idx = first_pusher_idx + n_init).
 */
extern "C" __global__ void prepare_push(Array2D<double> markers, int first_pusher_idx, int n_init) {
    long long n_rows = markers.shape[0];
    long long width = markers.shape[1] - 2 - first_pusher_idx;
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_rows * width) return;

    long long k = t / n_rows;
    long long ip = t - k * n_rows;
    markers(ip, first_pusher_idx + k) = k < n_init ? markers(ip, k) : 0.0;
}
