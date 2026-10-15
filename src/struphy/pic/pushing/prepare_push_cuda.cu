#include "cunumpy/array_view.cuh"

/**
 * Prepare the marker buffer for a push, as the three row-slice assignments at the start of Pusher._push:
 *
 *     markers[first_pusher_idx:first_shift_idx] = markers[:n_init]
 *     markers[first_shift_idx:residual_idx] = 0.0
 *     markers[residual_idx:-2] = 0.0
 *
 * One thread per entry of the row range [first_pusher_idx, n_cols - 2) of every marker, numbered row by row. In the
 * marker buffer (n_cols x n_rows, one column per marker, struphy.pic.marker_layout) neighbouring threads then read
 * and write neighbouring markers of the same row, and the whole range is written in a single launch (instead of one
 * per slice).
 *
 * @param markers Marker buffer (n_cols x n_rows), every marker including holes.
 * @param first_pusher_idx First row of the saved initial phase space coordinates.
 * @param n_init Number of saved coordinates, 3 + vdim (first_shift_idx = first_pusher_idx + n_init).
 */
extern "C" __global__ void prepare_push(Array2D<double> markers, int first_pusher_idx, int n_init) {
    long long n_rows = markers.shape[1];
    long long width = markers.shape[0] - 2 - first_pusher_idx;
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_rows * width) return;

    long long k = t / n_rows;
    long long ip = t - k * n_rows;
    markers(first_pusher_idx + k, ip) = k < n_init ? markers(k, ip) : 0.0;
}
