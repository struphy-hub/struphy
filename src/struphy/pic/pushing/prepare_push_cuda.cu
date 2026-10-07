/**
 * Prepare the marker buffer for a push, as the three column-slice assignments at the start of Pusher._push:
 *
 *     markers[:, first_init_idx:first_shift_idx] = markers[:, :n_init]
 *     markers[:, first_shift_idx:residual_idx] = 0.0
 *     markers[:, residual_idx:-2] = 0.0
 *
 * One thread per entry of the contiguous column range [first_init_idx, n_cols - 2) of every row, so that
 * neighbouring threads write neighbouring addresses of the row-major buffer and the whole range is written
 * in a single pass (instead of one strided pass over the buffer per slice).
 *
 * @param markers Marker buffer (n_rows x n_cols, row-major), every row including holes.
 * @param n_rows Number of rows of the buffer.
 * @param n_cols Number of columns of the buffer.
 * @param first_init_idx First column of the saved initial phase space coordinates.
 * @param n_init Number of saved coordinates, 3 + vdim (first_shift_idx = first_init_idx + n_init).
 */
extern "C" __global__ void prepare_push(double* markers, long long n_rows, int n_cols, int first_init_idx,
                                        int n_init) {
    long long width = n_cols - 2 - first_init_idx;
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_rows * width) return;

    long long ip = t / width;
    int col = first_init_idx + (int)(t - ip * width);
    double* row = markers + ip * n_cols;
    row[col] = col < first_init_idx + n_init ? row[col - first_init_idx] : 0.0;
}
