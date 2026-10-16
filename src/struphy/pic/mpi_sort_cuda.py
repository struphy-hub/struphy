"""CUDA kernel of :meth:`~struphy.pic.base.Particles.mpi_sort_markers` on the CuPy backend: the destination rank of
every marker row in one pass.

The NumPy-style path finds the markers to send and their destinations with about a dozen full passes over the marker
buffer (sorting positions, wrapping, a domain test per axis and per rank, masks and index lists). With millions of
rows per GPU and a few percent of them leaving per step, those passes cost far more than the transfer itself.
:func:`sort_destinations` computes, per row, ``dest = -1`` (the marker stays: on this rank's domain, a hole or a ghost),
the rank to send it to, or ``-2`` (it leaves but lies on no rank's domain, e.g. a NaN position: it is removed, as in
the NumPy-style path), reading only the position, shift and (for ``alpha != 1``) initial-position columns.

The arithmetic is that of the NumPy-style path, operation by operation and without fused multiply-adds, so the
destinations are the same bit for bit.
"""

from functools import cache

import numpy as np

_SRC = r"""
// numpy's floating-point remainder (npy_divmod) for a positive divisor b
__device__ __forceinline__ double py_mod(double a, double b)
{
    double m = fmod(a, b);
    if (m != 0.0) {
        if ((b < 0.0) != (m < 0.0)) m = __dadd_rn(m, b);
    } else {
        m = copysign(0.0, b);
    }
    return m;
}

// destination of row i: -1 (stays), the rank of its domain, or -2 (on no rank's domain)
__device__ __forceinline__ int row_destination(
    const double* __restrict__ markers, long long row_stride, long long col_stride, long long i,
    const bool* __restrict__ holes, const bool* __restrict__ ghosts,
    int shift_col, int init_col, const double* __restrict__ alpha, int use_alpha,
    const int* __restrict__ periodic, const double* __restrict__ bounds,
    const int* __restrict__ ranks, int n_ranks, int own_rank)
{
    if (holes[i] || ghosts[i]) return -1;

    double eta[3];
    for (int a = 0; a < 3; ++a) {
        double pos = markers[i * row_stride + a * col_stride];
        double shift = markers[i * row_stride + (shift_col + a) * col_stride];
        double e = __dadd_rn(pos, shift);
        if (use_alpha) {
            double init = markers[i * row_stride + (init_col + a) * col_stride];
            e = __dadd_rn(__dmul_rn(alpha[a], e), __dmul_rn(__dadd_rn(1.0, -alpha[a]), init));
        }
        // eta = 1 is the right end of a non-periodic axis and is not wrapped
        if (!(periodic[a] == 0 && e == 1.0)) e = py_mod(e, 1.0);
        eta[a] = e;
    }

    const double* b = bounds + 6 * own_rank;
    if (eta[0] >= b[0] && eta[0] < b[1] && eta[1] >= b[2] && eta[1] < b[3] && eta[2] >= b[4] && eta[2] < b[5]) {
        return -1;
    }
    for (int k = 0; k < n_ranks; ++k) {
        b = bounds + 6 * ranks[k];
        if (eta[0] >= b[0] && eta[0] < b[1] && eta[1] >= b[2] && eta[1] < b[3] && eta[2] >= b[4] && eta[2] < b[5]) {
            return ranks[k];
        }
    }
    return -2;
}

extern "C" __global__ void sort_destinations(
    const double* __restrict__ markers, long long row_stride, long long col_stride, long long n_rows,
    const bool* __restrict__ holes, const bool* __restrict__ ghosts,
    int shift_col, int init_col, const double* __restrict__ alpha, int use_alpha,
    const int* __restrict__ periodic,        // (3,)
    const double* __restrict__ bounds,       // (mpi_size, 6): left_1, right_1, left_2, right_2, left_3, right_3
    const int* __restrict__ ranks, int n_ranks,  // the other ranks, in the order they are checked
    int own_rank, int* __restrict__ dest,
    unsigned long long* __restrict__ counts)  // (mpi_size + 2,): rows per dest + 2 (dest = -2 first), zeroed by the caller
{
    // counts per destination: block-wide in shared memory, then one global atomic per destination and block
    extern __shared__ unsigned int block_counts[];
    int n_counts = n_ranks + 3;  // mpi_size + 2
    for (int k = threadIdx.x; k < n_counts; k += blockDim.x) block_counts[k] = 0;
    __syncthreads();

    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int d = -1;
    if (i < n_rows) {
        d = row_destination(markers, row_stride, col_stride, i, holes, ghosts, shift_col, init_col, alpha, use_alpha,
                            periodic, bounds, ranks, n_ranks, own_rank);
        dest[i] = d;
        if (d != -1) atomicAdd(&block_counts[d + 2], 1u);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n_counts; k += blockDim.x)
        if (block_counts[k]) atomicAdd(&counts[k], (unsigned long long)block_counts[k]);
}

"""


@cache
def _kernel():
    import cupy as cp

    return cp.RawKernel(_SRC, "sort_destinations")


def domain_bounds(domain_array) -> np.ndarray:
    """``(mpi_size, 6)`` host array of the half-open domain intervals ``[left, right)`` per axis, as in
    :meth:`~struphy.pic.base.Particles._is_on_domain_of_rank`: a right end at eta = 1 is moved to the next float, so
    that eta = 1 belongs to the last rank."""
    d = np.asarray(domain_array, dtype=np.float64)
    left = d[:, 0::3]
    right = d[:, 1::3]
    right = np.where(right == 1.0, np.nextafter(right, 2.0), right)
    out = np.empty((d.shape[0], 6), dtype=np.float64)
    out[:, 0::2] = left
    out[:, 1::2] = right
    return out


def sort_destinations(
    markers, holes, ghosts, shift_col, init_col, alpha, periodic, bounds, ranks, own_rank, dest, counts
):
    """Write the destination of every row of ``markers`` into ``dest`` (int32 device array, see the module docstring)
    and add the number of rows per destination to ``counts``.

    Parameters
    ----------
    markers : cupy.ndarray
        Marker buffer, shape (n_rows, n_cols), any strides.
    holes, ghosts : cupy.ndarray
        Boolean row masks.
    shift_col, init_col : int
        First column of the periodic shifts and of the initial positions (used only with ``alpha``).
    alpha : cupy.ndarray or None
        (3,) float64 weights of the sorting position, None for alpha = 1.
    periodic : cupy.ndarray
        (3,) int32, 1 for periodic axes.
    bounds : cupy.ndarray
        (mpi_size, 6) float64, see :func:`domain_bounds`.
    ranks : cupy.ndarray
        int32 array of the other ranks in the order they are checked.
    own_rank : int
        This rank.
    dest : cupy.ndarray
        int32 output, shape (n_rows,).
    counts : cupy.ndarray
        uint64 array of shape (mpi_size + 2,), zero on entry: ``counts[d + 2]`` gets the number of rows with
        ``dest = d`` for ``d = -2`` and every rank ``d`` (``counts[1]``, for ``d = -1``, stays zero).
    """
    n_rows = markers.shape[0]
    if n_rows == 0:
        return
    item = markers.itemsize
    row_stride, col_stride = markers.strides[0] // item, markers.strides[1] // item
    block = 256
    grid = ((n_rows + block - 1) // block,)
    use_alpha = alpha is not None
    alpha_arg = alpha if use_alpha else bounds  # any device pointer, not read
    _kernel()(
        grid,
        (block,),
        (
            markers,
            np.int64(row_stride),
            np.int64(col_stride),
            np.int64(n_rows),
            holes,
            ghosts,
            np.int32(shift_col),
            np.int32(init_col),
            alpha_arg,
            np.int32(use_alpha),
            periodic,
            bounds,
            ranks,
            np.int32(ranks.size),
            np.int32(own_rank),
            dest,
            counts,
        ),
        shared_mem=4 * (ranks.size + 3),
    )
