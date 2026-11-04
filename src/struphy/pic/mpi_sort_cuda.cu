// Destinations of the markers in Particles.mpi_sort_markers on the GPU (struphy.pic.mpi_sort_cuda).
#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "cunumpy/index.cuh"

// The domain test of Particles._is_on_domain_of_rank for one rank: [left, right) in every direction.
__device__ inline bool on_domain(const double* eta, const double* bounds, int r) {
    for (int a = 0; a < 3; ++a) {
        const double left = bounds[(r * 3 + a) * 2], right = bounds[(r * 3 + a) * 2 + 1];
        if (!(eta[a] >= left && eta[a] < right)) return false;
    }
    return true;
}

/**
 * The rank each marker has to go to, as Particles._sendrecv_determine_mtbs and _sendrecv_get_destinations decide it.
 *
 * Marker ip (a column of the buffer, ip < n_used) is sorted by eta = pos + shift (use_alpha == 0) or
 * alpha * (pos + shift) + (1 - alpha) * init per direction (rows 0-2, shift_row.., init_row..), reduced modulo 1 as
 * numpy.mod does; on a non-periodic axis eta = 1 stays 1. It stays (dest -1) if it is a hole or a ghost or lies in
 * the domain of this rank, else it goes to the rank whose domain contains eta (bounds: [left, right) per rank and
 * direction, the right end 1 shifted to the next float, see domain_bounds), or it is lost (dest -2) if no domain
 * contains it (e.g. a NaN position). The domains do not overlap, so the rank is unique.
 *
 * One thread per marker (n_threads = n_used). counts[dest + 2] counts the leaving markers (dest -2 or a rank).
 *
 * @param markers Marker buffer (n_cols x n_markers_max, one column per marker).
 * @param holes, ghosts Hole and ghost masks (n_markers_max).
 * @param shift_row, init_row First rows of the boundary shifts and of the initial positions.
 * @param alpha, use_alpha The three weights, used if use_alpha (else alpha = 1, eta = pos + shift).
 * @param periodic Per direction, 1 if periodic.
 * @param bounds (n_ranks, 3, 2): left and right end of each rank's domain per direction.
 * @param ranks The other ranks in the order they are checked (neighbours first), n_others entries.
 * @param n_others Number of entries of ranks.
 * @param rank This rank.
 * @param n_used Number of markers (columns) that can hold markers.
 * @param dest Output (n_markers_max): -1 stay, -2 lost, else the destination rank (written for ip < n_used).
 * @param counts Output (n_ranks + 2), added to: leaving markers per destination (dest + 2).
 */
extern "C" __global__ void sort_destinations(Array2D<double> markers, const bool* holes, const bool* ghosts,
                                             int shift_row, int init_row, const double* alpha, int use_alpha,
                                             const int* periodic,
                                             const double* bounds, const int* ranks, int n_others, int rank,
                                             long long n_used, int* dest, unsigned long long* counts) {
    CUNUMPY_THREAD_1D(ip, n_used);
    if (holes[ip] || ghosts[ip]) {
        dest[ip] = -1;
        return;
    }

    double eta[3];
    for (int a = 0; a < 3; ++a) {
        const double moved = markers(a, ip) + markers(shift_row + a, ip);
        double e = use_alpha ? alpha[a] * moved + (1.0 - alpha[a]) * markers(init_row + a, ip) : moved;
        const bool right_end = !periodic[a] && e == 1.0;
        // numpy.mod(e, 1.0): the remainder with the sign of the divisor
        double r = fmod(e, 1.0);
        if (r != 0.0) {
            if (r < 0.0) r += 1.0;
        } else {
            r = 0.0;
        }
        eta[a] = right_end ? 1.0 : r;
    }

    if (on_domain(eta, bounds, rank)) {
        dest[ip] = -1;
        return;
    }
    int d = -2;
    for (int k = 0; k < n_others; ++k) {
        if (on_domain(eta, bounds, ranks[k])) {
            d = ranks[k];
            break;
        }
    }
    dest[ip] = d;
    cunumpy_atomic_add(&counts[d + 2], 1ULL);
}
