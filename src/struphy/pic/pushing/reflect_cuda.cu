#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

/**
 * Reflect the velocities of markers pushed outside the logical cube, as in pusher_utilities_kernels.reflect.
 *
 * One thread per entry of outside_inds: v_logical = DF^{-1} v, v_logical[axis] *= -1, v = DF v_logical.
 *
 * @param args_markers Marker buffer (n_markers x n_cols, row-major); pyccel takes the markers array itself.
 * @param args_domain Mapping arguments; only Cuboid (kind_map == 10) is supported.
 * @param outside_inds Rows of the markers to reflect.
 * @param axis Logical velocity component to reverse (0, 1 or 2).
 * @param n_outside_inds Number of entries of outside_inds; CUDA-only, a raw device pointer carries no length.
 */
extern "C" __global__ void reflect(MarkerArgs args_markers, DomainArgs args_domain, const long long* outside_inds,
                                   int axis, int n_outside_inds) {
    // CUDA-only: the entry of outside_inds handled by this thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_outside_inds) return;
    int ip = outside_inds[i];
    struphy_cuda::reflect_velocity(ip, args_markers, args_domain, axis);
}
