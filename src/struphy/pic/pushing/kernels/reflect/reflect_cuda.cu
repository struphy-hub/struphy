#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

/**
 * Reflect velocities with the same arguments as the pyccel reflect (reflect_kernels.py).
 *
 * One thread per entry of outside_inds: v_logical = DF^{-1} v,
 * v_logical[axis] *= -1, v = DF v_logical.
 *
 * @param markers Marker array view; positions are already inside the logical cube.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param outside_inds View of the marker row indices to reflect.
 * @param axis Logical velocity component to reverse (0, 1 or 2).
 */
extern "C" __global__ void reflect(Array2D<double> markers, DomainArgs args_domain,
                                   Array1D<long long> outside_inds, int axis) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= outside_inds.shape[0]) return;
    struphy_cuda::reflect_velocity(outside_inds(i), markers, args_domain, axis);
}
