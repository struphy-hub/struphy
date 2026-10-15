#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the marker weights into a 0-form vector, as in charge_density_0form_kernels.charge_density_0form.
 *
 * B_p = w_p, deposited with the N-splines of V0; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and the weight column weight_idx.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments (unused; part of the common accumulation signature).
 * @param vec 0-form stencil vector data; written with atomic additions.
 *
 * Holes (markers[0, ip] == -1) are skipped. Both loops run over the columns that can hold markers, n_markers.
 */
extern "C" __global__ void charge_density_0form(MarkerArgs args_markers, DerhamArgs args_derham,
                                                DomainArgs args_domain, Array3D<double> vec) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    int weight_idx = args_markers.weight_idx;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.) return;

    // marker positions
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // filling is just the weights
    double filling = args_markers.markers(weight_idx, ip);

    particle_to_mat_kernels::vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, filling);
}
