#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"

using namespace struphy_cuda;

/**
 * Accumulate the magnetic moments times weights into a 0-form vector, as in
 * gc_mag_density_0form_kernels.gc_mag_density_0form.
 *
 * B_p = mu_p w_p * scale, deposited with the N-splines of V0; one thread per marker.
 *
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker); the weight is in column 5.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments (unused; part of the common accumulation signature).
 * @param vec 0-form stencil vector data; written with atomic additions.
 * @param scale Factor of the filling (e.g. 1/N).
 *
 * Holes (markers[0, ip] == -1) are skipped.
 */
extern "C" __global__ void gc_mag_density_0form(MarkerArgs args_markers, DerhamArgs args_derham,
                                                DomainArgs args_domain, Array3D<double> vec, double scale) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    int mu_idx = args_markers.mu_idx;

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // marker positions
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // marker weight and magnetic moment
    double weight = args_markers.markers(5, ip);
    double mu = args_markers.markers(mu_idx, ip);

    // filling =mu*w_p/N
    double filling = mu * weight * scale;

    particle_to_mat_kernels::vec_fill_b_v0(args_derham, eta1, eta2, eta3, vec, filling);
}
