#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Velocity update of PressureCoupling6D with the perpendicular velocity, as in push_pc_GXu_kernels.push_pc_GXu.
 *
 * v_p -= dt/2 DF^{-T} (grad(X u)^T v_perp) at eta_p, where row i of grad(X u) is the 1-form grad (X u)_i and
 * v_perp = (v_1, v_2, 0); one thread per marker.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param GXu_11 Coefficients of the first component of the 1-form grad (X u)_1.
 * @param GXu_12 Coefficients of the second component of grad (X u)_1.
 * @param GXu_13 Coefficients of the third component of grad (X u)_1.
 * @param GXu_21 Coefficients of the first component of the 1-form grad (X u)_2.
 * @param GXu_22 Coefficients of the second component of grad (X u)_2.
 * @param GXu_23 Coefficients of the third component of grad (X u)_2.
 * @param GXu_31 Coefficients of the first component of grad (X u)_3 (unused, as in pyccel).
 * @param GXu_32 Coefficients of the second component of grad (X u)_3 (unused).
 * @param GXu_33 Coefficients of the third component of grad (X u)_3 (unused).
 *
 * Holes (markers[0, ip] == -1) are skipped, as in pyccel.
 */
extern "C" __global__ void push_pc_GXu(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                       DerhamArgs args_derham, Array3D<double> GXu_11, Array3D<double> GXu_12,
                                       Array3D<double> GXu_13, Array3D<double> GXu_21, Array3D<double> GXu_22,
                                       Array3D<double> GXu_23, Array3D<double> GXu_31, Array3D<double> GXu_32,
                                       Array3D<double> GXu_33) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate metric coeffs
    double dfm[9], dfinv[9], dfinv_t[9];

    // allocate for field evaluations (GXu as nine row-major entries)
    double e[3], e_cart[3], GXu[9];

    // particle velocity
    double v[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs
    linalg_kernels::matrix_inv(dfm, dfinv);
    linalg_kernels::transpose(dfinv, dfinv_t);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // Evaluate grad(X(u, v)) at the particle positions (rows 0 and 1 of GXu)
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, GXu_11, GXu_12, GXu_13,
                                                 GXu);
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, GXu_21, GXu_22, GXu_23,
                                                 GXu + 3);

    e[0] = GXu[0] * v[0] + GXu[3] * v[1];
    e[1] = GXu[1] * v[0] + GXu[4] * v[1];
    e[2] = GXu[2] * v[0] + GXu[5] * v[1];

    linalg_kernels::matrix_vector(dfinv_t, e, e_cart);

    // update velocities
    for (int j = 0; j < 3; ++j) args_markers.markers(3 + j, ip) -= dt * e_cart[j] / 2.0;
}
