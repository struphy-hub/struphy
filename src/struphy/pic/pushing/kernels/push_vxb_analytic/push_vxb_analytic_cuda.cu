// NVRTC provides the device math functions (sqrt, cos, sin) without a host math.h header.
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"

using namespace struphy_cuda;

/**
 * Exact rotation of the velocity about the magnetic field, as in push_vxb_analytic_kernels.push_vxb_analytic.
 *
 * Solves d v_p / dt = v_p x DF B^2 / sqrt(g) with the field frozen at the marker position, one thread
 * per marker.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param b2_1 Coefficients of the first component of the magnetic 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic 2-form.
 *
 * Holes and boundary particles are skipped; markers in a vanishing field keep their velocity.
 */
extern "C" __global__ void push_vxb_analytic(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                             DerhamArgs args_derham, Array3D<double> b2_1, Array3D<double> b2_2,
                                             Array3D<double> b2_3) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // CUDA-only: prepare this marker here instead of in a separate pass of prepare_push (Pusher._push)
    if (stage == 0 && args_markers.prepare_rows) pusher_utilities_kernels::prepare_marker_row(ip, args_markers);

    int first_pusher_idx = args_markers.first_pusher_idx;

    // check if marker is a hole
    if (args_markers.markers(first_pusher_idx, ip) == -1. || args_markers.markers(args_markers.markers.shape[0] - 1, ip) == -2.)
        return;

    double e1 = args_markers.markers(0, ip);
    double e2 = args_markers.markers(1, ip);
    double e3 = args_markers.markers(2, ip);
    double v[3], dfm[9], b_form[3], b_cart[3], b_norm[3], vperp[3], vxb_norm[3], b_normxvperp[3];
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(3 + j, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(e1, e2, e3, args_domain, dfm);

    // metric coeffs
    double det_df = linalg_kernels::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(e1, e2, e3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // magnetic field 2-form
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b_form);

    // magnetic field: Cartesian components
    linalg_kernels::matrix_vector(dfm, b_form, b_cart);
    for (int j = 0; j < 3; ++j) b_cart[j] = b_cart[j] / det_df;

    // magnetic field: magnitude
    double b_abs = sqrt(b_cart[0] * b_cart[0] + b_cart[1] * b_cart[1] + b_cart[2] * b_cart[2]);

    // only push vxb if magnetic field is non-zero
    if (b_abs == 0.) return;

    // normalized magnetic field direction
    for (int j = 0; j < 3; ++j) b_norm[j] = b_cart[j] / b_abs;

    // parallel velocity v.b_norm
    double vpar = linalg_kernels::scalar_dot(v, b_norm);

    // first component of perpendicular velocity
    linalg_kernels::cross(v, b_norm, vxb_norm);
    linalg_kernels::cross(b_norm, vxb_norm, vperp);

    // second component of perpendicular velocity
    linalg_kernels::cross(b_norm, vperp, b_normxvperp);

    // analytic rotation
    for (int j = 0; j < 3; ++j)
        args_markers.markers(3 + j, ip) = vpar * b_norm[j] + cos(b_abs * dt) * vperp[j] - sin(b_abs * dt) * b_normxvperp[j];
}
