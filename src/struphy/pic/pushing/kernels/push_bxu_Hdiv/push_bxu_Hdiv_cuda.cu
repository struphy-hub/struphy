#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Velocity update of CurrentCoupling6DCurrent with U in H(div), as in push_bxu_Hdiv_kernels.push_bxu_Hdiv.
 *
 * v_p += dt (DF B^2 / sqrt(g)) x (DF U^2 / sqrt(g)) at eta_p; one thread per marker.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_cols x n_markers, one column per marker) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param b2_1 Coefficients of the first component of the magnetic field as a 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic field as a 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic field as a 2-form.
 * @param u2_1 Coefficients of the first component of U as a 2-form.
 * @param u2_2 Coefficients of the second component of U as a 2-form.
 * @param u2_3 Coefficients of the third component of U as a 2-form.
 * @param boundary_cut Markers with eta1 < boundary_cut or eta1 > 1 - boundary_cut are not pushed.
 *
 * Holes (markers[0, ip] == -1) are skipped, as in pyccel.
 */
extern "C" __global__ void push_bxu_Hdiv(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                         DerhamArgs args_derham, Array3D<double> b2_1, Array3D<double> b2_2,
                                         Array3D<double> b2_3, Array3D<double> u2_1, Array3D<double> u2_2,
                                         Array3D<double> u2_3, double boundary_cut) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= args_markers.n_markers) return;

    // allocate metric coeffs
    double dfm[9];

    // allocate for field evaluations (2-form and Cartesian components)
    double b_form[3], u_form[3], b_cart[3], u_cart[3], e_cart[3];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (args_markers.markers(0, ip) == -1.0) return;

    // boundary cut
    if (args_markers.markers(0, ip) < boundary_cut || args_markers.markers(0, ip) > 1.0 - boundary_cut) return;

    // marker data
    double eta1 = args_markers.markers(0, ip);
    double eta2 = args_markers.markers(1, ip);
    double eta3 = args_markers.markers(2, ip);

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs
    double det_df = linalg_kernels::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // magnetic field: 2-form components
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b_form);

    // magnetic field: Cartesian components
    linalg_kernels::matrix_vector(dfm, b_form, b_cart);
    for (int j = 0; j < 3; ++j) b_cart[j] = b_cart[j] / det_df;

    // velocity field: 2-form components
    evaluation_kernels_3d::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, u2_1, u2_2, u2_3, u_form);

    // velocity field: Cartesian components
    linalg_kernels::matrix_vector(dfm, u_form, u_cart);
    for (int j = 0; j < 3; ++j) u_cart[j] = u_cart[j] / det_df;

    // electric field E = B x U
    linalg_kernels::cross(b_cart, u_cart, e_cart);

    // update velocities
    for (int j = 0; j < 3; ++j) args_markers.markers(3 + j, ip) += dt * e_cart[j];
}
