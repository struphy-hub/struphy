#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

using namespace struphy_cuda;

/**
 * Velocity update in an electric field, as in push_v_with_efield_kernels.push_v_with_efield.
 *
 * v_p += dt * const_factor * DF^{-T}(eta_p) e(eta_p), with e the 1-form spline field; one thread per marker row.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_markers x n_cols, row-major), validity mask and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param e1_1 Coefficients of the first component of the electric 1-form.
 * @param e1_2 Coefficients of the second component of the electric 1-form.
 * @param e1_3 Coefficients of the third component of the electric 1-form.
 * @param const_factor Constant factor, usually related to the charge-to-mass ratio.
 *
 * Markers that are not valid (holes and ghosts) are skipped.
 */
extern "C" __global__ void push_v_with_efield(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                              DerhamArgs args_derham, Array3D<double> e1_1, Array3D<double> e1_2,
                                              Array3D<double> e1_3, double const_factor) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    // only do something if particle is valid (i.e. not a hole or ghost)
    if (!args_markers.valid_mks[ip]) return;

    double eta1 = args_markers.markers(ip, 0);
    double eta2 = args_markers.markers(ip, 1);
    double eta3 = args_markers.markers(ip, 2);
    double dfm[9], dfinv[9], dfinvt[9], e_form[3], e_cart[3];

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta1, eta2, eta3, args_domain, dfm);

    // metric coeffs
    linalg_kernels::matrix_inv(dfm, dfinv);
    linalg_kernels::transpose(dfinv, dfinvt);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // electric field: 1-form components
    evaluation_kernels_3d::eval_1form_spline_mpi(span1, span2, span3, args_derham, scratch, e1_1, e1_2, e1_3, e_form);

    // electric field: Cartesian components
    linalg_kernels::matrix_vector(dfinvt, e_form, e_cart);

    // update velocities
    for (int j = 0; j < 3; ++j) args_markers.markers(ip, 3 + j) += dt * const_factor * e_cart[j];
}
