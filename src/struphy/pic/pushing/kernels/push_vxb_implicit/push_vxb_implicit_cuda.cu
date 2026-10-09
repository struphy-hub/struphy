#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"

/**
 * Implicit (Crank-Nicolson) rotation of the velocity, as in push_vxb_implicit_kernels.push_vxb_implicit.
 *
 * Solves (I - dt/2 B_x) v^{n+1} = (I + dt/2 B_x) v^n, where B_x is the cross-product matrix of
 * DF B^2 / sqrt(g) at the marker position, one thread per marker row.
 *
 * @param dt Time step.
 * @param stage Stage index (unused; part of the common pusher signature).
 * @param args_markers Marker buffer (n_markers x n_cols, row-major) and column indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param b2_1 Coefficients of the first component of the magnetic 2-form.
 * @param b2_2 Coefficients of the second component of the magnetic 2-form.
 * @param b2_3 Coefficients of the third component of the magnetic 2-form.
 *
 * Holes are skipped (boundary particles are pushed, as in the pyccel kernel).
 */
extern "C" __global__ void push_vxb_implicit(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                                             DerhamArgs args_derham, Array3D<double> b2_1, Array3D<double> b2_2,
                                             Array3D<double> b2_3) {
    // CUDA-only: the marker row of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    int n_markers = args_markers.n_markers;
    if (ip >= n_markers) return;

    int first_pusher_idx = args_markers.first_pusher_idx;

    // check if marker is a hole
    if (args_markers.markers(ip, first_pusher_idx) == -1.) return;

    double e1 = args_markers.markers(ip, 0);
    double e2 = args_markers.markers(ip, 1);
    double e3 = args_markers.markers(ip, 2);
    double v[3], dfm[9], b_form[3], b_cart[3], vec[3], res[3];
    double identity[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    double b_prod[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double rhs[9], lhs[9], lhs_inv[9];
    for (int j = 0; j < 3; ++j) v[j] = args_markers.markers(ip, 3 + j);

    // evaluate Jacobian, result in dfm
    struphy_cuda::df(e1, e2, e3, args_domain, dfm);

    // metric coeffs
    double det_df = struphy_cuda::det(dfm);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    struphy_cuda::SplineScratch scratch;
    struphy_cuda::get_spans(e1, e2, e3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // magnetic field 2-form
    struphy_cuda::eval_2form_spline_mpi(span1, span2, span3, args_derham, scratch, b2_1, b2_2, b2_3, b_form);

    // magnetic field: Cartesian components
    struphy_cuda::matrix_vector(dfm, b_form, b_cart);
    for (int j = 0; j < 3; ++j) b_cart[j] = b_cart[j] / det_df;

    // cross-product matrix of b_cart (row-major)
    b_prod[1] = b_cart[2];
    b_prod[2] = -b_cart[1];
    b_prod[3] = -b_cart[2];
    b_prod[5] = b_cart[0];
    b_prod[6] = b_cart[1];
    b_prod[7] = -b_cart[0];

    // solve 3x3 system
    for (int j = 0; j < 9; ++j) {
        rhs[j] = identity[j] + dt / 2 * b_prod[j];
        lhs[j] = identity[j] - dt / 2 * b_prod[j];
    }

    struphy_cuda::matrix_inv(lhs, lhs_inv);

    struphy_cuda::matrix_vector(rhs, v, vec);
    struphy_cuda::matrix_vector(lhs_inv, vec, res);

    for (int j = 0; j < 3; ++j) args_markers.markers(ip, 3 + j) = res[j];
}
