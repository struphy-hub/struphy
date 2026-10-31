#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"

using namespace struphy_cuda;

/**
 * Evaluate the magnetic-moment energy mu_p (|B_0| + B_parallel)(eta_p) per marker, as in
 * eval_magnetic_energy_PBb_kernels.eval_magnetic_energy_PBb.
 *
 * @param markers Marker buffer (n_cols x n_markers, one column per marker); one thread per column.
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param args_domain Mapping arguments; every spline and analytic mapping.
 * @param first_diagnostics_idx Marker column of the result.
 * @param mu_idx Marker column of the magnetic moment.
 * @param abs_B0 Coefficients of the 0-form |B_0| (with ghost regions).
 * @param PBb Coefficients of the 0-form B_parallel of the perturbation (with ghost regions).
 *
 * Holes (markers[0, ip] == -1) are skipped. The evaluation point is mod(eta_p, 1). The Jacobian is evaluated but not
 * used, as in the pyccel kernel.
 */
extern "C" __global__ void eval_magnetic_energy_PBb(Array2D<double> markers, DerhamArgs args_derham,
                                                    DomainArgs args_domain, int first_diagnostics_idx, int mu_idx,
                                                    Array3D<double> abs_B0, Array3D<double> PBb) {
    // CUDA-only: the marker of this thread replaces the pyccel loop variable
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= markers.shape[1]) return;

    double eta[3], dfm[9];

    // only do something if particle is a "true" particle (i.e. not a hole)
    if (markers(0, ip) == -1.0) return;

    // numpy.mod(eta, 1.0) (Fortran MODULO): the result has the sign of the divisor
    for (int j = 0; j < 3; ++j) eta[j] = markers(j, ip) - floor(markers(j, ip));

    double mu = markers(mu_idx, ip);

    // spline evaluation; CUDA-only scratch holds the spline values pyccel keeps in args_derham
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta[0], eta[1], eta[2], args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // evaluate Jacobian, result in dfm
    evaluation_kernels::df(eta[0], eta[1], eta[2], args_domain, dfm);

    // abs_B0; 0form
    double abs_B = evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, abs_B0);

    // PBb; 0form
    double PB_b = evaluation_kernels_3d::eval_0form_spline_mpi(span1, span2, span3, args_derham, scratch, PBb);

    markers(first_diagnostics_idx, ip) = mu * (abs_B + PB_b);
}
