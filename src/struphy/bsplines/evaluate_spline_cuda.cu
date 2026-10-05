#include "struphy/bsplines/evaluation_kernels_3d.cuh"

/**
 * Evaluate a distributed tensor-product spline at points, as in evaluation_kernels_3d.eval_spline_mpi_markers.
 *
 * One thread per point. Used by SplineFunction on the CuPy backend for marker and meshgrid evaluation.
 *
 * @param args_derham Spline degrees pn (1 to 8), knots tn1/tn2/tn3 and start indices of the component space;
 *        pyccel takes pn, tn1, tn2, tn3 and starts as separate arguments.
 * @param _data Spline coefficients of the current process (the _data of a StencilVector), any strides.
 * @param eta1 First coordinates of the points (n_points entries); -1 flags a point outside the process domain.
 * @param eta2 Second coordinates of the points.
 * @param eta3 Third coordinates of the points.
 * @param kind1 Kind of basis along the first axis, kind[0] in pyccel: 0 = N-spline, 1 = D-spline.
 * @param kind2 Kind of basis along the second axis, kind[1] in pyccel.
 * @param kind3 Kind of basis along the third axis, kind[2] in pyccel.
 * @param values Output spline values (n_points entries); 0 for flagged points.
 * @param n_points Number of points; CUDA-only, a raw device pointer carries no length.
 */
extern "C" __global__ void evaluate_spline(DerhamArgs args_derham, Array3D<double> _data, const double* eta1,
                                           const double* eta2, const double* eta3, int kind1, int kind2, int kind3,
                                           double* values, int n_points) {
    // CUDA-only: the point of this thread replaces the pyccel loop variable
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= n_points) return;

    // point not in process domain
    if (eta1[ip] == -1. || eta2[ip] == -1. || eta3[ip] == -1.) {
        values[ip] = 0.;
        return;
    }

    // get spline values at eta; CUDA-only scratch holds bn1, ..., bd3 and the spans
    struphy_cuda::SplineScratch scratch;
    struphy_cuda::get_spans(eta1[ip], eta2[ip], eta3[ip], args_derham, scratch);

    const double* b1 = kind1 == 0 ? scratch.bn1 : scratch.bd1;
    const double* b2 = kind2 == 0 ? scratch.bn2 : scratch.bd2;
    const double* b3 = kind3 == 0 ? scratch.bn3 : scratch.bd3;

    values[ip] = struphy_cuda::eval_spline_mpi_kernel(args_derham.pn[0] - kind1, args_derham.pn[1] - kind2,
                                                      args_derham.pn[2] - kind3, b1, b2, b3, scratch.span1,
                                                      scratch.span2, scratch.span3, _data, args_derham.starts);
}
