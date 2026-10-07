#pragma once
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/bsplines/bsplines_kernels.cuh"
namespace struphy_cuda {
/** Per-thread outputs of get_spans; spline names match DerhamArguments. */
struct SplineScratch {
    int span1, span2, span3;
    double bn1[MAX_SPLINE_DEGREE + 1], bn2[MAX_SPLINE_DEGREE + 1], bn3[MAX_SPLINE_DEGREE + 1];
    double bd1[MAX_SPLINE_DEGREE], bd2[MAX_SPLINE_DEGREE], bd3[MAX_SPLINE_DEGREE];
};

/**
 * Compute knot spans and B-/D-spline values, as in evaluation_kernels_3d.get_spans.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param args_derham Knot sequences, lengths and degrees (between 1 and 8).
 * @param scratch Thread-local output spans and spline values.
 *
 * Pyccel returns span1/span2/span3 and writes bn1/bd1, etc. into args_derham.
 * CUDA writes all outputs into scratch to avoid sharing mutable arrays
 * between marker threads.
 */
__device__ inline void get_spans(double eta1, double eta2, double eta3,
                                const DerhamArgs& args_derham, SplineScratch& scratch) {
    int span1 = find_span(args_derham.tn1, args_derham.nt1, args_derham.pn[0], eta1);
    int span2 = find_span(args_derham.tn2, args_derham.nt2, args_derham.pn[1], eta2);
    int span3 = find_span(args_derham.tn3, args_derham.nt3, args_derham.pn[2], eta3);
    scratch.span1 = span1;
    scratch.span2 = span2;
    scratch.span3 = span3;
    b_d_splines_slim(args_derham.tn1, args_derham.pn[0], eta1, span1, scratch.bn1, scratch.bd1);
    b_d_splines_slim(args_derham.tn2, args_derham.pn[1], eta2, span2, scratch.bn2, scratch.bd2);
    b_d_splines_slim(args_derham.tn3, args_derham.pn[2], eta3, span3, scratch.bn3, scratch.bd3);
}
}

#include "cunumpy/array_view.cuh"
namespace struphy_cuda {
/**
 * Sum the non-zero contributions of a distributed spline, as in evaluation_kernels_3d.eval_spline_mpi_kernel.
 *
 * @param p1 Degree of the univariate splines along the first axis.
 * @param p2 Degree of the univariate splines along the second axis.
 * @param p3 Degree of the univariate splines along the third axis.
 * @param basis1 The p1 + 1 non-zero basis values along the first axis.
 * @param basis2 The p2 + 1 non-zero basis values along the second axis.
 * @param basis3 The p3 + 1 non-zero basis values along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param _data Spline coefficients of the current process (the _data of a StencilVector); strides as in cunumpy/array_view.cuh.
 * @param starts Start indices of the current process (three entries).
 * @return spline_value, the value of the tensor-product spline.
 */
__device__ inline double eval_spline_mpi_kernel(int p1, int p2, int p3, const double* basis1, const double* basis2,
                                                const double* basis3, int span1, int span2, int span3,
                                                Array3D<double> _data, const long long* starts) {
    double spline_value = 0.;
    for (int il1 = 0; il1 <= p1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        for (int il2 = 0; il2 <= p2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            for (int il3 = 0; il3 <= p3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                spline_value += _data(i1, i2, i3) * basis1[il1] * basis2[il2] * basis3[il3];
            }
        }
    }
    return spline_value;
}

/**
 * Evaluate the three components of a 1-form spline, as in evaluation_kernels_3d.eval_1form_spline_mpi.
 *
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param args_derham Spline degrees and start indices.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param form_coeffs_1 Coefficients of the first component (D N N).
 * @param form_coeffs_2 Coefficients of the second component (N D N).
 * @param form_coeffs_3 Coefficients of the third component (N N D).
 * @param out Output buffer for the three components.
 */
__device__ inline void eval_1form_spline_mpi(int span1, int span2, int span3, const DerhamArgs& args_derham,
                                             const SplineScratch& scratch, Array3D<double> form_coeffs_1,
                                             Array3D<double> form_coeffs_2, Array3D<double> form_coeffs_3,
                                             double* out) {
    out[0] = eval_spline_mpi_kernel(args_derham.pn[0] - 1, args_derham.pn[1], args_derham.pn[2], scratch.bd1,
                                    scratch.bn2, scratch.bn3, span1, span2, span3, form_coeffs_1, args_derham.starts);
    out[1] = eval_spline_mpi_kernel(args_derham.pn[0], args_derham.pn[1] - 1, args_derham.pn[2], scratch.bn1,
                                    scratch.bd2, scratch.bn3, span1, span2, span3, form_coeffs_2, args_derham.starts);
    out[2] = eval_spline_mpi_kernel(args_derham.pn[0], args_derham.pn[1], args_derham.pn[2] - 1, scratch.bn1,
                                    scratch.bn2, scratch.bd3, span1, span2, span3, form_coeffs_3, args_derham.starts);
}

/**
 * Evaluate the three components of a 2-form spline, as in evaluation_kernels_3d.eval_2form_spline_mpi.
 *
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param args_derham Spline degrees and start indices.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param form_coeffs_1 Coefficients of the first component (N D D).
 * @param form_coeffs_2 Coefficients of the second component (D N D).
 * @param form_coeffs_3 Coefficients of the third component (D D N).
 * @param out Output buffer for the three components.
 */
__device__ inline void eval_2form_spline_mpi(int span1, int span2, int span3, const DerhamArgs& args_derham,
                                             const SplineScratch& scratch, Array3D<double> form_coeffs_1,
                                             Array3D<double> form_coeffs_2, Array3D<double> form_coeffs_3,
                                             double* out) {
    out[0] = eval_spline_mpi_kernel(args_derham.pn[0], args_derham.pn[1] - 1, args_derham.pn[2] - 1, scratch.bn1,
                                    scratch.bd2, scratch.bd3, span1, span2, span3, form_coeffs_1, args_derham.starts);
    out[1] = eval_spline_mpi_kernel(args_derham.pn[0] - 1, args_derham.pn[1], args_derham.pn[2] - 1, scratch.bd1,
                                    scratch.bn2, scratch.bd3, span1, span2, span3, form_coeffs_2, args_derham.starts);
    out[2] = eval_spline_mpi_kernel(args_derham.pn[0] - 1, args_derham.pn[1] - 1, args_derham.pn[2], scratch.bd1,
                                    scratch.bd2, scratch.bn3, span1, span2, span3, form_coeffs_3, args_derham.starts);
}
}

// Point-wise evaluation shared by the eval_spline_mpi_* kernels (bsplines/kernels/<name>/<name>_cuda.cu).
/**
 * Point-wise evaluation of a distributed tensor-product spline, as in evaluation_kernels_3d.eval_spline_mpi.
 *
 * @param eta1 Evaluation point along the first axis.
 * @param eta2 Evaluation point along the second axis.
 * @param eta3 Evaluation point along the third axis.
 * @param _data Spline coefficients of the current process (the _data of a StencilVector), any strides.
 * @param kind Kind of 1d basis in each direction (three entries): 0 = N-spline, 1 = D-spline.
 * @param pn Spline degrees of V0 in each direction (three entries, 1 to 8).
 * @param tn1 Knot vector of V0 along the first axis (contiguous).
 * @param tn2 Knot vector of V0 along the second axis.
 * @param tn3 Knot vector of V0 along the third axis.
 * @param starts Start indices of the splines on the current process (three entries).
 * @return value, the value of the spline at (eta1, eta2, eta3).
 *
 * Pyccel allocates bn1, ..., bd3 per call; here they are fields of the thread-local SplineScratch.
 */
__device__ inline double eval_spline_mpi(double eta1, double eta2, double eta3, Array3D<double> _data,
                                         const long long* kind, const long long* pn, Array1D<double> tn1,
                                         Array1D<double> tn2, Array1D<double> tn3, const long long* starts) {
    struphy_cuda::SplineScratch scratch;

    // get spline values at eta
    scratch.span1 = struphy_cuda::find_span(tn1.data, tn1.shape[0], pn[0], eta1);
    scratch.span2 = struphy_cuda::find_span(tn2.data, tn2.shape[0], pn[1], eta2);
    scratch.span3 = struphy_cuda::find_span(tn3.data, tn3.shape[0], pn[2], eta3);
    struphy_cuda::b_d_splines_slim(tn1.data, pn[0], eta1, scratch.span1, scratch.bn1, scratch.bd1);
    struphy_cuda::b_d_splines_slim(tn2.data, pn[1], eta2, scratch.span2, scratch.bn2, scratch.bd2);
    struphy_cuda::b_d_splines_slim(tn3.data, pn[2], eta3, scratch.span3, scratch.bn3, scratch.bd3);

    const double* b1 = kind[0] == 0 ? scratch.bn1 : scratch.bd1;
    const double* b2 = kind[1] == 0 ? scratch.bn2 : scratch.bd2;
    const double* b3 = kind[2] == 0 ? scratch.bn3 : scratch.bd3;

    double value = struphy_cuda::eval_spline_mpi_kernel(pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3,
                                                        scratch.span1, scratch.span2, scratch.span3, _data, starts);
    return value;
}
