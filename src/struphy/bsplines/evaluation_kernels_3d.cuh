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
