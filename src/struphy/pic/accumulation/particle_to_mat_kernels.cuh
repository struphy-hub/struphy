// Device versions of the helpers in particle_to_mat_kernels.py (one marker at a time).
#pragma once
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/pic/accumulation/filler_kernels.cuh"
namespace struphy_cuda {
/**
 * Add one marker's contribution to a 0-form vector, as in particle_to_mat_kernels.vec_fill_b_v0.
 *
 * @param args_derham Spline degrees (1 to 8), knots and start indices.
 * @param eta1 Logical position along the first axis.
 * @param eta2 Logical position along the second axis.
 * @param eta3 Logical position along the third axis.
 * @param vec 0-form stencil vector data (the _data of a StencilVector); written atomically.
 * @param fill Number multiplied by the N-splines at (eta1, eta2, eta3).
 */
__device__ inline void vec_fill_b_v0(const DerhamArgs& args_derham, double eta1, double eta2, double eta3,
                                     Array3D<double> vec, double fill) {
    // degrees of the basis functions : B-splines (pn)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    // CUDA-only scratch holds the spans and the spline values pyccel keeps in args_derham
    SplineScratch scratch;
    get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    fill_vec(pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, vec,
             fill);
}
}
