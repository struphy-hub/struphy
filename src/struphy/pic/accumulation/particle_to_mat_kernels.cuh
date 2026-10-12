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

/**
 * Add one marker's contribution to the symmetric blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of a V1 -> V1
 * block matrix and to a V1 vector, as in particle_to_mat_kernels.m_v_fill_b_v1_symm.
 *
 * @param args_derham Spline degrees (1 to 8), knots and start indices; pn is also the padding of the blocks.
 * @param eta1 Logical position along the first axis.
 * @param eta2 Logical position along the second axis.
 * @param eta3 Logical position along the third axis.
 * @param mat11 (mu=1, nu=1)-block data (D N N x D N N), shape (n1, n2, n3, 2 * pn + 1); written atomically.
 * @param mat12 (mu=1, nu=2)-block data (D N N x N D N); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (D N N x N N D); written atomically.
 * @param mat22 (mu=2, nu=2)-block data (N D N x N D N); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (N D N x N N D); written atomically.
 * @param mat33 (mu=3, nu=3)-block data (N N D x N N D); written atomically.
 * @param fill11 Number multiplied by the basis functions and added to mat11.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill22 Number multiplied by the basis functions and added to mat22.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 * @param fill33 Number multiplied by the basis functions and added to mat33.
 * @param vec1 mu=1 component of the V1 vector data (D N N); written atomically.
 * @param vec2 mu=2 component of the V1 vector data (N D N); written atomically.
 * @param vec3 mu=3 component of the V1 vector data (N N D); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 *
 * Pyccel keeps the spline values in args_derham.bn1, ..., bd3; here they are in a thread-local SplineScratch.
 */
__device__ inline void m_v_fill_b_v1_symm(const DerhamArgs& args_derham, double eta1, double eta2, double eta3,
                                          Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                          Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                          double fill11, double fill12, double fill13, double fill22,
                                          double fill23, double fill33, Array3D<double> vec1, Array3D<double> vec2,
                                          Array3D<double> vec3, double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // CUDA-only scratch holds the spans and the spline values pyccel keeps in args_derham
    SplineScratch scratch;
    get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // fill matrix entries
    fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bd1, scratch.bn2,
                 scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat11, fill11, vec1, fill1);

    fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1, scratch.bd2,
                 scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat22, fill22, vec2, fill2);

    fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3, scratch.bn1, scratch.bn2,
                 scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat33, fill33, vec3, fill3);

    fill_mat(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1, scratch.bd2,
             scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12, fill12);

    fill_mat(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1, scratch.bn2,
             scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13, fill13);

    fill_mat(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1, scratch.bn2,
             scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23, fill23);
}
}
