// Device versions of the helpers in particle_to_mat_kernels.py (one marker at a time).
#pragma once
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/pic/accumulation/filler_kernels.cuh"
namespace struphy_cuda::particle_to_mat_kernels {
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
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    filler_kernels::fill_vec(pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec, fill);
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
    evaluation_kernels_3d::SplineScratch scratch;
    evaluation_kernels_3d::get_spans(eta1, eta2, eta3, args_derham, scratch);
    int span1 = scratch.span1, span2 = scratch.span2, span3 = scratch.span3;

    // fill matrix entries
    filler_kernels::fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bd1,
                                 scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat11, fill11, vec1, fill1);

    filler_kernels::fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                                 scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat22, fill22, vec2, fill2);

    filler_kernels::fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3, scratch.bn1,
                                 scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat33, fill33, vec3, fill3);

    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to a V1 vector at known spans, as in particle_to_mat_kernels.vec_fill_v1.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param vec1 mu=1 component of the V1 vector data (D N N); written atomically.
 * @param vec2 mu=2 component of the V1 vector data (N D N); written atomically.
 * @param vec3 mu=3 component of the V1 vector data (N N D); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void vec_fill_v1(const DerhamArgs& args_derham, const evaluation_kernels_3d::SplineScratch& scratch,
                                   int span1, int span2, int span3, Array3D<double> vec1, Array3D<double> vec2,
                                   Array3D<double> vec3, double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill vector entries
    filler_kernels::fill_vec(pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec1, fill1);
    filler_kernels::fill_vec(pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec2, fill2);
    filler_kernels::fill_vec(pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3, span1, span2, span3,
                             args_derham.starts, vec3, fill3);
}

/**
 * Add one marker's contribution to a V2 vector at known spans, as in particle_to_mat_kernels.vec_fill_v2.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param vec1 mu=1 component of the V2 vector data (N D D); written atomically.
 * @param vec2 mu=2 component of the V2 vector data (D N D); written atomically.
 * @param vec3 mu=3 component of the V2 vector data (D D N); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void vec_fill_v2(const DerhamArgs& args_derham, const evaluation_kernels_3d::SplineScratch& scratch,
                                   int span1, int span2, int span3, Array3D<double> vec1, Array3D<double> vec2,
                                   Array3D<double> vec3, double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill vector entries
    filler_kernels::fill_vec(pn1, pd2, pd3, scratch.bn1, scratch.bd2, scratch.bd3, span1, span2, span3,
                             args_derham.starts, vec1, fill1);
    filler_kernels::fill_vec(pd1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bd3, span1, span2, span3,
                             args_derham.starts, vec2, fill2);
    filler_kernels::fill_vec(pd1, pd2, pn3, scratch.bd1, scratch.bd2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec3, fill3);
}

/**
 * Add one marker's contribution to a vector-valued V0 vector ((H^1)^3) at known spans, as in
 * particle_to_mat_kernels.vec_fill_v0vec.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices.
 * @param scratch N-spline values from get_spans; pyccel reads them from args_derham.bn1, bn2, bn3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param vec1 First component of the vector data (N N N); written atomically.
 * @param vec2 Second component of the vector data (N N N); written atomically.
 * @param vec3 Third component of the vector data (N N N); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void vec_fill_v0vec(const DerhamArgs& args_derham,
                                      const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                      int span3, Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                      double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    // fill vector entries
    filler_kernels::fill_vec(pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec1, fill1);
    filler_kernels::fill_vec(pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec2, fill2);
    filler_kernels::fill_vec(pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, span1, span2, span3,
                             args_derham.starts, vec3, fill3);
}

/**
 * Add one marker's contribution to the antisymmetric blocks (1,2), (1,3), (2,3) of a V0vec -> V0vec ((H^1)^3) block
 * matrix at known spans, as in particle_to_mat_kernels.mat_fill_v0vec_asym.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat12 (mu=1, nu=2)-block data (N N N x N N N); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (N N N x N N N); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (N N N x N N N); written atomically.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 */
__device__ inline void mat_fill_v0vec_asym(const DerhamArgs& args_derham,
                                           const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                           int span3, Array6D<double> mat12, Array6D<double> mat13,
                                           Array6D<double> mat23, double fill12, double fill13, double fill23) {
    // degrees of the basis functions : B-splines (pn)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    // fill matrix entries
    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the antisymmetric blocks (1,2), (1,3), (2,3) of a V1 -> V1 block matrix at known
 * spans, as in particle_to_mat_kernels.mat_fill_v1_asym.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat12 (mu=1, nu=2)-block data (D N N x N D N); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (D N N x N N D); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (N D N x N N D); written atomically.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 */
__device__ inline void mat_fill_v1_asym(const DerhamArgs& args_derham,
                                        const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                        int span3, Array6D<double> mat12, Array6D<double> mat13, Array6D<double> mat23,
                                        double fill12, double fill13, double fill23) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the antisymmetric blocks (1,2), (1,3), (2,3) of a V2 -> V2 block matrix at known
 * spans, as in particle_to_mat_kernels.mat_fill_v2_asym.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat12 (mu=1, nu=2)-block data (N D D x D N D); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (N D D x D D N); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (D N D x D D N); written atomically.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 */
__device__ inline void mat_fill_v2_asym(const DerhamArgs& args_derham,
                                        const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                        int span3, Array6D<double> mat12, Array6D<double> mat13, Array6D<double> mat23,
                                        double fill12, double fill13, double fill23) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat(pn1, pd2, pd3, pd1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bd3, scratch.bd1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pn1, pd2, pd3, pd1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bd3, scratch.bd1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pd1, pn2, pd3, pd1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bd3, scratch.bd1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the symmetric blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of a V0vec -> V0vec
 * ((H^1)^3) block matrix and to a V0vec vector at known spans, as in particle_to_mat_kernels.m_v_fill_v0vec_symm.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat11 (mu=1, nu=1)-block data (N N N x N N N); written atomically.
 * @param mat12 (mu=1, nu=2)-block data (N N N x N N N); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (N N N x N N N); written atomically.
 * @param mat22 (mu=2, nu=2)-block data (N N N x N N N); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (N N N x N N N); written atomically.
 * @param mat33 (mu=3, nu=3)-block data (N N N x N N N); written atomically.
 * @param fill11 Number multiplied by the basis functions and added to mat11.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill22 Number multiplied by the basis functions and added to mat22.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 * @param fill33 Number multiplied by the basis functions and added to mat33.
 * @param vec1 mu=1 component of the vector data (N N N); written atomically.
 * @param vec2 mu=2 component of the vector data (N N N); written atomically.
 * @param vec3 mu=3 component of the vector data (N N N); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void m_v_fill_v0vec_symm(
    const DerhamArgs& args_derham, const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2, int span3,
    Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13, Array6D<double> mat22, Array6D<double> mat23,
    Array6D<double> mat33, double fill11, double fill12, double fill13, double fill22, double fill23, double fill33,
    Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3, double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    // fill matrix entries
    filler_kernels::fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                                 scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat11, fill11, vec1, fill1);

    filler_kernels::fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                                 scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat22, fill22, vec2, fill2);

    filler_kernels::fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                                 scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat33, fill33, vec3, fill3);

    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pn1, pn2, pn3, pn1, pn2, pn3, scratch.bn1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the symmetric blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of a V1 ->
 * V1 block matrix and to a V1 vector at known spans, as in particle_to_mat_kernels.m_v_fill_v1_symm.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat11 (mu=1, nu=1)-block data (D N N x D N N); written atomically.
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
 * @param vec1 mu=1 component of the vector data (D N N); written atomically.
 * @param vec2 mu=2 component of the vector data (N D N); written atomically.
 * @param vec3 mu=3 component of the vector data (N N D); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void m_v_fill_v1_symm(const DerhamArgs& args_derham,
                                        const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                        int span3, Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                        Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                        double fill11, double fill12, double fill13, double fill22, double fill23,
                                        double fill33, Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                        double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bd1,
                                 scratch.bn2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat11, fill11, vec1, fill1);

    filler_kernels::fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                                 scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat22, fill22, vec2, fill2);

    filler_kernels::fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3, scratch.bn1,
                                 scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat33, fill33, vec3, fill3);

    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the symmetric blocks (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) of a V2 ->
 * V2 block matrix and to a V2 vector at known spans, as in particle_to_mat_kernels.m_v_fill_v2_symm.
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat11 (mu=1, nu=1)-block data (N D D x N D D); written atomically.
 * @param mat12 (mu=1, nu=2)-block data (N D D x D N D); written atomically.
 * @param mat13 (mu=1, nu=3)-block data (N D D x D D N); written atomically.
 * @param mat22 (mu=2, nu=2)-block data (D N D x D N D); written atomically.
 * @param mat23 (mu=2, nu=3)-block data (D N D x D D N); written atomically.
 * @param mat33 (mu=3, nu=3)-block data (D D N x D D N); written atomically.
 * @param fill11 Number multiplied by the basis functions and added to mat11.
 * @param fill12 Number multiplied by the basis functions and added to mat12.
 * @param fill13 Number multiplied by the basis functions and added to mat13.
 * @param fill22 Number multiplied by the basis functions and added to mat22.
 * @param fill23 Number multiplied by the basis functions and added to mat23.
 * @param fill33 Number multiplied by the basis functions and added to mat33.
 * @param vec1 mu=1 component of the vector data (N D D); written atomically.
 * @param vec2 mu=2 component of the vector data (D N D); written atomically.
 * @param vec3 mu=3 component of the vector data (D D N); written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1.
 * @param fill2 Number multiplied by the basis functions and added to vec2.
 * @param fill3 Number multiplied by the basis functions and added to vec3.
 */
__device__ inline void m_v_fill_v2_symm(const DerhamArgs& args_derham,
                                        const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2,
                                        int span3, Array6D<double> mat11, Array6D<double> mat12, Array6D<double> mat13,
                                        Array6D<double> mat22, Array6D<double> mat23, Array6D<double> mat33,
                                        double fill11, double fill12, double fill13, double fill22, double fill23,
                                        double fill33, Array3D<double> vec1, Array3D<double> vec2, Array3D<double> vec3,
                                        double fill1, double fill2, double fill3) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3, scratch.bn1, scratch.bd2, scratch.bd3, scratch.bn1,
                                 scratch.bd2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat11, fill11, vec1, fill1);

    filler_kernels::fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bd3, scratch.bd1,
                                 scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat22, fill22, vec2, fill2);

    filler_kernels::fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3, scratch.bd1, scratch.bd2, scratch.bn3, scratch.bd1,
                                 scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                 mat33, fill33, vec3, fill3);

    filler_kernels::fill_mat(pn1, pd2, pd3, pd1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bd3, scratch.bd1,
                             scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn, mat12,
                             fill12);

    filler_kernels::fill_mat(pn1, pd2, pd3, pd1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bd3, scratch.bd1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat13,
                             fill13);

    filler_kernels::fill_mat(pd1, pn2, pd3, pd1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bd3, scratch.bd1,
                             scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn, mat23,
                             fill23);
}

/**
 * Add one marker's contribution to the perpendicular pressure coupling at known spans, as in
 * particle_to_mat_kernels.m_v_fill_v1_pressure: block (mu, nu), mu <= nu, of the V1 -> V1 matrix (a, b) gets
 * fill_mu_nu v_a v_b and component mu of the V1 vector a gets fill_mu v_a, for (a, b) = (x, x), (x, y), (y, y).
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat11_11 (mu=1, nu=1)-block of the matrix of v_x v_x; written atomically.
 * @param mat12_11 (mu=1, nu=2)-block of the matrix of v_x v_x; written atomically.
 * @param mat13_11 (mu=1, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat22_11 (mu=2, nu=2)-block of the matrix of v_x v_x; written atomically.
 * @param mat23_11 (mu=2, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat33_11 (mu=3, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat11_12 (mu=1, nu=1)-block of the matrix of v_x v_y; written atomically.
 * @param mat12_12 (mu=1, nu=2)-block of the matrix of v_x v_y; written atomically.
 * @param mat13_12 (mu=1, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat22_12 (mu=2, nu=2)-block of the matrix of v_x v_y; written atomically.
 * @param mat23_12 (mu=2, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat33_12 (mu=3, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat11_22 (mu=1, nu=1)-block of the matrix of v_y v_y; written atomically.
 * @param mat12_22 (mu=1, nu=2)-block of the matrix of v_y v_y; written atomically.
 * @param mat13_22 (mu=1, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param mat22_22 (mu=2, nu=2)-block of the matrix of v_y v_y; written atomically.
 * @param mat23_22 (mu=2, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param mat33_22 (mu=3, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param fill11 Number multiplied by the basis functions and added to mat11_*.
 * @param fill12 Number multiplied by the basis functions and added to mat12_*.
 * @param fill13 Number multiplied by the basis functions and added to mat13_*.
 * @param fill22 Number multiplied by the basis functions and added to mat22_*.
 * @param fill23 Number multiplied by the basis functions and added to mat23_*.
 * @param fill33 Number multiplied by the basis functions and added to mat33_*.
 * @param vec1_1 mu=1 component of the vector of v_x; written atomically.
 * @param vec2_1 mu=2 component of the vector of v_x; written atomically.
 * @param vec3_1 mu=3 component of the vector of v_x; written atomically.
 * @param vec1_2 mu=1 component of the vector of v_y; written atomically.
 * @param vec2_2 mu=2 component of the vector of v_y; written atomically.
 * @param vec3_2 mu=3 component of the vector of v_y; written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1_*.
 * @param fill2 Number multiplied by the basis functions and added to vec2_*.
 * @param fill3 Number multiplied by the basis functions and added to vec3_*.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 */
__device__ inline void m_v_fill_v1_pressure(
    const DerhamArgs& args_derham, const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2, int span3,
    Array6D<double> mat11_11, Array6D<double> mat12_11, Array6D<double> mat13_11, Array6D<double> mat22_11,
    Array6D<double> mat23_11, Array6D<double> mat33_11, Array6D<double> mat11_12, Array6D<double> mat12_12,
    Array6D<double> mat13_12, Array6D<double> mat22_12, Array6D<double> mat23_12, Array6D<double> mat33_12,
    Array6D<double> mat11_22, Array6D<double> mat12_22, Array6D<double> mat13_22, Array6D<double> mat22_22,
    Array6D<double> mat23_22, Array6D<double> mat33_22, double fill11, double fill12, double fill13, double fill22,
    double fill23, double fill33, Array3D<double> vec1_1, Array3D<double> vec2_1, Array3D<double> vec3_1,
    Array3D<double> vec1_2, Array3D<double> vec2_2, Array3D<double> vec3_2, double fill1, double fill2, double fill3,
    double vx, double vy) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat_vec_pressure(pd1, pn2, pn3, pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3,
                                          scratch.bd1, scratch.bn2, scratch.bn3, span1, span2, span3,
                                          args_derham.starts, args_derham.pn, mat11_11, mat11_12, mat11_22, fill11,
                                          vec1_1, vec1_2, fill1, vx, vy);

    filler_kernels::fill_mat_vec_pressure(pn1, pd2, pn3, pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3,
                                          scratch.bn1, scratch.bd2, scratch.bn3, span1, span2, span3,
                                          args_derham.starts, args_derham.pn, mat22_11, mat22_12, mat22_22, fill22,
                                          vec2_1, vec2_2, fill2, vx, vy);

    filler_kernels::fill_mat_vec_pressure(pn1, pn2, pd3, pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3,
                                          scratch.bn1, scratch.bn2, scratch.bd3, span1, span2, span3,
                                          args_derham.starts, args_derham.pn, mat33_11, mat33_12, mat33_22, fill33,
                                          vec3_1, vec3_2, fill3, vx, vy);

    filler_kernels::fill_mat_pressure(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                                      scratch.bd2, scratch.bn3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                      mat12_11, mat12_12, mat12_22, fill12, vx, vy);

    filler_kernels::fill_mat_pressure(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bn1,
                                      scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                      mat13_11, mat13_12, mat13_22, fill13, vx, vy);

    filler_kernels::fill_mat_pressure(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1,
                                      scratch.bn2, scratch.bd3, span1, span2, span3, args_derham.starts, args_derham.pn,
                                      mat23_11, mat23_12, mat23_22, fill23, vx, vy);
}

/**
 * Add one marker's contribution to the full pressure coupling at known spans, as in
 * particle_to_mat_kernels.m_v_fill_v1_pressure_full: block (mu, nu), mu <= nu, of the V1 -> V1 matrix (a, b), a <= b,
 * gets fill_mu_nu v_a v_b and component mu of the V1 vector a gets fill_mu v_a (a, b = x, y, z).
 *
 * @param args_derham Spline degrees (1 to 8) and start indices; pn is also the padding of the blocks.
 * @param scratch N- and D-spline values from get_spans; pyccel reads them from args_derham.bn1, ..., bd3.
 * @param span1 Knot span index along the first axis (from get_spans).
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param mat11_11 (mu=1, nu=1)-block of the matrix of v_x v_x; written atomically.
 * @param mat12_11 (mu=1, nu=2)-block of the matrix of v_x v_x; written atomically.
 * @param mat13_11 (mu=1, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat22_11 (mu=2, nu=2)-block of the matrix of v_x v_x; written atomically.
 * @param mat23_11 (mu=2, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat33_11 (mu=3, nu=3)-block of the matrix of v_x v_x; written atomically.
 * @param mat11_12 (mu=1, nu=1)-block of the matrix of v_x v_y; written atomically.
 * @param mat12_12 (mu=1, nu=2)-block of the matrix of v_x v_y; written atomically.
 * @param mat13_12 (mu=1, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat22_12 (mu=2, nu=2)-block of the matrix of v_x v_y; written atomically.
 * @param mat23_12 (mu=2, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat33_12 (mu=3, nu=3)-block of the matrix of v_x v_y; written atomically.
 * @param mat11_13 (mu=1, nu=1)-block of the matrix of v_x v_z; written atomically.
 * @param mat12_13 (mu=1, nu=2)-block of the matrix of v_x v_z; written atomically.
 * @param mat13_13 (mu=1, nu=3)-block of the matrix of v_x v_z; written atomically.
 * @param mat22_13 (mu=2, nu=2)-block of the matrix of v_x v_z; written atomically.
 * @param mat23_13 (mu=2, nu=3)-block of the matrix of v_x v_z; written atomically.
 * @param mat33_13 (mu=3, nu=3)-block of the matrix of v_x v_z; written atomically.
 * @param mat11_22 (mu=1, nu=1)-block of the matrix of v_y v_y; written atomically.
 * @param mat12_22 (mu=1, nu=2)-block of the matrix of v_y v_y; written atomically.
 * @param mat13_22 (mu=1, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param mat22_22 (mu=2, nu=2)-block of the matrix of v_y v_y; written atomically.
 * @param mat23_22 (mu=2, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param mat33_22 (mu=3, nu=3)-block of the matrix of v_y v_y; written atomically.
 * @param mat11_23 (mu=1, nu=1)-block of the matrix of v_y v_z; written atomically.
 * @param mat12_23 (mu=1, nu=2)-block of the matrix of v_y v_z; written atomically.
 * @param mat13_23 (mu=1, nu=3)-block of the matrix of v_y v_z; written atomically.
 * @param mat22_23 (mu=2, nu=2)-block of the matrix of v_y v_z; written atomically.
 * @param mat23_23 (mu=2, nu=3)-block of the matrix of v_y v_z; written atomically.
 * @param mat33_23 (mu=3, nu=3)-block of the matrix of v_y v_z; written atomically.
 * @param mat11_33 (mu=1, nu=1)-block of the matrix of v_z v_z; written atomically.
 * @param mat12_33 (mu=1, nu=2)-block of the matrix of v_z v_z; written atomically.
 * @param mat13_33 (mu=1, nu=3)-block of the matrix of v_z v_z; written atomically.
 * @param mat22_33 (mu=2, nu=2)-block of the matrix of v_z v_z; written atomically.
 * @param mat23_33 (mu=2, nu=3)-block of the matrix of v_z v_z; written atomically.
 * @param mat33_33 (mu=3, nu=3)-block of the matrix of v_z v_z; written atomically.
 * @param fill11 Number multiplied by the basis functions and added to mat11_*.
 * @param fill12 Number multiplied by the basis functions and added to mat12_*.
 * @param fill13 Number multiplied by the basis functions and added to mat13_*.
 * @param fill22 Number multiplied by the basis functions and added to mat22_*.
 * @param fill23 Number multiplied by the basis functions and added to mat23_*.
 * @param fill33 Number multiplied by the basis functions and added to mat33_*.
 * @param vec1_1 mu=1 component of the vector of v_x; written atomically.
 * @param vec2_1 mu=2 component of the vector of v_x; written atomically.
 * @param vec3_1 mu=3 component of the vector of v_x; written atomically.
 * @param vec1_2 mu=1 component of the vector of v_y; written atomically.
 * @param vec2_2 mu=2 component of the vector of v_y; written atomically.
 * @param vec3_2 mu=3 component of the vector of v_y; written atomically.
 * @param vec1_3 mu=1 component of the vector of v_z; written atomically.
 * @param vec2_3 mu=2 component of the vector of v_z; written atomically.
 * @param vec3_3 mu=3 component of the vector of v_z; written atomically.
 * @param fill1 Number multiplied by the basis functions and added to vec1_*.
 * @param fill2 Number multiplied by the basis functions and added to vec2_*.
 * @param fill3 Number multiplied by the basis functions and added to vec3_*.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 * @param vz Third velocity component.
 */
__device__ inline void m_v_fill_v1_pressure_full(
    const DerhamArgs& args_derham, const evaluation_kernels_3d::SplineScratch& scratch, int span1, int span2, int span3,
    Array6D<double> mat11_11, Array6D<double> mat12_11, Array6D<double> mat13_11, Array6D<double> mat22_11,
    Array6D<double> mat23_11, Array6D<double> mat33_11, Array6D<double> mat11_12, Array6D<double> mat12_12,
    Array6D<double> mat13_12, Array6D<double> mat22_12, Array6D<double> mat23_12, Array6D<double> mat33_12,
    Array6D<double> mat11_13, Array6D<double> mat12_13, Array6D<double> mat13_13, Array6D<double> mat22_13,
    Array6D<double> mat23_13, Array6D<double> mat33_13, Array6D<double> mat11_22, Array6D<double> mat12_22,
    Array6D<double> mat13_22, Array6D<double> mat22_22, Array6D<double> mat23_22, Array6D<double> mat33_22,
    Array6D<double> mat11_23, Array6D<double> mat12_23, Array6D<double> mat13_23, Array6D<double> mat22_23,
    Array6D<double> mat23_23, Array6D<double> mat33_23, Array6D<double> mat11_33, Array6D<double> mat12_33,
    Array6D<double> mat13_33, Array6D<double> mat22_33, Array6D<double> mat23_33, Array6D<double> mat33_33,
    double fill11, double fill12, double fill13, double fill22, double fill23, double fill33, Array3D<double> vec1_1,
    Array3D<double> vec2_1, Array3D<double> vec3_1, Array3D<double> vec1_2, Array3D<double> vec2_2,
    Array3D<double> vec3_2, Array3D<double> vec1_3, Array3D<double> vec2_3, Array3D<double> vec3_3, double fill1,
    double fill2, double fill3, double vx, double vy, double vz) {
    // degrees of the basis functions : B-splines (pn) and D-splines (pd)
    int pn1 = args_derham.pn[0];
    int pn2 = args_derham.pn[1];
    int pn3 = args_derham.pn[2];

    int pd1 = pn1 - 1;
    int pd2 = pn2 - 1;
    int pd3 = pn3 - 1;

    // fill matrix entries
    filler_kernels::fill_mat_vec_pressure_full(
        pd1, pn2, pn3, pd1, pn2, pn3, scratch.bd1, scratch.bn2, scratch.bn3, scratch.bd1, scratch.bn2, scratch.bn3,
        span1, span2, span3, args_derham.starts, args_derham.pn, mat11_11, mat11_12, mat11_13, mat11_22, mat11_23,
        mat11_33, fill11, vec1_1, vec1_2, vec1_3, fill1, vx, vy, vz);

    filler_kernels::fill_mat_vec_pressure_full(
        pn1, pd2, pn3, pn1, pd2, pn3, scratch.bn1, scratch.bd2, scratch.bn3, scratch.bn1, scratch.bd2, scratch.bn3,
        span1, span2, span3, args_derham.starts, args_derham.pn, mat22_11, mat22_12, mat22_13, mat22_22, mat22_23,
        mat22_33, fill22, vec2_1, vec2_2, vec2_3, fill2, vx, vy, vz);

    filler_kernels::fill_mat_vec_pressure_full(
        pn1, pn2, pd3, pn1, pn2, pd3, scratch.bn1, scratch.bn2, scratch.bd3, scratch.bn1, scratch.bn2, scratch.bd3,
        span1, span2, span3, args_derham.starts, args_derham.pn, mat33_11, mat33_12, mat33_13, mat33_22, mat33_23,
        mat33_33, fill33, vec3_1, vec3_2, vec3_3, fill3, vx, vy, vz);

    filler_kernels::fill_mat_pressure_full(pd1, pn2, pn3, pn1, pd2, pn3, scratch.bd1, scratch.bn2, scratch.bn3,
                                           scratch.bn1, scratch.bd2, scratch.bn3, span1, span2, span3,
                                           args_derham.starts, args_derham.pn, mat12_11, mat12_12, mat12_13, mat12_22,
                                           mat12_23, mat12_33, fill12, vx, vy, vz);

    filler_kernels::fill_mat_pressure_full(pd1, pn2, pn3, pn1, pn2, pd3, scratch.bd1, scratch.bn2, scratch.bn3,
                                           scratch.bn1, scratch.bn2, scratch.bd3, span1, span2, span3,
                                           args_derham.starts, args_derham.pn, mat13_11, mat13_12, mat13_13, mat13_22,
                                           mat13_23, mat13_33, fill13, vx, vy, vz);

    filler_kernels::fill_mat_pressure_full(pn1, pd2, pn3, pn1, pn2, pd3, scratch.bn1, scratch.bd2, scratch.bn3,
                                           scratch.bn1, scratch.bn2, scratch.bd3, span1, span2, span3,
                                           args_derham.starts, args_derham.pn, mat23_11, mat23_12, mat23_13, mat23_22,
                                           mat23_23, mat23_33, fill23, vx, vy, vz);
}
}  // namespace struphy_cuda::particle_to_mat_kernels
