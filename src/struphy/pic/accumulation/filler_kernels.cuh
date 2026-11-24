// Device versions of the fillers in filler_kernels.py: add the contribution of one marker to a stencil vector
// or to a block of a stencil matrix. Many markers write to the same entries, so every addition is atomic.
#pragma once
#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
namespace struphy_cuda::filler_kernels {
/**
 * Add basis functions times filling to a stencil vector, as in filler_kernels.fill_vec.
 *
 * @param pi1 Spline degree along the first axis.
 * @param pi2 Spline degree along the second axis.
 * @param pi3 Spline degree along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param vec Stencil vector data (the _data of a StencilVector), any strides; written atomically.
 * @param filling Number multiplied by the basis functions.
 *
 * Unlike the pyccel loop, many threads add to the same entries concurrently, hence cunumpy_atomic_add;
 * the summation order (and the last bits of the result) depends on the thread schedule.
 */
__device__ inline void fill_vec(int pi1, int pi2, int pi3, const double* bi1, const double* bi2, const double* bi3,
                                int span1, int span2, int span3, const long long* starts, Array3D<double> vec,
                                double filling) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1] * filling;
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];
                cunumpy_atomic_add(&vec(i1, i2, i3), b3);
            }
        }
    }
}

/**
 * Add basis functions times filling to a block of a stencil matrix, as in filler_kernels.fill_mat.
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat Stencil matrix data (the _data of a StencilMatrix, shape (n1, n2, n3, 2 * pads + 1)), any strides;
 *            written atomically.
 * @param filling Number multiplied by the basis functions.
 *
 * The summation order of concurrent threads (and the last bits of the result) depends on the thread schedule.
 */
__device__ inline void fill_mat(int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1,
                                const double* bi2, const double* bi3, const double* bj1, const double* bj2,
                                const double* bj3, int span1, int span2, int span3, const long long* starts,
                                const long long* pads, Array6D<double> mat, double filling) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1] * filling;
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1];
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat(i1, i2, i3, j1, j2, j3), b6);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Add basis functions times fillings to a block of a stencil matrix and to a stencil vector, as in
 * filler_kernels.fill_mat_vec.
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat Stencil matrix data (the _data of a StencilMatrix), any strides; written atomically.
 * @param filling_mat Number multiplied by the basis functions and added to mat.
 * @param vec Stencil vector data of the codomain (the _data of a StencilVector), any strides; written atomically.
 * @param filling_vec Number multiplied by the basis functions of the codomain and added to vec.
 */
__device__ inline void fill_mat_vec(int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1,
                                    const double* bi2, const double* bi3, const double* bj1, const double* bj2,
                                    const double* bj3, int span1, int span2, int span3, const long long* starts,
                                    const long long* pads, Array6D<double> mat, double filling_mat,
                                    Array3D<double> vec, double filling_vec) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                cunumpy_atomic_add(&vec(i1, i2, i3), b3 * filling_vec);

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat(i1, i2, i3, j1, j2, j3), b6);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Add basis functions times filling times v_a v_b (a, b = x, y) to three blocks of stencil matrices, as in
 * filler_kernels.fill_mat_pressure (the perpendicular pressure coupling of LinearMHDVlasovPC).
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat_11 Block data that gets v_x v_x times the contribution; written atomically.
 * @param mat_12 Block data that gets v_x v_y times the contribution; written atomically.
 * @param mat_22 Block data that gets v_y v_y times the contribution; written atomically.
 * @param filling_mat Number multiplied by the basis functions.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 */
__device__ inline void fill_mat_pressure(int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1,
                                         const double* bi2, const double* bi3, const double* bj1, const double* bj2,
                                         const double* bj3, int span1, int span2, int span3, const long long* starts,
                                         const long long* pads, Array6D<double> mat_11, Array6D<double> mat_12,
                                         Array6D<double> mat_22, double filling_mat, double vx, double vy) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat_11(i1, i2, i3, j1, j2, j3), b6 * vx * vx);
                            cunumpy_atomic_add(&mat_12(i1, i2, i3, j1, j2, j3), b6 * vx * vy);
                            cunumpy_atomic_add(&mat_22(i1, i2, i3, j1, j2, j3), b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Add basis functions times fillings times velocity components to three blocks of stencil matrices (v_a v_b, a, b =
 * x, y) and to two stencil vectors (v_a), as in filler_kernels.fill_mat_vec_pressure.
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat_11 Block data that gets v_x v_x times the contribution; written atomically.
 * @param mat_12 Block data that gets v_x v_y times the contribution; written atomically.
 * @param mat_22 Block data that gets v_y v_y times the contribution; written atomically.
 * @param filling_mat Number multiplied by the basis functions and added to the matrices.
 * @param vec_1 Vector data that gets v_x times the contribution; written atomically.
 * @param vec_2 Vector data that gets v_y times the contribution; written atomically.
 * @param filling_vec Number multiplied by the basis functions of the codomain and added to the vectors.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 */
__device__ inline void fill_mat_vec_pressure(int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1,
                                             const double* bi2, const double* bi3, const double* bj1, const double* bj2,
                                             const double* bj3, int span1, int span2, int span3,
                                             const long long* starts, const long long* pads, Array6D<double> mat_11,
                                             Array6D<double> mat_12, Array6D<double> mat_22, double filling_mat,
                                             Array3D<double> vec_1, Array3D<double> vec_2, double filling_vec,
                                             double vx, double vy) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                cunumpy_atomic_add(&vec_1(i1, i2, i3), b3 * filling_vec * vx);
                cunumpy_atomic_add(&vec_2(i1, i2, i3), b3 * filling_vec * vy);

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat_11(i1, i2, i3, j1, j2, j3), b6 * vx * vx);
                            cunumpy_atomic_add(&mat_12(i1, i2, i3, j1, j2, j3), b6 * vx * vy);
                            cunumpy_atomic_add(&mat_22(i1, i2, i3, j1, j2, j3), b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Add basis functions times filling times v_a v_b (a <= b, a, b = x, y, z) to six blocks of stencil matrices, as in
 * filler_kernels.fill_mat_pressure_full (the full pressure coupling of LinearMHDVlasovPC).
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat_11 Block data that gets v_x v_x times the contribution; written atomically.
 * @param mat_12 Block data that gets v_x v_y times the contribution; written atomically.
 * @param mat_13 Block data that gets v_x v_z times the contribution; written atomically.
 * @param mat_22 Block data that gets v_y v_y times the contribution; written atomically.
 * @param mat_23 Block data that gets v_y v_z times the contribution; written atomically.
 * @param mat_33 Block data that gets v_z v_z times the contribution; written atomically.
 * @param filling_mat Number multiplied by the basis functions.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 * @param vz Third velocity component.
 */
__device__ inline void fill_mat_pressure_full(int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1,
                                              const double* bi2, const double* bi3, const double* bj1,
                                              const double* bj2, const double* bj3, int span1, int span2, int span3,
                                              const long long* starts, const long long* pads, Array6D<double> mat_11,
                                              Array6D<double> mat_12, Array6D<double> mat_13, Array6D<double> mat_22,
                                              Array6D<double> mat_23, Array6D<double> mat_33, double filling_mat,
                                              double vx, double vy, double vz) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat_11(i1, i2, i3, j1, j2, j3), b6 * vx * vx);
                            cunumpy_atomic_add(&mat_12(i1, i2, i3, j1, j2, j3), b6 * vx * vy);
                            cunumpy_atomic_add(&mat_13(i1, i2, i3, j1, j2, j3), b6 * vx * vz);
                            cunumpy_atomic_add(&mat_22(i1, i2, i3, j1, j2, j3), b6 * vy * vy);
                            cunumpy_atomic_add(&mat_23(i1, i2, i3, j1, j2, j3), b6 * vy * vz);
                            cunumpy_atomic_add(&mat_33(i1, i2, i3, j1, j2, j3), b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Add basis functions times fillings times velocity components to six blocks of stencil matrices (v_a v_b, a <= b,
 * a, b = x, y, z) and to three stencil vectors (v_a), as in filler_kernels.fill_mat_vec_pressure_full.
 *
 * @param pi1 Spline degree of the codomain (row indices) along the first axis.
 * @param pi2 Spline degree of the codomain along the second axis.
 * @param pi3 Spline degree of the codomain along the third axis.
 * @param pj1 Spline degree of the domain (column indices) along the first axis.
 * @param pj2 Spline degree of the domain along the second axis.
 * @param pj3 Spline degree of the domain along the third axis.
 * @param bi1 The pi1 + 1 non-vanishing N- or D-spline values of the codomain along the first axis.
 * @param bi2 The pi2 + 1 non-vanishing spline values of the codomain along the second axis.
 * @param bi3 The pi3 + 1 non-vanishing spline values of the codomain along the third axis.
 * @param bj1 The pj1 + 1 non-vanishing spline values of the domain along the first axis.
 * @param bj2 The pj2 + 1 non-vanishing spline values of the domain along the second axis.
 * @param bj3 The pj3 + 1 non-vanishing spline values of the domain along the third axis.
 * @param span1 Knot span index along the first axis.
 * @param span2 Knot span index along the second axis.
 * @param span3 Knot span index along the third axis.
 * @param starts Start indices of the codomain (three entries).
 * @param pads Paddings of the codomain (three entries).
 * @param mat_11 Block data that gets v_x v_x times the contribution; written atomically.
 * @param mat_12 Block data that gets v_x v_y times the contribution; written atomically.
 * @param mat_13 Block data that gets v_x v_z times the contribution; written atomically.
 * @param mat_22 Block data that gets v_y v_y times the contribution; written atomically.
 * @param mat_23 Block data that gets v_y v_z times the contribution; written atomically.
 * @param mat_33 Block data that gets v_z v_z times the contribution; written atomically.
 * @param filling_mat Number multiplied by the basis functions and added to the matrices.
 * @param vec_1 Vector data that gets v_x times the contribution; written atomically.
 * @param vec_2 Vector data that gets v_y times the contribution; written atomically.
 * @param vec_3 Vector data that gets v_z times the contribution; written atomically.
 * @param filling_vec Number multiplied by the basis functions of the codomain and added to the vectors.
 * @param vx First velocity component.
 * @param vy Second velocity component.
 * @param vz Third velocity component.
 */
__device__ inline void fill_mat_vec_pressure_full(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3, const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3, int span1, int span2, int span3, const long long* starts,
    const long long* pads, Array6D<double> mat_11, Array6D<double> mat_12, Array6D<double> mat_13,
    Array6D<double> mat_22, Array6D<double> mat_23, Array6D<double> mat_33, double filling_mat, Array3D<double> vec_1,
    Array3D<double> vec_2, Array3D<double> vec_3, double filling_vec, double vx, double vy, double vz) {
    for (int il1 = 0; il1 <= pi1; ++il1) {
        long long i1 = span1 + il1 - starts[0];
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; ++il2) {
            long long i2 = span2 + il2 - starts[1];
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; ++il3) {
                long long i3 = span3 + il3 - starts[2];
                double b3 = b2 * bi3[il3];

                cunumpy_atomic_add(&vec_1(i1, i2, i3), b3 * filling_vec * vx);
                cunumpy_atomic_add(&vec_2(i1, i2, i3), b3 * filling_vec * vy);
                cunumpy_atomic_add(&vec_3(i1, i2, i3), b3 * filling_vec * vz);

                for (int jl1 = 0; jl1 <= pj1; ++jl1) {
                    long long j1 = pads[0] + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; ++jl2) {
                        long long j2 = pads[1] + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; ++jl3) {
                            long long j3 = pads[2] + jl3 - il3;
                            double b6 = b5 * bj3[jl3];

                            cunumpy_atomic_add(&mat_11(i1, i2, i3, j1, j2, j3), b6 * vx * vx);
                            cunumpy_atomic_add(&mat_12(i1, i2, i3, j1, j2, j3), b6 * vx * vy);
                            cunumpy_atomic_add(&mat_13(i1, i2, i3, j1, j2, j3), b6 * vx * vz);
                            cunumpy_atomic_add(&mat_22(i1, i2, i3, j1, j2, j3), b6 * vy * vy);
                            cunumpy_atomic_add(&mat_23(i1, i2, i3, j1, j2, j3), b6 * vy * vz);
                            cunumpy_atomic_add(&mat_33(i1, i2, i3, j1, j2, j3), b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}
}  // namespace struphy_cuda::filler_kernels
