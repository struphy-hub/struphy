#pragma once
// Device versions of the spline mappings in geometry/spline_mappings_kernels.py (kind_map 0, 1 and 2).
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/bsplines/bsplines_kernels.cuh"
#include "struphy/bsplines/evaluation_kernels_2d.cuh"
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/domains/constants_cuda.cuh"
namespace struphy_cuda {
/**
 * Row i of a 2d array view, as the Pyccel slice ind[i, :] (no copy).
 *
 * CUDA-only helper for the slices tmp1 = ind1[span1 - degree[0], :] of the spline mappings.
 *
 * @param ind 2d array view, e.g. DomainArgs.ind1 (shape (number of cells, degree + 1)).
 * @param i Row index.
 * @return A 1d view of row i with the strides of ind.
 */
__device__ inline Array1D<long long> spline_index_row(const Array2D<long long>& ind, long long i) {
    return Array1D<long long>{ind.data + i * ind.strides[0], {ind.shape[1]}, {ind.strides[1]}};
}

/**
 * The plane c[:, :, 0] of a 3d array view, as the Pyccel slices cx = args.cx[:, :, 0] (no copy).
 *
 * CUDA-only helper for the 2d spline mappings, whose control points are stored with a third axis of length one.
 *
 * @param c 3d array view, e.g. DomainArgs.cx.
 * @return A 2d view of c[:, :, 0] with the strides of c.
 */
__device__ inline Array2D<double> first_plane(const Array3D<double>& c) {
    return Array2D<double>{c.data, {c.shape[0], c.shape[1]}, {c.strides[0], c.strides[1]}};
}

/**
 * Evaluate a 3d spline mapping, as in spline_mappings_kernels.spline_3d (kind_map 0).
 *
 * F_n = sum_ijk c^n_ijk N_i(eta1) N_j(eta2) N_k(eta3) for n = x, y, z, with the control points
 * args.cx, args.cy, args.cz.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param degree Spline degrees of the mapping (three entries, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param ind3 Same along the third axis.
 * @param args Knot sequences t1, t2, t3 (contiguous) and control points cx, cy, cz.
 * @param f_out Output buffer for the three physical coordinates.
 *
 * The Pyccel stack arrays b1, b2, b3 are fixed-size, thread-local arrays here.
 */
__device__ inline void spline_3d(double eta1, double eta2, double eta3, const long long* degree,
                                 const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                 const Array2D<long long>& ind3, const DomainArgs& args, double* f_out) {
    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);
    int span3 = find_span(args.t3.data, args.t3.shape[0], degree[2], eta3);

    // degree + 1 non-zero mapping splines
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1], b3[MAX_SPLINE_DEGREE + 1];

    b_splines_slim(args.t1.data, degree[0], eta1, span1, b1);
    b_splines_slim(args.t2.data, degree[1], eta2, span2, b2);
    b_splines_slim(args.t3.data, degree[2], eta3, span3, b3);

    // Evaluate spline mapping
    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);
    Array1D<long long> tmp3 = spline_index_row(ind3, span3 - degree[2]);

    f_out[0] = evaluation_kernel_3d(degree[0], degree[1], degree[2], b1, b2, b3, tmp1, tmp2, tmp3, args.cx);
    f_out[1] = evaluation_kernel_3d(degree[0], degree[1], degree[2], b1, b2, b3, tmp1, tmp2, tmp3, args.cy);
    f_out[2] = evaluation_kernel_3d(degree[0], degree[1], degree[2], b1, b2, b3, tmp1, tmp2, tmp3, args.cz);
}

/**
 * Jacobian matrix of spline_3d, as in spline_mappings_kernels.spline_3d_df (kind_map 0).
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param degree Spline degrees of the mapping (three entries, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param ind3 Same along the third axis.
 * @param args Knot sequences t1, t2, t3 (contiguous) and control points cx, cy, cz.
 * @param df_out Output 3x3 matrix stored as nine row-major entries, df_out[3 * i + j] = dF_i / deta_j.
 *
 * The Pyccel stack arrays b1, ..., der3 are fixed-size, thread-local arrays here.
 */
__device__ inline void spline_3d_df(double eta1, double eta2, double eta3, const long long* degree,
                                    const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                    const Array2D<long long>& ind3, const DomainArgs& args, double* df_out) {
    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);
    int span3 = find_span(args.t3.data, args.t3.shape[0], degree[2], eta3);

    // non-zero splines of mapping, and derivatives
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1], b3[MAX_SPLINE_DEGREE + 1];
    double der1[MAX_SPLINE_DEGREE + 1], der2[MAX_SPLINE_DEGREE + 1], der3[MAX_SPLINE_DEGREE + 1];

    b_der_splines_slim(args.t1.data, degree[0], eta1, span1, b1, der1);
    b_der_splines_slim(args.t2.data, degree[1], eta2, span2, b2, der2);
    b_der_splines_slim(args.t3.data, degree[2], eta3, span3, b3, der3);

    // Evaluation of Jacobian
    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);
    Array1D<long long> tmp3 = spline_index_row(ind3, span3 - degree[2]);

    const Array3D<double>* c[3] = {&args.cx, &args.cy, &args.cz};
    for (int n = 0; n < 3; ++n) {
        df_out[3 * n + 0] = evaluation_kernel_3d(degree[0], degree[1], degree[2], der1, b2, b3, tmp1, tmp2, tmp3, *c[n]);
        df_out[3 * n + 1] = evaluation_kernel_3d(degree[0], degree[1], degree[2], b1, der2, b3, tmp1, tmp2, tmp3, *c[n]);
        df_out[3 * n + 2] = evaluation_kernel_3d(degree[0], degree[1], degree[2], b1, b2, der3, tmp1, tmp2, tmp3, *c[n]);
    }
}

/**
 * Evaluate a 2d spline mapping of a straight cylinder, as in spline_mappings_kernels.spline_2d_straight (kind_map 1).
 *
 * F_x = sum_ij c^x_ij N_i(eta1) N_j(eta2), F_y = sum_ij c^y_ij N_i(eta1) N_j(eta2), F_z = lz * eta3, with the
 * control points args.cx[:, :, 0], args.cy[:, :, 0].
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param degree Spline degrees of the mapping (first two entries used, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param args Knot sequences t1, t2 (contiguous) and control points cx, cy (third axis of length one).
 * @param lz Length of the cylinder (args.params[0] in Pyccel's evaluation_kernels.f).
 * @param f_out Output buffer for the three physical coordinates.
 *
 * At the pole (eta1 == 0 with equal control points c[0, 0] == c[0, 1]) the value is the control point itself,
 * exactly as in Pyccel.
 */
__device__ inline void spline_2d_straight(double eta1, double eta2, double eta3, const long long* degree,
                                          const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                          const DomainArgs& args, double lz, double* f_out) {
    Array2D<double> cx = first_plane(args.cx);
    Array2D<double> cy = first_plane(args.cy);

    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);

    // degree + 1 non-zero mapping splines
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1];

    b_splines_slim(args.t1.data, degree[0], eta1, span1, b1);
    b_splines_slim(args.t2.data, degree[1], eta2, span2, b2);

    // Evaluate mapping
    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);

    f_out[0] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cx);
    f_out[1] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cy);
    f_out[2] = lz * eta3;

    if (eta1 == 0. && cx(0, 0) == cx(0, 1)) f_out[0] = cx(0, 0);

    if (eta1 == 0. && cy(0, 0) == cy(0, 1)) f_out[1] = cy(0, 0);
}

/**
 * Jacobian matrix of spline_2d_straight, as in spline_mappings_kernels.spline_2d_straight_df (kind_map 1).
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param degree Spline degrees of the mapping (first two entries used, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param args Knot sequences t1, t2 (contiguous) and control points cx, cy (third axis of length one).
 * @param lz Length of the cylinder.
 * @param df_out Output 3x3 matrix stored as nine row-major entries, df_out[3 * i + j] = dF_i / deta_j.
 *
 * Like Pyccel, takes no eta3 (the Jacobian does not depend on it).
 */
__device__ inline void spline_2d_straight_df(double eta1, double eta2, const long long* degree,
                                             const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                             const DomainArgs& args, double lz, double* df_out) {
    Array2D<double> cx = first_plane(args.cx);
    Array2D<double> cy = first_plane(args.cy);

    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);

    // non-zero splines of mapping, and derivatives
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1];
    double der1[MAX_SPLINE_DEGREE + 1], der2[MAX_SPLINE_DEGREE + 1];

    b_der_splines_slim(args.t1.data, degree[0], eta1, span1, b1, der1);
    b_der_splines_slim(args.t2.data, degree[1], eta2, span2, b2, der2);

    // Evaluation of Jacobian
    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);

    df_out[0] = evaluation_kernel_2d(degree[0], degree[1], der1, b2, tmp1, tmp2, cx);
    df_out[1] = evaluation_kernel_2d(degree[0], degree[1], b1, der2, tmp1, tmp2, cx);
    df_out[2] = 0.;
    df_out[3] = evaluation_kernel_2d(degree[0], degree[1], der1, b2, tmp1, tmp2, cy);
    df_out[4] = evaluation_kernel_2d(degree[0], degree[1], b1, der2, tmp1, tmp2, cy);
    df_out[5] = 0.;
    df_out[6] = 0.;
    df_out[7] = 0.;
    df_out[8] = lz;

    if (eta1 == 0. && cx(0, 0) == cx(0, 1)) df_out[1] = 0.;

    if (eta1 == 0. && cy(0, 0) == cy(0, 1)) df_out[4] = 0.;
}

/**
 * Evaluate a 2d spline mapping of a torus, as in spline_mappings_kernels.spline_2d_torus (kind_map 2).
 *
 * With S_R, S_z the splines with control points args.cx[:, :, 0], args.cy[:, :, 0] in the poloidal plane:
 * F_x = S_R cos(2 pi eta3 / tor_period), F_y = -S_R sin(2 pi eta3 / tor_period), F_z = S_z.
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third (toroidal) axis.
 * @param degree Spline degrees of the mapping (first two entries used, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param args Knot sequences t1, t2 (contiguous) and control points cx, cy (third axis of length one).
 * @param tor_period Toroidal periodicity (args.params[0] in Pyccel's evaluation_kernels.f).
 * @param f_out Output buffer for the three physical coordinates.
 */
__device__ inline void spline_2d_torus(double eta1, double eta2, double eta3, const long long* degree,
                                       const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                       const DomainArgs& args, double tor_period, double* f_out) {
    Array2D<double> cx = first_plane(args.cx);
    Array2D<double> cy = first_plane(args.cy);

    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);

    // degree + 1 non-zero mapping splines
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1];

    b_splines_slim(args.t1.data, degree[0], eta1, span1, b1);
    b_splines_slim(args.t2.data, degree[1], eta2, span2, b2);

    // Evaluate mapping
    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);

    f_out[0] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cx) * cos(2 * pi * eta3 / tor_period);
    f_out[1] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cx) * (-1) *
               sin(2 * pi * eta3 / tor_period);
    f_out[2] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cy);

    if (eta1 == 0. && cx(0, 0) == cx(0, 1)) {
        f_out[0] = cx(0, 0) * cos(2 * pi * eta3 / tor_period);
        f_out[1] = cx(0, 0) * (-1) * sin(2 * pi * eta3 / tor_period);
    }

    if (eta1 == 0. && cy(0, 0) == cy(0, 1)) f_out[2] = cy(0, 0);
}

/**
 * Jacobian matrix of spline_2d_torus, as in spline_mappings_kernels.spline_2d_torus_df (kind_map 2).
 *
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third (toroidal) axis.
 * @param degree Spline degrees of the mapping (first two entries used, 1 to MAX_SPLINE_DEGREE).
 * @param ind1 Indices of the non-vanishing splines per cell along the first axis, shape (cells, degree[0] + 1).
 * @param ind2 Same along the second axis.
 * @param args Knot sequences t1, t2 (contiguous) and control points cx, cy (third axis of length one).
 * @param tor_period Toroidal periodicity.
 * @param df_out Output 3x3 matrix stored as nine row-major entries, df_out[3 * i + j] = dF_i / deta_j.
 */
__device__ inline void spline_2d_torus_df(double eta1, double eta2, double eta3, const long long* degree,
                                          const Array2D<long long>& ind1, const Array2D<long long>& ind2,
                                          const DomainArgs& args, double tor_period, double* df_out) {
    Array2D<double> cx = first_plane(args.cx);
    Array2D<double> cy = first_plane(args.cy);

    // mapping spans
    int span1 = find_span(args.t1.data, args.t1.shape[0], degree[0], eta1);
    int span2 = find_span(args.t2.data, args.t2.shape[0], degree[1], eta2);

    // non-zero splines of mapping, and derivatives
    double b1[MAX_SPLINE_DEGREE + 1], b2[MAX_SPLINE_DEGREE + 1];
    double der1[MAX_SPLINE_DEGREE + 1], der2[MAX_SPLINE_DEGREE + 1];

    b_der_splines_slim(args.t1.data, degree[0], eta1, span1, b1, der1);
    b_der_splines_slim(args.t2.data, degree[1], eta2, span2, b2, der2);

    Array1D<long long> tmp1 = spline_index_row(ind1, span1 - degree[0]);
    Array1D<long long> tmp2 = spline_index_row(ind2, span2 - degree[1]);

    df_out[0] = evaluation_kernel_2d(degree[0], degree[1], der1, b2, tmp1, tmp2, cx) * cos(2 * pi * eta3 / tor_period);
    df_out[1] = evaluation_kernel_2d(degree[0], degree[1], b1, der2, tmp1, tmp2, cx) * cos(2 * pi * eta3 / tor_period);
    df_out[2] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cx) * sin(2 * pi * eta3 / tor_period) *
                (-2 * pi / tor_period);
    df_out[3] = evaluation_kernel_2d(degree[0], degree[1], der1, b2, tmp1, tmp2, cx) * (-1) *
                sin(2 * pi * eta3 / tor_period);
    df_out[4] = evaluation_kernel_2d(degree[0], degree[1], b1, der2, tmp1, tmp2, cx) * (-1) *
                sin(2 * pi * eta3 / tor_period);
    df_out[5] = evaluation_kernel_2d(degree[0], degree[1], b1, b2, tmp1, tmp2, cx) * (-1) *
                cos(2 * pi * eta3 / tor_period) * 2 * pi / tor_period;
    df_out[6] = evaluation_kernel_2d(degree[0], degree[1], der1, b2, tmp1, tmp2, cy);
    df_out[7] = evaluation_kernel_2d(degree[0], degree[1], b1, der2, tmp1, tmp2, cy);
    df_out[8] = 0.;

    if (eta1 == 0. && cx(0, 0) == cx(0, 1)) {
        df_out[1] = 0.;
        df_out[4] = 0.;
    }

    if (eta1 == 0. && cy(0, 0) == cy(0, 1)) df_out[7] = 0.;
}
}
