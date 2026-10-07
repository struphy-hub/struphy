#pragma once
namespace struphy_cuda {
// Fixed capacity of the per-thread spline scratch arrays.
constexpr int MAX_SPLINE_DEGREE = 8;

/**
 * Compute the knot span containing eta, as in bsplines_kernels.find_span.
 *
 * @param t Knot sequence.
 * @param nt Number of knots; CUDA pointers do not carry the length len(t).
 * @param p B-spline degree.
 * @param eta Evaluation point.
 * @return Span index of the non-vanishing splines, clamped at the boundaries.
 */
__device__ inline int find_span(const double* t, int nt, int p, double eta) {
    int low = p, high = nt - 1 - p;
    int returnVal;
    if (eta <= t[low]) {
        returnVal = low;
    } else if (eta >= t[high]) {
        returnVal = high - 1;
    } else {
        int span = (low + high) / 2;
        while (eta < t[span] || eta >= t[span + 1]) {
            if (eta < t[span]) high = span;
            else low = span;
            span = (low + high) / 2;
        }
        returnVal = span;
    }
    return returnVal;
}

/**
 * Evaluate the p + 1 non-vanishing B-splines, as in bsplines_kernels.basis_funs.
 *
 * @param t Knot sequence.
 * @param p Degree, at most MAX_SPLINE_DEGREE.
 * @param eta Evaluation point.
 * @param span Knot span index from find_span.
 * @param values Output buffer with at least p + 1 entries.
 *
 * The Pyccel left/right scratch arguments are thread-local arrays here.
 */
__device__ inline void basis_funs(const double* t, int p, double eta, int span, double* values) {
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    values[0] = 1.;
    for (int j = 0; j < p; ++j) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.;
        for (int r = 0; r <= j; ++r) {
            double temp = values[r] / (right[r] + left[j - r]);
            values[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        values[j + 1] = saved;
    }
}

/**
 * Evaluate B- and D-splines, as in bsplines_kernels.b_d_splines_slim.
 *
 * @param tn B-spline knot sequence.
 * @param pn B-spline degree, between 1 and MAX_SPLINE_DEGREE.
 * @param eta Evaluation point.
 * @param span Knot span index from find_span.
 * @param bn Output buffer for pn + 1 B-spline values.
 * @param bd Output buffer for pn D-spline values of degree pd = pn - 1.
 *
 * D-splines are the scaled B-splines from the penultimate recursion step.
 * left/right are fixed-size, thread-local scratch arrays.
 */
__device__ inline void b_d_splines_slim(const double* tn, int pn, double eta, int span, double* bn, double* bd) {
    int pd = pn - 1;
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    bn[0] = 1.;
    for (int j = 0; j < pn; ++j) {
        left[j] = eta - tn[span - j];
        right[j] = tn[span + 1 + j] - eta;
        double saved = 0.;
        if (j == pn - 1) {
            for (int il = 0; il <= pd; ++il) {
                bd[pd - il] = pn / (tn[span - il + pn] - tn[span - il]) * bn[pd - il];
            }
        }
        for (int r = 0; r <= j; ++r) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

/**
 * Evaluate the pn + 1 non-vanishing B-splines, as in bsplines_kernels.b_splines_slim.
 *
 * @param tn Knot sequence.
 * @param pn B-spline degree, at most MAX_SPLINE_DEGREE.
 * @param eta Evaluation point.
 * @param span Knot span index from find_span.
 * @param values Output buffer with at least pn + 1 entries.
 *
 * Same recursion as basis_funs; the Pyccel left/right arrays are thread-local arrays here.
 */
__device__ inline void b_splines_slim(const double* tn, int pn, double eta, int span, double* values) {
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    values[0] = 1.;
    for (int j = 0; j < pn; ++j) {
        left[j] = eta - tn[span - j];
        right[j] = tn[span + 1 + j] - eta;
        double saved = 0.;
        for (int r = 0; r <= j; ++r) {
            double temp = values[r] / (right[r] + left[j - r]);
            values[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        values[j + 1] = saved;
    }
}

/**
 * Evaluate the pn + 1 non-vanishing B-splines and their derivatives, as in bsplines_kernels.b_der_splines_slim.
 *
 * @param tn Knot sequence.
 * @param pn B-spline degree, between 1 and MAX_SPLINE_DEGREE.
 * @param eta Evaluation point.
 * @param span Knot span index from find_span.
 * @param bn Output buffer for the pn + 1 B-spline values.
 * @param der Output buffer for the pn + 1 derivatives of the B-splines.
 *
 * The Pyccel scratch arrays left, right, diff and the triangular table values ((pn + 1) x (pn + 1)) are
 * fixed-size, thread-local arrays here.
 */
__device__ inline void b_der_splines_slim(const double* tn, int pn, double eta, int span, double* bn, double* der) {
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE], diff[MAX_SPLINE_DEGREE];
    double values[MAX_SPLINE_DEGREE + 1][MAX_SPLINE_DEGREE + 1];
    values[0][0] = 1.;
    for (int j = 0; j < pn; ++j) {
        left[j] = eta - tn[span - j];
        right[j] = tn[span + 1 + j] - eta;
        double saved = 0.;
        for (int r = 0; r <= j; ++r) {
            diff[r] = 1. / (right[r] + left[j - r]);
            double temp = values[j][r] * diff[r];
            values[j + 1][r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        values[j + 1][j + 1] = saved;
    }

    for (int r = 0; r < pn; ++r) diff[r] = diff[r] * pn;

    // compute derivatives
    // j = 0
    double saved = values[pn - 1][0] * diff[0];
    der[0] = -saved;

    // j = 1, ... , pn - 1
    for (int j = 1; j < pn; ++j) {
        double temp = saved;
        saved = values[pn - 1][j] * diff[j];
        der[j] = temp - saved;
    }

    // j = pn
    for (int j = 0; j <= pn; ++j) bn[j] = values[pn][j];
    der[pn] = saved;
}
}
