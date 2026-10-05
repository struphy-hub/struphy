#pragma once
#include "struphy/kernel_arguments/array_view.cuh"
namespace struphy_cuda {
// Fixed capacity of the per-thread spline scratch arrays.
constexpr int MAX_SPLINE_DEGREE = 8;

// Raw pointers are used by the contiguous DerhamArgs knots; Array1D keeps
// arbitrary CuPy views indexable without exposing their strides to callers.
__device__ inline double knot_value(const double* t, long long i) {
    return t[i];
}

template <class T>
__device__ inline T knot_value(const Array1D<T>& t, long long i) {
    return t(i);
}

/**
 * Compute the knot span containing eta, as in bsplines_kernels.find_span.
 *
 * @param t Knot sequence, as a contiguous pointer or an Array1D view.
 * @param nt Number of knots; CUDA pointers do not carry the length len(t).
 * @param p B-spline degree.
 * @param eta Evaluation point.
 * @return Span index of the non-vanishing splines, clamped at the boundaries.
 */
template <class KnotArray>
__device__ inline int find_span(const KnotArray& t, int nt, int p, double eta) {
    int low = p, high = nt - 1 - p;
    int returnVal;
    if (eta <= knot_value(t, low)) {
        returnVal = low;
    } else if (eta >= knot_value(t, high)) {
        returnVal = high - 1;
    } else {
        int span = (low + high) / 2;
        while (eta < knot_value(t, span) || eta >= knot_value(t, span + 1)) {
            if (eta < knot_value(t, span)) high = span;
            else low = span;
            span = (low + high) / 2;
        }
        returnVal = span;
    }
    return returnVal;
}

template <class T>
__device__ inline int find_span(const Array1D<T>& t, int p, double eta) {
    return find_span(t, static_cast<int>(t.shape[0]), p, eta);
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
 * @param tn B-spline knot sequence, as a contiguous pointer or an Array1D view.
 * @param pn B-spline degree, between 1 and MAX_SPLINE_DEGREE.
 * @param eta Evaluation point.
 * @param span Knot span index from find_span.
 * @param bn Output buffer for pn + 1 B-spline values.
 * @param bd Output buffer for pn D-spline values of degree pd = pn - 1.
 *
 * D-splines are the scaled B-splines from the penultimate recursion step.
 * left/right are fixed-size, thread-local scratch arrays.
 */
template <class KnotArray>
__device__ inline void b_d_splines_slim(const KnotArray& tn, int pn, double eta, int span, double* bn, double* bd) {
    int pd = pn - 1;
    double left[MAX_SPLINE_DEGREE], right[MAX_SPLINE_DEGREE];
    bn[0] = 1.;
    for (int j = 0; j < pn; ++j) {
        left[j] = eta - knot_value(tn, span - j);
        right[j] = knot_value(tn, span + 1 + j) - eta;
        double saved = 0.;
        if (j == pn - 1) {
            for (int il = 0; il <= pd; ++il) {
                bd[pd - il] = pn / (knot_value(tn, span - il + pn) - knot_value(tn, span - il)) * bn[pd - il];
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
}
