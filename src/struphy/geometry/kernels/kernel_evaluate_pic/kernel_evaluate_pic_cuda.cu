#include "struphy/geometry/evaluation_kernels.cuh"

// Same arguments, in the same order, as the pyccel kernel in kernel_evaluate_pic_kernels.py.

/**
 * Whether a marker lies outside the logical cube [0, 1]^3, as tested in the pyccel loop of kernel_evaluate_pic.
 *
 * CUDA-only helper for the condition `e1 < 0.0 or e1 > 1.0 or ...` (a NaN coordinate counts as inside, as in pyccel).
 *
 * @param markers Marker array view; the logical position is in columns 0-2.
 * @param i Marker row.
 * @return True if the marker is outside (a hole has coordinates -1).
 */
__device__ inline bool marker_is_outside(const Array2D<double>& markers, long long i) {
    double e1 = markers(i, 0);
    double e2 = markers(i, 1);
    double e3 = markers(i, 2);
    return e1 < 0.0 || e1 > 1.0 || e2 < 0.0 || e2 > 1.0 || e3 < 0.0 || e3 > 1.0;
}

/**
 * Evaluation of metric coefficients for given markers, as in kernel_evaluate_pic_kernels.kernel_evaluate_pic.
 *
 * One thread per marker row i. The pyccel kernel writes the result of the i-th marker to row `counter` of mat_f
 * and returns the final counter. A CUDA kernel returns nothing, so the counter is not returned; struphy's caller
 * (Domain._evaluate_metric_coefficient) passes remove_outside=false, for which counter == i, and removes the rows of
 * outside markers itself, on both backends. With remove_outside=true each thread counts the inside markers before
 * row i (O(i) work per thread, O(N^2) in total): correct, so that both versions fill mat_f identically, but slow.
 *
 * @param markers Evaluation points in marker format (eta1, eta2, eta3 = markers[:, 0:3]), any strides.
 * @param kind_coeff Which coefficient: -1 identity, 0 mapping F, 1 DF, 2 det(DF), 3 DF^(-1), 4 G, 5 G^(-1).
 * @param args Mapping arguments; every analytic mapping (spline mappings trap until CUDA strategy PR 19).
 * @param mat_f Output of shape (N, 3, 3), any strides. Rows of outside markers are set to -1 (identity: their
 *        coordinates in the first column); entries a coefficient does not write keep their value.
 * @param remove_outside Whether to skip markers outside [0, 1]^3 (compacting the rows of mat_f).
 * @param avoid_round_off Whether to set the analytically known zero entries of the mapping exactly to zero.
 */
extern "C" __global__ void kernel_evaluate_pic(Array2D<double> markers, int kind_coeff, DomainArgs args,
                                               Array3D<double> mat_f, bool remove_outside, bool avoid_round_off) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= markers.shape[0]) return;

    double tmp0[3], tmp1[9], tmp2[9], tmp3[9], out[9];

    double e1 = markers(i, 0);
    double e2 = markers(i, 1);
    double e3 = markers(i, 2);
    bool outside = marker_is_outside(markers, i);

    if (outside && remove_outside) return;

    // CUDA-only: the pyccel counter at row i (rows written before this marker)
    long long counter = i;
    if (remove_outside) {
        counter = 0;
        for (long long j = 0; j < i; ++j)
            if (!marker_is_outside(markers, j)) counter += 1;
    }

    if (outside) {
        if (kind_coeff >= 0) {
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) mat_f(counter, k, l) = -1.0;
        } else {
            mat_f(counter, 0, 0) = e1;
            mat_f(counter, 1, 0) = e2;
            mat_f(counter, 2, 0) = e3;
        }
    } else {
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) out[3 * k + l] = mat_f(counter, k, l);

        struphy_cuda::select_metric_coeff(e1, e2, e3, kind_coeff, args, tmp0, tmp1, tmp2, tmp3, avoid_round_off,
                                          out);

        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) mat_f(counter, k, l) = out[3 * k + l];
    }
}
