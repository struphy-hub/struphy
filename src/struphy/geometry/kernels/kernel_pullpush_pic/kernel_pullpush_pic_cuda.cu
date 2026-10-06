#include "struphy/geometry/transform_kernels.cuh"

// Same arguments, in the same order, as the pyccel kernel in kernel_pullpush_pic_kernels.py.

/**
 * Whether a marker lies outside the logical cube [0, 1]^3, as tested in the pyccel loop of kernel_pullpush_pic.
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
 * Pull-backs, push-forwards and transformations for given markers, as in
 * kernel_pullpush_pic_kernels.kernel_pullpush_pic.
 *
 * One thread per marker row i (launched with n_threads = markers.shape[0], since the first array is a). The pyccel
 * kernel reads row `counter_a` of a and writes row `counter_o` of out, and returns the final counter_o. A CUDA kernel
 * returns nothing, so the counter is not returned; struphy's caller (Domain._pull_push_transform) passes
 * remove_outside=false and an `a` with one row per marker, for which counter_a == counter_o == i, and removes the
 * rows of outside markers itself, on both backends. Otherwise each thread counts the inside markers before row i
 * (O(i) work per thread, O(N^2) in total): correct, so that both versions fill out identically, but slow.
 *
 * @param a Values to transform, shape (N, n_comp) (with holes: one row per marker) or (n_inside, n_comp) (without
 *        holes), n_comp 1 (scalar kinds) or 3 (vector kinds), any strides.
 * @param markers Evaluation points in marker format (eta1, eta2, eta3 = markers[:, 0:3]), any strides.
 * @param kind_transform Which general transformation: 0 pull, 1 push, otherwise tran.
 * @param kind_fun Which detailed transformation (see pull, push, tran in transform_kernels.cuh).
 * @param args_domain Mapping arguments; every analytic mapping (spline mappings trap until CUDA strategy PR 19).
 * @param out Output values, shape (N, 3), any strides. Rows of outside markers are set to -1; entries a
 *        transformation does not write keep their value.
 * @param remove_outside Whether to skip markers outside [0, 1]^3 (compacting the rows of out).
 */
extern "C" __global__ void kernel_pullpush_pic(Array2D<double> a, Array2D<double> markers, int kind_transform,
                                               int kind_fun, DomainArgs args_domain, Array2D<double> out,
                                               bool remove_outside) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long np = markers.shape[0];
    if (i >= np) return;

    // CUDA-only: per-thread copies of the pyccel stack arrays tmp1 (length a.shape[1]) and tmp2 (out.shape[1]),
    // at most three entries each
    double tmp1[3], tmp2[3];
    long long n_tmp1 = a.shape[1] < 3 ? a.shape[1] : 3;
    long long n_tmp2 = out.shape[1] < 3 ? out.shape[1] : 3;

    // check if a has holes or not
    bool a_has_holes = a.shape[0] == np;

    double e1 = markers(i, 0);
    double e2 = markers(i, 1);
    double e3 = markers(i, 2);
    bool outside = marker_is_outside(markers, i);

    if (outside && remove_outside) return;

    // CUDA-only: the pyccel counters at row i
    long long counter_a = i;
    long long counter_o = i;
    if (remove_outside || !a_has_holes) {
        long long n_inside_before = 0;
        for (long long j = 0; j < i; ++j)
            if (!marker_is_outside(markers, j)) n_inside_before += 1;
        if (!a_has_holes) counter_a = n_inside_before;
        if (remove_outside) counter_o = n_inside_before;
    }

    // treatment of a hole
    if (outside) {
        for (long long k = 0; k < out.shape[1]; ++k) out(counter_o, k) = -1.0;
    }
    // treatment of "true" marker
    else {
        for (long long k = 0; k < n_tmp1; ++k) tmp1[k] = a(counter_a, k);
        for (long long k = 0; k < n_tmp2; ++k) tmp2[k] = out(counter_o, k);

        if (kind_transform == 0)
            struphy_cuda::pull(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);
        else if (kind_transform == 1)
            struphy_cuda::push(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);
        else
            struphy_cuda::tran(tmp1, e1, e2, e3, kind_fun, args_domain, tmp2);

        for (long long k = 0; k < n_tmp2; ++k) out(counter_o, k) = tmp2[k];
    }
}
