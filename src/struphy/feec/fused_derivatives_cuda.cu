// The discrete gradient with its boundary operators as one kernel (struphy.feec.fused_derivatives).
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

// Per component c, 8 entries of `meta`: first owned index (3, padded coordinates), owned sizes (3), derivative
// direction, negative (0 or 1).
#define FUSED_META 8

__device__ inline Array3D<double> pick(int c, Array3D<double> a0, Array3D<double> a1, Array3D<double> a2) {
    return c == 0 ? a0 : (c == 1 ? a1 : a2);
}

/**
 * out_c = B_out,c D_c B_in x for the components c of a gradient, B the 0/1 masks of the boundary operators and D_c the
 * derivative in direction dir_c (feectools DirectionalDerivativeOperator): on the owned entries i of out_c (padded
 * coordinates, shared by x and out_c as in DirectionalDerivativeOperator),
 *
 *     out_c[i] = ((x[i + e] * m_in[i + e]) - (x[i] * m_in[i])) * m_out_c[i]      (operands swapped if negative),
 *
 * the operations of the composed operators in the same order (bit for bit the same). One thread per owned entry of
 * all components (n_threads = sum of the owned sizes), component 0 first; the ghost regions of out are not written.
 *
 * @param x, m_in Input vector data (ghost regions in sync) and the mask of the input boundary operator.
 * @param out0, out1, out2, m0, m1, m2 Data of the output components and the masks of the output boundary operator.
 * @param meta FUSED_META entries per component, see above.
 * @param n_comp Number of components (1 to 3).
 */
extern "C" __global__ void fused_boundary_gradient(Array3D<double> x, Array3D<double> m_in, Array3D<double> out0,
                                                   Array3D<double> out1, Array3D<double> out2, Array3D<double> m0,
                                                   Array3D<double> m1, Array3D<double> m2, const long long* meta,
                                                   int n_comp) {
    long long n_total = 0;
    for (int c = 0; c < n_comp; ++c)
        n_total += meta[FUSED_META * c + 3] * meta[FUSED_META * c + 4] * meta[FUSED_META * c + 5];
    CUNUMPY_THREAD_1D(t, n_total);

    // CUDA-only: the component and owned entry of this thread
    int c = 0;
    long long local = t;
    while (c < n_comp - 1) {
        const long long size = meta[FUSED_META * c + 3] * meta[FUSED_META * c + 4] * meta[FUSED_META * c + 5];
        if (local < size) break;
        local -= size;
        ++c;
    }
    const long long* m = meta + FUSED_META * c;
    const long long i3 = m[2] + local % m[5];
    const long long i2 = m[1] + (local / m[5]) % m[4];
    const long long i1 = m[0] + local / (m[4] * m[5]);
    const int dir = (int)m[6];
    const long long j1 = i1 + (dir == 0), j2 = i2 + (dir == 1), j3 = i3 + (dir == 2);

    const double a = x(j1, j2, j3) * m_in(j1, j2, j3);
    const double b = x(i1, i2, i3) * m_in(i1, i2, i3);
    const double d = m[7] ? b - a : a - b;
    Array3D<double> out = pick(c, out0, out1, out2);
    Array3D<double> mask = pick(c, m0, m1, m2);
    out(i1, i2, i3) = d * mask(i1, i2, i3);
}

/**
 * The transpose: out = B_out (sum_c D_c^T B_in,c y_c), with D_c^T the transposed derivative; on the owned entries i of
 * out,
 *
 *     t_c = (y_c[i - e_c] * m_c[i - e_c]) - (y_c[i] * m_c[i])      (operands swapped if negative),
 *     out[i] = ((t_0 + t_1) + t_2) * m_out[i],
 *
 * summed in the order of BlockLinearOperator.dot. One thread per owned entry of out (n_threads = its owned size).
 *
 * @param y0, y1, y2, m0, m1, m2 Input components (ghost regions in sync) and the masks of the input boundary operator.
 * @param out, m_out Output data and the mask of the output boundary operator.
 * @param meta FUSED_META entries per component (the owned region of out, the same for every component).
 * @param n_comp Number of components (1 to 3).
 */
extern "C" __global__ void fused_boundary_gradient_t(Array3D<double> y0, Array3D<double> y1, Array3D<double> y2,
                                                     Array3D<double> m0, Array3D<double> m1, Array3D<double> m2,
                                                     Array3D<double> out, Array3D<double> m_out,
                                                     const long long* meta, int n_comp) {
    CUNUMPY_THREAD_1D(t, meta[3] * meta[4] * meta[5]);
    const long long i3 = meta[2] + t % meta[5];
    const long long i2 = meta[1] + (t / meta[5]) % meta[4];
    const long long i1 = meta[0] + t / (meta[4] * meta[5]);

    double acc = 0.0;
    for (int c = 0; c < n_comp; ++c) {
        const long long* m = meta + FUSED_META * c;
        const int dir = (int)m[6];
        const long long j1 = i1 - (dir == 0), j2 = i2 - (dir == 1), j3 = i3 - (dir == 2);
        Array3D<double> y = pick(c, y0, y1, y2);
        Array3D<double> mask = pick(c, m0, m1, m2);
        const double a = y(j1, j2, j3) * mask(j1, j2, j3);
        const double b = y(i1, i2, i3) * mask(i1, i2, i3);
        const double d = m[7] ? b - a : a - b;
        acc = c == 0 ? d : acc + d;
    }
    out(i1, i2, i3) = acc * m_out(i1, i2, i3);
}
