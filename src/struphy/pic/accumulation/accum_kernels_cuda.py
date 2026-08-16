"""Hand-written CUDA replacements for select accumulation (particle-to-grid
deposition) kernels, used only under ``ARRAY_BACKEND=cupy``.

Unlike the pusher kernels in :mod:`~struphy.pic.pushing.pusher_kernels_cuda`
(each marker only ever writes to its own row -- embarrassingly parallel, no
cross-thread interaction), accumulation kernels *scatter* every marker's
contribution into a shared grid array (:func:`~struphy.pic.accumulation.filler_kernels.fill_vec`'s
``vec[i1, i2, i3] += ...``): many markers whose (p+1)^3 local basis-function
support overlaps the same grid cell write to the same memory location. The
CPU kernel handles this by running the marker loop strictly sequentially (its
OpenMP ``reduction`` pragma is commented out in the source specifically
because of this race). The GPU port instead uses ``atomicAdd`` -- one thread
per marker, same as the pushers, but the grid write goes through an atomic
rather than a plain store. Double-precision ``atomicAdd`` is natively
supported on every CUDA compute capability this codebase targets (>= 6.0),
so no software fallback is needed.

Currently covered: :func:`~struphy.pic.accumulation.accum_kernels.charge_density_0form`,
used by :class:`~struphy.propagators.push_deterministic_diffusion.PushDeterministicDiffusion`
every step to build the (H^1) density field consumed by
:func:`~struphy.pic.pushing.pusher_kernels_cuda.push_deterministic_diffusion_stage_general_gpu`.
This one needs no domain-mapping Jacobian at all (the H^1 filling weight is
just the marker weight), so it reuses only the B-spline evaluation device
functions, not the geometry-mapping ones.
"""

_CHARGE_DENSITY_0FORM_SRC = r"""
#define MAXP 8

__device__ int find_span_dev(const double* t, int p, int len_t, double eta)
{
    int low = p;
    int high = len_t - 1 - p;

    if (eta <= t[low]) return low;
    if (eta >= t[high]) return high - 1;

    int span = (low + high) / 2;
    while (eta < t[span] || eta >= t[span + 1]) {
        if (eta < t[span]) high = span;
        else low = span;
        span = (low + high) / 2;
    }
    return span;
}

// Only the N-spline values (bn) are needed for an H^1/0-form fill; D-spline
// values are computed alongside (same recursion as
// pusher_kernels_cuda.py's b_d_splines_dev) and simply unused.
__device__ void b_splines_dev(const double* t, int p, double eta, int span, double* bn)
{
    double left[MAXP];
    double right[MAXP];

    for (int i = 0; i <= p; i++) bn[i] = 0.0;
    bn[0] = 1.0;

    for (int j = 0; j < p; j++) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.0;
        for (int r = 0; r <= j; r++) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

extern "C" __global__
void charge_density_0form_cuda(
    const double* markers,
    const int n_cols,
    const int n_markers,
    const int weight_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    double* vec, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double filling = row[weight_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bn1[il1] * filling;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bn2[il2];
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bn3[il3];
                atomicAdd(&vec[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3], b3);
            }
        }
    }
}
"""

_charge_density_0form_kernel = None


def _get_charge_density_0form_kernel():
    global _charge_density_0form_kernel
    if _charge_density_0form_kernel is None:
        import cupy as cp

        _charge_density_0form_kernel = cp.RawKernel(_CHARGE_DENSITY_0FORM_SRC, "charge_density_0form_cuda")
    return _charge_density_0form_kernel


def charge_density_0form_gpu(
    markers,
    weight_idx: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    vec_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.charge_density_0form`.

    ``markers`` is the host marker array, transferred to the device once per
    call (matching the pusher kernels' round-trip pattern). ``vec_dev`` is
    the target :class:`~feectools.linalg.stencil.StencilVector`'s ``._data``
    -- already device-resident under CuPy and already zeroed by the caller
    (:meth:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector._accumulate`
    always does ``dat[:] = 0.0`` before invoking the kernel), so this
    function only needs to add to it, not read markers back afterward: the
    caller reads ``vec_dev`` directly since it was written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = cp.asarray(markers)
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_charge_density_0form_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(weight_idx),
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            vec_dev,
            np.int32(vec_dev.shape[1]),
            np.int32(vec_dev.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# linear_vlasov_ampere: accumulates into a symmetric V1 -> V1 block matrix
# (mat11, mat12, mat13, mat22, mat23, mat33) plus a V1 vector (vec1, vec2,
# vec3), using DF^-1(eta_p) @ v_p at each marker -- unlike
# charge_density_0form this needs the full domain-mapping Jacobian, so the
# kernel source below is prefixed with pusher_kernels_cuda's
# _GENERAL_GEOMETRY_SRC (df_dispatch_dev and friends) rather than
# duplicating it.
#
# The row/column basis combinations for the 6 matrix blocks and the fill
# formulas mirror struphy.pic.accumulation.particle_to_mat_kernels.m_v_fill_b_v1_symm
# exactly (which itself calls filler_kernels.fill_mat_vec/fill_mat) --
# fill_mat_vec_dev/fill_mat_dev below are direct ports of those two.
# ---------------------------------------------------------------------------

_LINEAR_VLASOV_AMPERE_EXTRA_SRC = r"""
__device__ void outer_dev(const double* a, const double* b, double* c)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            c[3*i+j] = a[i] * b[j];
}

// Port of filler_kernels.fill_mat_vec: fills one matrix block (banded
// storage, j = pad + jl - il) and, along the shared (i1,i2,i3) row loop,
// also fills the corresponding vector block.
__device__ void fill_mat_vec_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat, int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec, int vn2, int vn3,
    double filling_vec)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];

                atomicAdd(&vec[(size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3], b3 * filling_vec);

                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat[idx], b6);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat: matrix-only block fill (off-diagonal
// blocks, no associated vector).
__device__ void fill_mat_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat, int d2, int d3, int d4, int d5, int d6,
    double filling_mat)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1] * filling_mat;
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1];
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat[idx], b6);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__
void linear_vlasov_ampere_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const double* f0_values,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    double* mat11, double* mat12, double* mat13,
    double* mat22, double* mat23, double* mat33,
    double* vec1, double* vec2, double* vec3,
    const int m11_d2, const int m11_d3, const int m11_d4, const int m11_d5, const int m11_d6,
    const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6,
    const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6,
    const int m22_d2, const int m22_d3, const int m22_d4, const int m22_d5, const int m22_d6,
    const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6,
    const int m33_d2, const int m33_d3, const int m33_d4, const int m33_d5, const int m33_d6,
    const int v1_n2, const int v1_n3,
    const int v2_n2, const int v2_n3,
    const int v3_n2, const int v3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};
    const double weight = row[6];
    const double s0 = row[7];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    double df_inv[9], df_inv_v[3];
    matrix_inv_dev(dfm, df_inv);
    matvec_dev(df_inv, v, df_inv_v);

    double filling_m[9];
    outer_dev(df_inv_v, df_inv_v, filling_m);
    const double fm_scale = f0_values[ip] / s0;
    for (int k = 0; k < 9; k++) filling_m[k] *= fm_scale;

    double filling_v[3];
    filling_v[0] = weight * df_inv_v[0];
    filling_v[1] = weight * df_inv_v[1];
    filling_v[2] = weight * df_inv_v[2];

    const double fill11 = filling_m[0], fill12 = filling_m[1], fill13 = filling_m[2];
    const double fill22 = filling_m[4], fill23 = filling_m[5], fill33 = filling_m[8];

    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    fill_mat_vec_dev(pd1,p2,p3, pd1,p2,p3, bd1,bn2,bn3, bd1,bn2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11,
        vec1, v1_n2,v1_n3, filling_v[0]);

    fill_mat_vec_dev(p1,pd2,p3, p1,pd2,p3, bn1,bd2,bn3, bn1,bd2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22,
        vec2, v2_n2,v2_n3, filling_v[1]);

    fill_mat_vec_dev(p1,p2,pd3, p1,p2,pd3, bn1,bn2,bd3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33,
        vec3, v3_n2,v3_n3, filling_v[2]);

    fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);

    fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);

    fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);
}
"""


def _linear_vlasov_ampere_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC


_linear_vlasov_ampere_kernel = None


def _get_linear_vlasov_ampere_kernel():
    global _linear_vlasov_ampere_kernel
    if _linear_vlasov_ampere_kernel is None:
        import cupy as cp

        _linear_vlasov_ampere_kernel = cp.RawKernel(_linear_vlasov_ampere_source(), "linear_vlasov_ampere_cuda")
    return _linear_vlasov_ampere_kernel


def linear_vlasov_ampere_gpu(
    markers,
    kind_map: int,
    params_dev,
    f0_values_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    mat11_dev,
    mat12_dev,
    mat13_dev,
    mat22_dev,
    mat23_dev,
    mat33_dev,
    vec1_dev,
    vec2_dev,
    vec3_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.linear_vlasov_ampere`.

    ``markers`` is the host marker array, round-tripped through the device
    once per call (this kernel only reads markers, never writes them back).
    ``params_dev``/``f0_values_dev`` and all ``mat*_dev``/``vec*_dev`` arrays
    are expected to already be device-resident (cached once by the caller);
    the ``mat*_dev``/``vec*_dev`` arrays must already be zeroed, matching
    :meth:`~struphy.pic.accumulation.particles_to_grid.Accumulator._accumulate`'s
    ``dat[:] = 0.0`` reset before the kernel call.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = cp.asarray(markers)
    f0_values_dev = cp.ascontiguousarray(f0_values_dev)
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def dims(a):
        return (
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_linear_vlasov_ampere_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            f0_values_dev,
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            mat11_dev,
            mat12_dev,
            mat13_dev,
            mat22_dev,
            mat23_dev,
            mat33_dev,
            vec1_dev,
            vec2_dev,
            vec3_dev,
            *dims(mat11_dev),
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat22_dev),
            *dims(mat23_dev),
            *dims(mat33_dev),
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )
