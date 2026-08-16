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
    dev_markers = markers
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


# ---------------------------------------------------------------------------
# vlasov_maxwell: same symmetric V1 -> V1 6-block-matrix-plus-vector fill as
# linear_vlasov_ampere (reuses fill_mat_vec_dev/fill_mat_dev from
# _LINEAR_VLASOV_AMPERE_EXTRA_SRC above), but with a different filling:
# A_p = w_p * G^-1(eta_p) (the metric inverse, not an outer product of
# velocity) and B_p = w_p * DF^-1(eta_p) v_p -- no f0_values/s0 involved, so
# unlike linear_vlasov_ampere this one can't hit the inf/nan-from-div-by-s0
# path. Also note: the CPU reference only skips markers[ip,0]==-1.0 (no
# markers[ip,-1]==-2.0 check), unlike linear_vlasov_ampere -- ported as-is.
# ---------------------------------------------------------------------------

_VLASOV_MAXWELL_EXTRA_SRC = r"""
extern "C" __global__
void vlasov_maxwell_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
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
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};
    const double weight = row[6];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    double df_inv[9], df_inv_v[3];
    matrix_inv_dev(dfm, df_inv);
    matvec_dev(df_inv, v, df_inv_v);

    // g_inv = DF^-1 @ DF^-T ; g_inv[i,j] = sum_k df_inv[i,k]*df_inv[j,k]
    double filling_m[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += df_inv[3*i+k] * df_inv[3*j+k];
            filling_m[3*i+j] = weight * s;
        }
    }

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


def _vlasov_maxwell_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _VLASOV_MAXWELL_EXTRA_SRC


_vlasov_maxwell_kernel = None


def _get_vlasov_maxwell_kernel():
    global _vlasov_maxwell_kernel
    if _vlasov_maxwell_kernel is None:
        import cupy as cp

        _vlasov_maxwell_kernel = cp.RawKernel(_vlasov_maxwell_source(), "vlasov_maxwell_cuda")
    return _vlasov_maxwell_kernel


def vlasov_maxwell_gpu(
    markers,
    kind_map: int,
    params_dev,
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
    :func:`~struphy.pic.accumulation.accum_kernels.vlasov_maxwell`. Same
    calling convention as :func:`linear_vlasov_ampere_gpu`, minus
    ``f0_values`` (this kernel doesn't need a background distribution).
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
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

    _get_vlasov_maxwell_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
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
    dev_markers = markers
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


# ---------------------------------------------------------------------------
# cc_lin_mhd_6d_1: accumulates into the 3 antisymmetric off-diagonal blocks
# (mat12, mat13, mat23) of a V_u -> V_u matrix, no vector, where V_u is
# whichever of H1vec/Hcurl/Hdiv the propagator's ``u_space`` option selects
# (runtime int ``basis_u`` in {0, 1, 2}). All 3 branches ultimately do a
# 3-block antisymmetric fill using fill_mat_dev (from
# _LINEAR_VLASOV_AMPERE_EXTRA_SRC above) with the row/col basis-degree
# combination matching struphy's mat_fill_v0vec_asym (basis_u=0, N-N-N both
# sides), mat_fill_v1_asym (basis_u=1, D-N-N/N-D-N/N-N-D -- same combination
# already used for linear_vlasov_ampere/vlasov_maxwell's off-diagonal
# blocks) and mat_fill_v2_asym (basis_u=2, Hdiv's N-D-D/D-N-D/D-D-N). Since
# basis_u is one value per kernel LAUNCH (not per marker), the branch is
# warp-coherent -- every thread takes the same path, no divergence cost.
# basis_u=0 needs no domain Jacobian at all (see the CPU reference: dfm is
# computed unconditionally there but only actually used by basis_u 1/2), so
# df_dispatch_dev is only called inside the basis_u==1/2 branches here.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_6D_1_SRC = r"""
extern "C" __global__
void cc_lin_mhd_6d_1_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int b2_1_n2, const int b2_1_n3,
    const double* b2_2, const int b2_2_n2, const int b2_2_n3,
    const double* b2_3, const int b2_3_n2, const int b2_3_n3,
    const int basis_u, const double scale_mat, const double boundary_cut,
    double* mat12, double* mat13, double* mat23,
    const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6,
    const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6,
    const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[0] < boundary_cut || row[0] > 1.0 - boundary_cut) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double weight = row[6];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        b2_1, b2_1_n2, b2_1_n3, b2_2, b2_2_n2, b2_2_n3, b2_3, b2_3_n2, b2_3_n3, b);

    // b_prod = bx() as a row-major 3x3 matrix
    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};

    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    double fill12, fill13, fill23;

    if (basis_u == 0) {
        fill12 = -weight * b_prod[1] * scale_mat;
        fill13 = -weight * b_prod[2] * scale_mat;
        fill23 = -weight * b_prod[5] * scale_mat;

        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);

    } else if (basis_u == 1) {
        double dfm[9];
        if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
        double df_inv[9], g_inv[9];
        matrix_inv_dev(dfm, df_inv);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) s += df_inv[3*i+k] * df_inv[3*j+k];
                g_inv[3*i+j] = s;
            }
        double tmp1[9], tmp2[9];
        matmat_dev(g_inv, b_prod, tmp1);
        matmat_dev(tmp1, g_inv, tmp2);

        fill12 = -weight * tmp2[1] * scale_mat;
        fill13 = -weight * tmp2[2] * scale_mat;
        fill23 = -weight * tmp2[5] * scale_mat;

        fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);

    } else if (basis_u == 2) {
        double dfm[9];
        if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
        const double det_df = det3_dev(dfm);
        const double det2 = det_df * det_df;

        fill12 = -weight * b_prod[1] * scale_mat / det2;
        fill13 = -weight * b_prod[2] * scale_mat / det2;
        fill23 = -weight * b_prod[5] * scale_mat / det2;

        // Hdiv component shapes: comp1 = N-D-D, comp2 = D-N-D, comp3 = D-D-N
        fill_mat_dev(p1,pd2,pd3, pd1,p2,pd3, bn1,bd2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(p1,pd2,pd3, pd1,pd2,p3, bn1,bd2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(pd1,p2,pd3, pd1,pd2,p3, bd1,bn2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);
    }
}
"""


def _cc_lin_mhd_6d_1_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_6D_1_SRC


_cc_lin_mhd_6d_1_kernel = None


def _get_cc_lin_mhd_6d_1_kernel():
    global _cc_lin_mhd_6d_1_kernel
    if _cc_lin_mhd_6d_1_kernel is None:
        import cupy as cp

        _cc_lin_mhd_6d_1_kernel = cp.RawKernel(_cc_lin_mhd_6d_1_source(), "cc_lin_mhd_6d_1_cuda")
    return _cc_lin_mhd_6d_1_kernel


def cc_lin_mhd_6d_1_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    basis_u: int,
    scale_mat: float,
    boundary_cut: float,
    mat12_dev,
    mat13_dev,
    mat23_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.cc_lin_mhd_6d_1`.
    ``b2_*_dev`` are the Hdiv (2-form) magnetic field FE coefficients,
    already device-resident.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    b2_1_dev = cp.ascontiguousarray(b2_1_dev)
    b2_2_dev = cp.ascontiguousarray(b2_2_dev)
    b2_3_dev = cp.ascontiguousarray(b2_3_dev)
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

    _get_cc_lin_mhd_6d_1_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
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
            b2_1_dev,
            np.int32(b2_1_dev.shape[1]),
            np.int32(b2_1_dev.shape[2]),
            b2_2_dev,
            np.int32(b2_2_dev.shape[1]),
            np.int32(b2_2_dev.shape[2]),
            b2_3_dev,
            np.int32(b2_3_dev.shape[1]),
            np.int32(b2_3_dev.shape[2]),
            np.int32(basis_u),
            np.float64(scale_mat),
            np.float64(boundary_cut),
            mat12_dev,
            mat13_dev,
            mat23_dev,
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat23_dev),
        ),
    )


# ---------------------------------------------------------------------------
# cc_lin_mhd_6d_2: like cc_lin_mhd_6d_1 (B2 field evaluation, bx() matrix,
# runtime basis_u in {0, 1, 2} selecting H1vec/Hcurl/Hdiv), but fills the
# full symmetric 6-block matrix plus a vector (like linear_vlasov_ampere /
# vlasov_maxwell), not just the 3 antisymmetric off-diagonal blocks. Basis
# combinations per branch (matching struphy's m_v_fill_v0vec_symm /
# m_v_fill_v1_symm / m_v_fill_v2_symm): basis_u=0 uses N-N-N everywhere
# (all 6 matrix blocks AND the vector); basis_u=1 is the same D-N-N/N-D-N/
# N-N-D combination already used for linear_vlasov_ampere/vlasov_maxwell;
# basis_u=2 is Hdiv's N-D-D/D-N-D/D-D-N (same as cc_lin_mhd_6d_1's
# basis_u=2). Per the CPU reference, basis_u=0 and 2 only ever need df_inv
# (g_inv is computed there but never actually used in those two branches --
# not replicated here); only basis_u=1 needs the full g_inv = DF^-1 DF^-T.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_6D_2_SRC = r"""
extern "C" __global__
void cc_lin_mhd_6d_2_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int b2_1_n2, const int b2_1_n3,
    const double* b2_2, const int b2_2_n2, const int b2_2_n3,
    const double* b2_3, const int b2_3_n2, const int b2_3_n3,
    const int basis_u, const double scale_mat, const double scale_vec, const double boundary_cut,
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
    if (row[0] == -1.0) return;
    if (row[0] < boundary_cut || row[0] > 1.0 - boundary_cut) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};
    const double weight = row[6];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        b2_1, b2_1_n2, b2_1_n3, b2_2, b2_2_n2, b2_2_n3, b2_3, b2_3_n2, b2_3_n3, b);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);
    double df_inv[9];
    matrix_inv_dev(dfm, df_inv);

    double tmp1[9], tmp_m[9], tmp_v[3];

    if (basis_u == 1) {
        double g_inv[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) s += df_inv[3*i+k] * df_inv[3*j+k];
                g_inv[3*i+j] = s;
            }
        double tmp0[9];
        matmat_dev(g_inv, b_prod, tmp0);
        matmat_dev(tmp0, df_inv, tmp1);
    } else {
        // basis_u == 0 or 2: tmp1 = b_prod @ df_inv (g_inv computed but
        // unused in the CPU reference for these two branches)
        matmat_dev(b_prod, df_inv, tmp1);
    }

    // tmp_m = tmp1 @ tmp1^T ; tmp_v = tmp1 @ v
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += tmp1[3*i+k] * tmp1[3*j+k];
            tmp_m[3*i+j] = s;
        }
    }
    matvec_dev(tmp1, v, tmp_v);

    double mat_scale = weight * scale_mat;
    double vec_scale = weight * scale_vec;
    if (basis_u == 2) {
        mat_scale /= det_df * det_df;
        vec_scale /= det_df;
    }

    double filling_m[9], filling_v[3];
    for (int k = 0; k < 9; k++) filling_m[k] = tmp_m[k] * mat_scale;
    for (int k = 0; k < 3; k++) filling_v[k] = tmp_v[k] * vec_scale;

    const double fill11 = filling_m[0], fill12 = filling_m[1], fill13 = filling_m[2];
    const double fill22 = filling_m[4], fill23 = filling_m[5], fill33 = filling_m[8];

    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    if (basis_u == 0) {
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11, vec1, v1_n2,v1_n3, filling_v[0]);
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22, vec2, v2_n2,v2_n3, filling_v[1]);
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33, vec3, v3_n2,v3_n3, filling_v[2]);

        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);

    } else if (basis_u == 1) {
        fill_mat_vec_dev(pd1,p2,p3, pd1,p2,p3, bd1,bn2,bn3, bd1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11, vec1, v1_n2,v1_n3, filling_v[0]);
        fill_mat_vec_dev(p1,pd2,p3, p1,pd2,p3, bn1,bd2,bn3, bn1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22, vec2, v2_n2,v2_n3, filling_v[1]);
        fill_mat_vec_dev(p1,p2,pd3, p1,p2,pd3, bn1,bn2,bd3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33, vec3, v3_n2,v3_n3, filling_v[2]);

        fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);

    } else if (basis_u == 2) {
        // Hdiv component shapes: comp1 = N-D-D, comp2 = D-N-D, comp3 = D-D-N
        fill_mat_vec_dev(p1,pd2,pd3, p1,pd2,pd3, bn1,bd2,bd3, bn1,bd2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11, vec1, v1_n2,v1_n3, filling_v[0]);
        fill_mat_vec_dev(pd1,p2,pd3, pd1,p2,pd3, bd1,bn2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22, vec2, v2_n2,v2_n3, filling_v[1]);
        fill_mat_vec_dev(pd1,pd2,p3, pd1,pd2,p3, bd1,bd2,bn3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33, vec3, v3_n2,v3_n3, filling_v[2]);

        fill_mat_dev(p1,pd2,pd3, pd1,p2,pd3, bn1,bd2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);
        fill_mat_dev(p1,pd2,pd3, pd1,pd2,p3, bn1,bd2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);
        fill_mat_dev(pd1,p2,pd3, pd1,pd2,p3, bd1,bn2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);
    }
}
"""


def _cc_lin_mhd_6d_2_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_6D_2_SRC


_cc_lin_mhd_6d_2_kernel = None


def _get_cc_lin_mhd_6d_2_kernel():
    global _cc_lin_mhd_6d_2_kernel
    if _cc_lin_mhd_6d_2_kernel is None:
        import cupy as cp

        _cc_lin_mhd_6d_2_kernel = cp.RawKernel(_cc_lin_mhd_6d_2_source(), "cc_lin_mhd_6d_2_cuda")
    return _cc_lin_mhd_6d_2_kernel


def cc_lin_mhd_6d_2_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    basis_u: int,
    scale_mat: float,
    scale_vec: float,
    boundary_cut: float,
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
    :func:`~struphy.pic.accumulation.accum_kernels.cc_lin_mhd_6d_2`.
    ``b2_*_dev`` are the Hdiv (2-form) magnetic field FE coefficients,
    already device-resident.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    b2_1_dev = cp.ascontiguousarray(b2_1_dev)
    b2_2_dev = cp.ascontiguousarray(b2_2_dev)
    b2_3_dev = cp.ascontiguousarray(b2_3_dev)
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

    _get_cc_lin_mhd_6d_2_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
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
            b2_1_dev,
            np.int32(b2_1_dev.shape[1]),
            np.int32(b2_1_dev.shape[2]),
            b2_2_dev,
            np.int32(b2_2_dev.shape[1]),
            np.int32(b2_2_dev.shape[2]),
            b2_3_dev,
            np.int32(b2_3_dev.shape[1]),
            np.int32(b2_3_dev.shape[2]),
            np.int32(basis_u),
            np.float64(scale_mat),
            np.float64(scale_vec),
            np.float64(boundary_cut),
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


# ---------------------------------------------------------------------------
# pc_lin_mhd_6d_full / pc_lin_mhd_6d: accumulate a "pressure tensor" -- the
# same DF^-1(eta_p) DF^-T(eta_p) V1 -> V1 filling as vlasov_maxwell, but
# additionally scaled by every v_a*v_b product (a,b in x,y,z) of the marker
# velocity, giving one full symmetric 6-block matrix PER velocity-pair (6
# pairs: xx, xy, xz, yy, yz, zz -> 6*6=36 matrix arrays) plus one vector PER
# velocity-component (3*3=9 vector arrays) -- see
# particle_to_mat_kernels.m_v_fill_v1_pressure_full and
# filler_kernels.fill_mat_vec_pressure_full/fill_mat_pressure_full, which
# this is a direct port of (fill_mat_vec_pressure_full_dev/
# fill_mat_pressure_full_dev below are the CUDA equivalents, generalizing
# fill_mat_vec_dev/fill_mat_dev from a single mat/vec output to six/three).
#
# pc_lin_mhd_6d (no "_full") is the same accumulation restricted to the
# (x, y) "perpendicular" velocity plane only: 3 velocity-pairs (xx, xy, yy)
# and 2 velocity-components (x, y), i.e. 6*3=18 matrix arrays and 3*2=6
# vector arrays -- see m_v_fill_v1_pressure/fill_mat_vec_pressure/
# fill_mat_pressure. Both variants are called with the SAME 36+9=45 output
# arrays (the propagators share one call signature for _full and non-_full)
# but pc_lin_mhd_6d only ever writes the 24 "perp" ones -- the CPU reference
# leaves the other 21 untouched (at whatever the caller zeroed them to), and
# so does this port: pc_lin_mhd_6d_gpu accepts all 45 positionally (to match
# Accumulator._accumulate's ``*self._args_data`` unpacking) but only passes
# the 24 it needs into the CUDA launch.
#
# Both variants only differ from vlasov_maxwell's filling in the v_a*v_b
# scaling and in which marker column holds the weight: pc_lin_mhd_6d_full
# uses markers[ip, 8], pc_lin_mhd_6d uses markers[ip, 6] (matching the CPU
# reference exactly).
# ---------------------------------------------------------------------------

_SPATIAL_BLOCKS = ("11", "12", "13", "22", "23", "33")
# (row_degrees_index, col_degrees_index) as 0/1/2 picking from (p, pd) per
# axis -- i.e. which of bn/bd (and p/pd) each spatial block's row/col use in
# each of the 3 axes. 0 = N-spline/degree p, 1 = D-spline/degree p-1.
_SPATIAL_BASIS = {
    "11": ((1, 0, 0), (1, 0, 0)),
    "22": ((0, 1, 0), (0, 1, 0)),
    "33": ((0, 0, 1), (0, 0, 1)),
    "12": ((1, 0, 0), (0, 1, 0)),
    "13": ((1, 0, 0), (0, 0, 1)),
    "23": ((0, 1, 0), (0, 0, 1)),
}
_DIAG_SPATIAL_BLOCKS = ("11", "22", "33")

_PC_PRESSURE_FILLERS_SRC = r"""
// Port of filler_kernels.fill_mat_vec_pressure_full: like fill_mat_vec_dev
// but scatters into 6 matrix blocks (scaled by vx*vx, vx*vy, vx*vz, vy*vy,
// vy*vz, vz*vz) and 3 vector blocks (scaled by vx, vy, vz) in one pass.
__device__ void fill_mat_vec_pressure_full_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_13, double* mat_22, double* mat_23, double* mat_33,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec_1, double* vec_2, double* vec_3, int vn2, int vn3,
    double filling_vec,
    double vx, double vy, double vz)
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
                size_t vidx = (size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3;
                double bv = b3 * filling_vec;
                atomicAdd(&vec_1[vidx], bv * vx);
                atomicAdd(&vec_2[vidx], bv * vy);
                atomicAdd(&vec_3[vidx], bv * vz);

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
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_13[idx], b6 * vx * vz);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                            atomicAdd(&mat_23[idx], b6 * vy * vz);
                            atomicAdd(&mat_33[idx], b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_pressure_full: same as above minus the
// vector part (off-diagonal spatial blocks have no associated vector).
__device__ void fill_mat_pressure_full_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_13, double* mat_22, double* mat_23, double* mat_33,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double vx, double vy, double vz)
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
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_13[idx], b6 * vx * vz);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                            atomicAdd(&mat_23[idx], b6 * vy * vz);
                            atomicAdd(&mat_33[idx], b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_vec_pressure: the "perp" (xy-plane only)
// variant -- 3 matrix blocks (vx*vx, vx*vy, vy*vy) and 2 vector blocks
// (vx, vy).
__device__ void fill_mat_vec_pressure_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_22,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec_1, double* vec_2, int vn2, int vn3,
    double filling_vec,
    double vx, double vy)
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
                size_t vidx = (size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3;
                double bv = b3 * filling_vec;
                atomicAdd(&vec_1[vidx], bv * vx);
                atomicAdd(&vec_2[vidx], bv * vy);

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
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_pressure: "perp" matrix-only variant.
__device__ void fill_mat_pressure_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_22,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double vx, double vy)
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
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}
"""


def _basis_args(row_or_col):
    """row_or_col is a 3-tuple of 0/1 (0=N-spline/degree p, 1=D-spline/degree p-1)
    for (axis1, axis2, axis3). Returns (degree_expr_list, basis_expr_list)."""
    deg = [
        "p1" if row_or_col[0] == 0 else "pd1",
        "p2" if row_or_col[1] == 0 else "pd2",
        "p3" if row_or_col[2] == 0 else "pd3",
    ]
    bas = [("bn1", "bd1")[row_or_col[0]], ("bn2", "bd2")[row_or_col[1]], ("bn3", "bd3")[row_or_col[2]]]
    return deg, bas


def _build_pc_lin_mhd_6d_kernel_src(full: bool) -> str:
    vel_pairs = _SPATIAL_BLOCKS if full else ("11", "12", "22")
    vec_is = ("1", "2", "3") if full else ("1", "2")
    kernel_name = "pc_lin_mhd_6d_full_cuda" if full else "pc_lin_mhd_6d_cuda"
    weight_col = 8 if full else 6

    mat_params = ", ".join(f"double* mat{sp}_{vel}" for vel in vel_pairs for sp in _SPATIAL_BLOCKS)
    vec_params = ", ".join(f"double* vec{mu}_{i}" for i in vec_is for mu in ("1", "2", "3"))
    mat_dim_params = ", ".join(
        f"const int m{sp}_d2, const int m{sp}_d3, const int m{sp}_d4, const int m{sp}_d5, const int m{sp}_d6"
        for sp in _SPATIAL_BLOCKS
    )
    vec_dim_params = ", ".join(f"const int v{mu}_n2, const int v{mu}_n3" for mu in ("1", "2", "3"))

    lines = []
    lines.append(f'extern "C" __global__\nvoid {kernel_name}(')
    lines.append("    const double* markers, const int n_cols, const int n_markers,")
    lines.append("    const int kind_map, const double* params,")
    lines.append("    const int p1, const int p2, const int p3,")
    lines.append("    const double* tn1, const int len_tn1,")
    lines.append("    const double* tn2, const int len_tn2,")
    lines.append("    const double* tn3, const int len_tn3,")
    lines.append("    const int start0, const int start1, const int start2,")
    lines.append("    const double ep_scale,")
    lines.append(f"    {mat_params},")
    lines.append(f"    {vec_params},")
    lines.append(f"    {mat_dim_params},")
    lines.append(f"    {vec_dim_params})")
    lines.append("{")
    lines.append("    int ip = blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append("    if (ip >= n_markers) return;")
    lines.append("")
    lines.append("    const double* row = markers + (size_t)ip * n_cols;")
    lines.append("    if (row[0] == -1.0) return;")
    lines.append("")
    lines.append("    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];")
    lines.append("    const double v[3] = {row[3], row[4], row[5]};")
    lines.append(f"    const double weight = row[{weight_col}];")
    lines.append("")
    lines.append("    double dfm[9];")
    lines.append("    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;")
    lines.append("    double df_inv[9];")
    lines.append("    matrix_inv_dev(dfm, df_inv);")
    lines.append("    double g_inv[9];")
    lines.append("    for (int i = 0; i < 3; i++)")
    lines.append("        for (int j = 0; j < 3; j++) {")
    lines.append("            double s = 0.0;")
    lines.append("            for (int k = 0; k < 3; k++) s += df_inv[3*i+k] * df_inv[3*j+k];")
    lines.append("            g_inv[3*i+j] = s;")
    lines.append("        }")
    lines.append("    double tmp_v[3];")
    lines.append("    matvec_dev(df_inv, v, tmp_v);")
    lines.append("")
    # fill11..fill33 are per-SPATIAL-block (mu,nu) scalars (weight*g_inv[mu,nu]*
    # ep_scale) -- needed by every spatial block's filler call regardless of
    # ``full``, since even the "perp" (non-full) variant fills all 6 spatial
    # blocks, just with fewer velocity-pairs per block. Only fill3/vz (the
    # z-velocity-component scalar, used solely by the vector fill) is
    # full-only.
    lines.append("    const double fill11 = weight * g_inv[0] * ep_scale;")
    lines.append("    const double fill12 = weight * g_inv[1] * ep_scale;")
    lines.append("    const double fill13 = weight * g_inv[2] * ep_scale;")
    lines.append("    const double fill22 = weight * g_inv[4] * ep_scale;")
    lines.append("    const double fill23 = weight * g_inv[5] * ep_scale;")
    lines.append("    const double fill33 = weight * g_inv[8] * ep_scale;")
    # fill1/fill2/fill3 are per-DIAG-SPATIAL-BLOCK (mu=1/2/3) vector filling
    # scalars (weight*tmp_v[mu-1]*ep_scale, tmp_v = DF^-1 @ v) -- needed by
    # spatial block 33's diagonal fill regardless of ``full`` too, just like
    # fill11..fill33 above. Only ``vz`` (the raw marker velocity's own
    # z-component, used as a multiplier for the 3rd velocity-pair/-component
    # outputs) is full-only.
    lines.append("    const double fill1 = weight * tmp_v[0] * ep_scale;")
    lines.append("    const double fill2 = weight * tmp_v[1] * ep_scale;")
    lines.append("    const double fill3 = weight * tmp_v[2] * ep_scale;")
    lines.append("    const double vx = v[0], vy = v[1]" + (", vz = v[2];" if full else ";"))
    lines.append("")
    lines.append("    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);")
    lines.append("    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);")
    lines.append("    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);")
    lines.append("    double bn1[MAXP+1], bd1[MAXP];")
    lines.append("    double bn2[MAXP+1], bd2[MAXP];")
    lines.append("    double bn3[MAXP+1], bd3[MAXP];")
    lines.append("    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);")
    lines.append("    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);")
    lines.append("    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);")
    lines.append("    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;")
    lines.append("")

    for sp in _SPATIAL_BLOCKS:
        row, col = _SPATIAL_BASIS[sp]
        row_deg, row_bas = _basis_args(row)
        col_deg, col_bas = _basis_args(col)
        mat_out = ", ".join(f"mat{sp}_{vel}" for vel in vel_pairs)
        dims = f"m{sp}_d2,m{sp}_d3,m{sp}_d4,m{sp}_d5,m{sp}_d6"
        fillmat = {"11": "fill11", "22": "fill22", "33": "fill33", "12": "fill12", "13": "fill13", "23": "fill23"}[sp]
        common = (
            f"{row_deg[0]},{row_deg[1]},{row_deg[2]}, {col_deg[0]},{col_deg[1]},{col_deg[2]}, "
            f"{row_bas[0]},{row_bas[1]},{row_bas[2]}, {col_bas[0]},{col_bas[1]},{col_bas[2]}, "
            f"span1,span2,span3, start0,start1,start2, p1,p2,p3"
        )
        if sp in _DIAG_SPATIAL_BLOCKS:
            mu = sp[0]
            vec_out = ", ".join(f"vec{mu}_{i}" for i in vec_is)
            vdims = f"v{mu}_n2,v{mu}_n3"
            fillvec = {"11": "fill1", "22": "fill2", "33": "fill3"}[sp]
            if full:
                lines.append(
                    f"    fill_mat_vec_pressure_full_dev({common},\n"
                    f"        {mat_out}, {dims}, {fillmat},\n"
                    f"        {vec_out}, {vdims}, {fillvec}, vx,vy,vz);"
                )
            else:
                lines.append(
                    f"    fill_mat_vec_pressure_dev({common},\n"
                    f"        {mat_out}, {dims}, {fillmat},\n"
                    f"        {vec_out}, {vdims}, {fillvec}, vx,vy);"
                )
        else:
            if full:
                lines.append(
                    f"    fill_mat_pressure_full_dev({common},\n        {mat_out}, {dims}, {fillmat}, vx,vy,vz);"
                )
            else:
                lines.append(f"    fill_mat_pressure_dev({common},\n        {mat_out}, {dims}, {fillmat}, vx,vy);")
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


_PC_LIN_MHD_6D_FULL_SRC = _build_pc_lin_mhd_6d_kernel_src(full=True)
_PC_LIN_MHD_6D_SRC = _build_pc_lin_mhd_6d_kernel_src(full=False)


def _pc_lin_mhd_6d_full_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PC_PRESSURE_FILLERS_SRC + _PC_LIN_MHD_6D_FULL_SRC


def _pc_lin_mhd_6d_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PC_PRESSURE_FILLERS_SRC + _PC_LIN_MHD_6D_SRC


_pc_lin_mhd_6d_full_kernel = None
_pc_lin_mhd_6d_kernel = None


def _get_pc_lin_mhd_6d_full_kernel():
    global _pc_lin_mhd_6d_full_kernel
    if _pc_lin_mhd_6d_full_kernel is None:
        import cupy as cp

        _pc_lin_mhd_6d_full_kernel = cp.RawKernel(_pc_lin_mhd_6d_full_source(), "pc_lin_mhd_6d_full_cuda")
    return _pc_lin_mhd_6d_full_kernel


def _get_pc_lin_mhd_6d_kernel():
    global _pc_lin_mhd_6d_kernel
    if _pc_lin_mhd_6d_kernel is None:
        import cupy as cp

        _pc_lin_mhd_6d_kernel = cp.RawKernel(_pc_lin_mhd_6d_source(), "pc_lin_mhd_6d_cuda")
    return _pc_lin_mhd_6d_kernel


def _pc_lin_mhd_6d_launch(
    kernel,
    markers,
    kind_map: int,
    params_dev,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    ep_scale: float,
    mat_args_45: dict,
    vec_args_45: dict,
    vel_pairs,
    vec_is,
):
    """Shared launch logic for pc_lin_mhd_6d_full_gpu/pc_lin_mhd_6d_gpu.
    ``mat_args_45``/``vec_args_45`` map every full-45-array name
    (``mat{sp}_{vel}`` / ``vec{mu}_{i}``) to its device array; only the
    subset named in ``vel_pairs``/``vec_is`` is actually passed to the
    kernel launch.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
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

    args = [
        dev_markers,
        np.int32(markers.shape[1]),
        np.int32(n_markers),
        np.int32(kind_map),
        params_dev,
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
        np.float64(ep_scale),
    ]
    for vel in vel_pairs:
        for sp in _SPATIAL_BLOCKS:
            args.append(mat_args_45[f"mat{sp}_{vel}"])
    for i in vec_is:
        for mu in ("1", "2", "3"):
            args.append(vec_args_45[f"vec{mu}_{i}"])
    for sp in _SPATIAL_BLOCKS:
        args.extend(dims(mat_args_45[f"mat{sp}_11"]))
    for mu in ("1", "2", "3"):
        v = vec_args_45[f"vec{mu}_1"]
        args.extend((np.int32(v.shape[1]), np.int32(v.shape[2])))

    kernel((blocks,), (threads,), tuple(args))


def pc_lin_mhd_6d_full_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    ep_scale: float,
    *mat_and_vec_args,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.pc_lin_mhd_6d_full`.
    ``mat_and_vec_args`` are the 45 output arrays in the exact positional
    order of the CPU kernel's signature (36 matrix blocks: velocity-pair
    outer -- xx, xy, xz, yy, yz, zz -- spatial-block inner -- 11, 12, 13,
    22, 23, 33; then 9 vector blocks: velocity-component outer -- x, y, z
    -- spatial-component inner -- 1, 2, 3), matching
    ``Accumulator._args_data``'s construction for ``symmetry="pressure"``.
    """
    mat_args_45 = {
        f"mat{sp}_{vel}": mat_and_vec_args[k]
        for k, (vel, sp) in enumerate((vel, sp) for vel in _SPATIAL_BLOCKS for sp in _SPATIAL_BLOCKS)
    }
    vec_args_45 = {
        f"vec{mu}_{i}": mat_and_vec_args[36 + k]
        for k, (i, mu) in enumerate((i, mu) for i in ("1", "2", "3") for mu in ("1", "2", "3"))
    }
    _pc_lin_mhd_6d_launch(
        _get_pc_lin_mhd_6d_full_kernel(),
        markers,
        kind_map,
        params_dev,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        ep_scale,
        mat_args_45,
        vec_args_45,
        _SPATIAL_BLOCKS,
        ("1", "2", "3"),
    )


def pc_lin_mhd_6d_gpu(
    markers,
    kind_map: int,
    params_dev,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    ep_scale: float,
    *mat_and_vec_args,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels.pc_lin_mhd_6d`. Same
    45-array positional convention as :func:`pc_lin_mhd_6d_full_gpu`
    (the propagator passes the identical 45-array signature for both), but
    -- matching the CPU reference exactly -- only the "perp" (x, y) subset
    (18 of the 36 matrix arrays, 6 of the 9 vector arrays) is ever written;
    the rest are left untouched (they stay at whatever
    ``Accumulator._accumulate``'s ``dat[:] = 0.0`` reset left them at).
    """
    mat_args_45 = {
        f"mat{sp}_{vel}": mat_and_vec_args[k]
        for k, (vel, sp) in enumerate((vel, sp) for vel in _SPATIAL_BLOCKS for sp in _SPATIAL_BLOCKS)
    }
    vec_args_45 = {
        f"vec{mu}_{i}": mat_and_vec_args[36 + k]
        for k, (i, mu) in enumerate((i, mu) for i in ("1", "2", "3") for mu in ("1", "2", "3"))
    }
    _pc_lin_mhd_6d_launch(
        _get_pc_lin_mhd_6d_kernel(),
        markers,
        kind_map,
        params_dev,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        ep_scale,
        mat_args_45,
        vec_args_45,
        ("11", "12", "22"),
        ("1", "2"),
    )
