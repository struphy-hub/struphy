"""Hand-written CUDA replacement for
:func:`~struphy.pic.accumulation.accum_kernels_gc.gc_mag_density_0form`,
used only under ``ARRAY_BACKEND=cupy``. See
:mod:`~struphy.pic.pushing.pusher_kernels_gc_cuda` for the scope of this
branch's 5D guiding-center porting (2 explicit pushers + this one
accumulator, out of 21 real kernels in the gc family).

Same atomicAdd-scatter approach as
:func:`~struphy.pic.accumulation.accum_kernels_cuda.charge_density_0form_gpu`
-- this kernel is nearly identical (an H^1/0-form vec_fill_b_v0 scatter),
just with a ``mu * weight * scale`` filling instead of a plain weight, and
``mu`` read from the marker's ``mu_idx`` column instead of a fixed offset.
"""

_GC_MAG_DENSITY_0FORM_SRC = r"""
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
void gc_mag_density_0form_cuda(
    const double* markers,
    const int n_cols,
    const int n_markers,
    const int mu_idx,
    const double scale,
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
    const double weight = row[5];
    const double mu = row[mu_idx];
    const double filling = mu * weight * scale;

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

_gc_mag_density_0form_kernel = None


def _get_gc_mag_density_0form_kernel():
    global _gc_mag_density_0form_kernel
    if _gc_mag_density_0form_kernel is None:
        import cupy as cp

        _gc_mag_density_0form_kernel = cp.RawKernel(_GC_MAG_DENSITY_0FORM_SRC, "gc_mag_density_0form_cuda")
    return _gc_mag_density_0form_kernel


def gc_mag_density_0form_gpu(
    markers,
    mu_idx: int,
    scale: float,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    vec_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels_gc.gc_mag_density_0form`.
    ``vec_dev`` is already device-resident and already zeroed by the caller.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev_markers = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_gc_mag_density_0form_kernel()(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(mu_idx),
            np.float64(scale),
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
# gc_density_0form is byte-for-byte the same computation as
# accum_kernels.charge_density_0form (an H^1/0-form vec_fill_b_v0 scatter with
# the marker weight as filling); only the docstring differs. Rather than
# duplicating the CUDA source, reuse the already-validated kernel.
# ---------------------------------------------------------------------------

from struphy.pic.accumulation.accum_kernels_cuda import (  # noqa: E402
    charge_density_0form_gpu as _charge_density_0form_gpu,
)


def gc_density_0form_gpu(markers, weight_idx, pn, tn1_dev, tn2_dev, tn3_dev, starts, vec_dev):
    """GPU replacement for
    :func:`~struphy.pic.accumulation.accum_kernels_gc.gc_density_0form`.

    Identical to
    :func:`~struphy.pic.accumulation.accum_kernels_cuda.charge_density_0form_gpu`
    (same filling, same 0-form scatter), so it simply delegates.
    """
    _charge_density_0form_gpu(markers, weight_idx, pn, tn1_dev, tn2_dev, tn3_dev, starts, vec_dev)


# ---------------------------------------------------------------------------
# cc_lin_mhd_5d_D: same 3-block antisymmetric V_u -> V_u fill as
# accum_kernels.cc_lin_mhd_6d_1 (runtime basis_u in {0,1,2} selecting
# H1vec/Hcurl/Hdiv, fill_mat_dev for each block), but the scalar prefactor is
# the guiding-centre density factor
#
#     -w_p * (1 - b_para/b*_para) * ep_scale / epsilon
#
# with b*_para = norm_b1 . (b2 + epsilon*v_par*curl_norm_b). It therefore needs
# a 1-form (norm_b1) and a second 2-form (curl_norm_b) evaluation on top of
# the B-field, but reuses fill_mat_dev from accum_kernels_cuda's
# _LINEAR_VLASOV_AMPERE_EXTRA_SRC unchanged.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_5D_D_SRC = r"""
extern "C" __global__
void cc_lin_mhd_5d_D_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const double epsilon, const double ep_scale,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int bb1_n2, const int bb1_n3,
    const double* b2_2, const int bb2_n2, const int bb2_n3,
    const double* b2_3, const int bb3_n2, const int bb3_n3,
    const double* nb11, const int nb1_n2, const int nb1_n3,
    const double* nb12, const int nb2_n2, const int nb2_n3,
    const double* nb13, const int nb3_n2, const int nb3_n3,
    const double* cnb1, const int cb1_n2, const int cb1_n3,
    const double* cnb2, const int cb2_n2, const int cb2_n3,
    const double* cnb3, const int cb3_n2, const int cb3_n3,
    const int basis_u,
    double* mat12, double* mat13, double* mat23,
    const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6,
    const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6,
    const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];
    const double weight = row[5];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b2_1,bb1_n2,bb1_n3, b2_2,bb2_n2,bb2_n3, b2_3,bb3_n2,bb3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb11,nb1_n2,nb1_n3, nb12,nb2_n2,nb2_n3, nb13,nb3_n2,nb3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,cb1_n2,cb1_n3, cnb2,cb2_n2,cb2_n3, cnb3,cb3_n2,cb3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + epsilon * v * curl_norm_b[k];

    const double b_para = dot3_dev(norm_b1, b);
    const double b_star_para = dot3_dev(norm_b1, b_star);
    const double density_const = 1.0 - b_para / b_star_para;

    const double pref = -weight * density_const * ep_scale / epsilon;
    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    double f12, f13, f23;

    if (basis_u == 0) {
        f12 = pref * b_prod[1];
        f13 = pref * b_prod[2];
        f23 = pref * b_prod[5];

        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);

    } else if (basis_u == 1) {
        double df_inv[9], g_inv[9];
        matrix_inv_dev(dfm, df_inv);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double sacc = 0.0;
                for (int k = 0; k < 3; k++) sacc += df_inv[3*i+k] * df_inv[3*j+k];
                g_inv[3*i+j] = sacc;
            }
        double tmp1[9], tmp2[9];
        matmat_dev(g_inv, b_prod, tmp1);
        matmat_dev(tmp1, g_inv, tmp2);

        f12 = pref * tmp2[1];
        f13 = pref * tmp2[2];
        f23 = pref * tmp2[5];

        fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);

    } else if (basis_u == 2) {
        const double det2 = det_df * det_df;
        f12 = pref * b_prod[1] / det2;
        f13 = pref * b_prod[2] / det2;
        f23 = pref * b_prod[5] / det2;

        fill_mat_dev(p1,pd2,pd3, pd1,p2,pd3, bn1,bd2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(p1,pd2,pd3, pd1,pd2,p3, bn1,bd2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(pd1,p2,pd3, pd1,pd2,p3, bd1,bn2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);
    }
}
"""

_cc_lin_mhd_5d_D_kernel = None


def _get_cc_lin_mhd_5d_D_kernel():
    global _cc_lin_mhd_5d_D_kernel
    if _cc_lin_mhd_5d_D_kernel is None:
        import cupy as cp

        from struphy.pic.accumulation.accum_kernels_cuda import _LINEAR_VLASOV_AMPERE_EXTRA_SRC
        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _cc_lin_mhd_5d_D_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_5D_D_SRC,
            "cc_lin_mhd_5d_D_cuda",
        )
    return _cc_lin_mhd_5d_D_kernel


def cc_lin_mhd_5d_D_gpu(
    markers, kind_map, params_dev, epsilon, ep_scale,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    b2, norm_b1, curl_norm_b, basis_u,
    mat12_dev, mat13_dev, mat23_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_D`.

    ``b2``/``norm_b1``/``curl_norm_b`` are 3-tuples of device-resident FE
    coefficient arrays; the ``mat*_dev`` are already zeroed by the caller.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    def dims(a):
        return (
            np.int32(a.shape[1]), np.int32(a.shape[2]), np.int32(a.shape[3]),
            np.int32(a.shape[4]), np.int32(a.shape[5]),
        )

    _get_cc_lin_mhd_5d_D_kernel()(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(kind_map), params_dev,
            np.float64(epsilon), np.float64(ep_scale),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(b2[0]), *d(b2[1]), *d(b2[2]),
            *d(norm_b1[0]), *d(norm_b1[1]), *d(norm_b1[2]),
            *d(curl_norm_b[0]), *d(curl_norm_b[1]), *d(curl_norm_b[2]),
            np.int32(basis_u),
            mat12_dev, mat13_dev, mat23_dev,
            *dims(mat12_dev), *dims(mat13_dev), *dims(mat23_dev),
        ),
    )
