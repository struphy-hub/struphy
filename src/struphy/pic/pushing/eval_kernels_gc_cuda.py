"""Hand-written CUDA replacement for
:func:`~struphy.pic.pushing.eval_kernels_gc.driftkinetic_hamiltonian`, used
only under ``ARRAY_BACKEND=cupy``.

This is the ``eval_kernel`` of the discrete-gradient guiding-centre
propagators: it is re-run on *every* Picard iteration of every RK stage (89
calls in a 3-step ``LinearMHDDriftkineticCC`` run), writing the Hamiltonian
at the weighted evaluation point into one marker column. With markers
device-resident, leaving it on the host would cost a full marker round trip
per iteration -- by far the most frequent host crossing left in that model.

It is a plain per-marker 0-form spline evaluation, so it reuses the shared
``find_span_dev``/``b_splines_dev``/``eval_0form_dev`` device functions.
"""

_DK_HAMILTONIAN_SRC = r"""
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

__device__ double eval_0form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bn2, const double* bn3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c, int n2x, int n3x)
{
    double out = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out += c[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3]
                     * bn1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    return out;
}

__device__ double mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void driftkinetic_hamiltonian_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr,
    const int first_init_idx, const int first_shift_idx, const int mu_idx,
    const double a0, const double a1, const double a2, const double a3,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* B_dot_b, const int b_n2, const int b_n3,
    const double* phi_c, const int p_n2, const int p_n3,
    const int evaluate_e_field)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double alpha[3] = {a0, a1, a2};
    double eta[3];
    for (int i = 0; i < 3; i++) {
        const double eta_k = row[i] + row[first_shift_idx + i];
        const double eta_n = row[first_init_idx + i];
        eta[i] = mod1_dev(alpha[i] * eta_k + (1.0 - alpha[i]) * eta_n);
    }

    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v = a3 * v_k + (1.0 - a3) * v_n;
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    b_splines_dev(tn1, p1, eta[0], span1, bn1);
    b_splines_dev(tn2, p2, eta[1], span2, bn2);
    b_splines_dev(tn3, p3, eta[2], span3, bn3);

    double phi = 0.0;
    if (evaluate_e_field) {
        phi = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi_c, p_n2, p_n3);
    }

    const double bdb = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, B_dot_b, b_n2, b_n3);

    row[column_nr] = epsilon * v * v / 2.0 + epsilon * mu * bdb + phi;
}
"""

_dk_kernel = None


def _get_dk_kernel():
    global _dk_kernel
    if _dk_kernel is None:
        import cupy as cp

        _dk_kernel = cp.RawKernel(_DK_HAMILTONIAN_SRC, "driftkinetic_hamiltonian_cuda")
    return _dk_kernel


def driftkinetic_hamiltonian_gpu(
    markers,
    alpha,
    column_nr,
    first_init_idx,
    first_shift_idx,
    mu_idx,
    args_derham,
    epsilon,
    B_dot_b_coeffs,
    phi_coeffs,
    evaluate_e_field,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.driftkinetic_hamiltonian`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    bdb = cp.ascontiguousarray(B_dot_b_coeffs)
    phi = cp.ascontiguousarray(phi_coeffs)
    a = [float(x) for x in (alpha[0], alpha[1], alpha[2], alpha[3])]

    _get_dk_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(mu_idx),
            np.float64(a[0]),
            np.float64(a[1]),
            np.float64(a[2]),
            np.float64(a[3]),
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            bdb,
            np.int32(bdb.shape[1]),
            np.int32(bdb.shape[2]),
            phi,
            np.int32(phi.shape[1]),
            np.int32(phi.shape[2]),
            np.int32(bool(evaluate_e_field)),
        ),
    )


# ---------------------------------------------------------------------------
# grad_driftkinetic_hamiltonian / bstar_parallel_3form / bstar_2form /
# unit_b_1form: the remaining marker-column init/eval kernels of the
# discrete-gradient guiding-centre propagators. Unlike driftkinetic_hamiltonian
# above (self-contained 0-form-only source), these also need 1-/2-form
# evaluation and (for bstar_parallel_3form) the domain Jacobian, so they are
# built from pusher_kernels_cuda._GENERAL_GEOMETRY_SRC instead. All four
# share the same alpha-weighted evaluation point
#   eta_i = mod(alpha_i * (eta_i + shift_i) + (1 - alpha_i) * eta_i^n, 1)
# (and, for the two that need v_parallel, the same alpha-weighted v), factored
# into one device helper.
# ---------------------------------------------------------------------------

_GC_MARKER_COLUMN_SRC = r"""
__device__ void weighted_eta_v_dev(
    const double* row, int first_init_idx, int first_shift_idx,
    const double* alpha, double* eta, double* v_out)
{
    for (int k = 0; k < 3; k++) {
        const double eta_k = row[k] + row[first_shift_idx + k];
        const double eta_n = row[first_init_idx + k];
        double e = alpha[k] * eta_k + (1.0 - alpha[k]) * eta_n;
        double r = fmod(e, 1.0);
        if (r < 0.0) r += 1.0;
        eta[k] = r;
    }
    if (v_out) {
        const double v_k = row[3];
        const double v_n = row[first_init_idx + 3];
        *v_out = alpha[3] * v_k + (1.0 - alpha[3]) * v_n;
    }
}

extern "C" __global__
void grad_driftkinetic_hamiltonian_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx, const int mu_idx,
    const double* alpha,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3];
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, 0);
    const double mu = row[mu_idx];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double grad_H[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,g1_n2,g1_n3, gb2,g2_n2,g2_n3, gb3,g3_n2,g3_n3, grad_H);
    for (int k = 0; k < 3; k++) grad_H[k] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            ef1,e1_n2,e1_n3, ef2,e2_n2,e2_n3, ef3,e3_n2,e3_n3, e_field);
        for (int k = 0; k < 3; k++) grad_H[k] -= e_field[k];
    }

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = grad_H[comps[j]];
}

extern "C" __global__
void bstar_parallel_3form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3], v;
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, &v);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta[0], eta[1], eta[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bn2[MAXP+1], bn3[MAXP+1];
    double bd1[MAXP], bd2[MAXP], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);

    b_star_parallel = (b_star_parallel * epsilon * v + B_dot_b) * det_df;

    row[column_nr] = b_star_parallel;
}

extern "C" __global__
void bstar_2form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b1, const int b1_n2, const int b1_n3,
    const double* b2, const int b2_n2, const int b2_n3,
    const double* b3, const int b3_n2, const int b3_n3,
    const double* cb1, const int c1_n2, const int c1_n3,
    const double* cb2, const int c2_n2, const int c2_n3,
    const double* cb3, const int c3_n2, const int c3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3], v;
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, &v);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double bb[3], b_star[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b1,b1_n2,b1_n3, b2,b2_n2,b2_n3, b3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cb1,c1_n2,c1_n3, cb2,c2_n2,c2_n3, cb3,c3_n2,c3_n3, b_star);

    for (int k = 0; k < 3; k++) b_star[k] = b_star[k] * epsilon * v + bb[k];

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = b_star[comps[j]];
}

extern "C" __global__
void unit_b_1form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* ub1, const int u1_n2, const int u1_n3,
    const double* ub2, const int u2_n2, const int u2_n3,
    const double* ub3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3];
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, 0);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double unit_b1[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ub1,u1_n2,u1_n3, ub2,u2_n2,u2_n3, ub3,u3_n2,u3_n3, unit_b1);

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = unit_b1[comps[j]];
}
"""

_gc_marker_column_kernels = {}


def _get_gc_marker_column_kernel(name):
    if name not in _gc_marker_column_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _gc_marker_column_kernels[name] = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _GC_MARKER_COLUMN_SRC, name
        )
    return _gc_marker_column_kernels[name]


def grad_driftkinetic_hamiltonian_gpu(
    markers, alpha, column_nr, comps, first_init_idx, first_shift_idx, mu_idx,
    args_derham, epsilon, grad_b_full_coeffs, e_field_coeffs, evaluate_e_field,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.grad_driftkinetic_hamiltonian`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("grad_driftkinetic_hamiltonian_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(column_nr), np.int32(comps_dev.shape[0]), comps_dev,
            np.int32(first_init_idx), np.int32(first_shift_idx), np.int32(mu_idx),
            alpha_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]), np.int32(args_derham.pn[1]), np.int32(args_derham.pn[2]),
            tn1, np.int32(tn1.shape[0]),
            tn2, np.int32(tn2.shape[0]),
            tn3, np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]), np.int32(args_derham.starts[1]), np.int32(args_derham.starts[2]),
            *d(grad_b_full_coeffs[0]), *d(grad_b_full_coeffs[1]), *d(grad_b_full_coeffs[2]),
            *d(e_field_coeffs[0]), *d(e_field_coeffs[1]), *d(e_field_coeffs[2]),
            np.int32(bool(evaluate_e_field)),
        ),
    )


def bstar_parallel_3form_gpu(
    markers, alpha, column_nr, first_init_idx, first_shift_idx,
    kind_map, params_dev, args_derham, epsilon, B_dot_b_coeffs, curl_unit_b_dot_b0_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.bstar_parallel_3form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("bstar_parallel_3form_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(column_nr),
            np.int32(first_init_idx), np.int32(first_shift_idx),
            alpha_dev,
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]), np.int32(args_derham.pn[1]), np.int32(args_derham.pn[2]),
            tn1, np.int32(tn1.shape[0]),
            tn2, np.int32(tn2.shape[0]),
            tn3, np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]), np.int32(args_derham.starts[1]), np.int32(args_derham.starts[2]),
            *d(B_dot_b_coeffs), *d(curl_unit_b_dot_b0_coeffs),
        ),
    )


def bstar_2form_gpu(
    markers, alpha, column_nr, comps, first_init_idx, first_shift_idx,
    args_derham, epsilon, b2_coeffs, curl_unit_b2_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.bstar_2form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("bstar_2form_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(column_nr), np.int32(comps_dev.shape[0]), comps_dev,
            np.int32(first_init_idx), np.int32(first_shift_idx),
            alpha_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]), np.int32(args_derham.pn[1]), np.int32(args_derham.pn[2]),
            tn1, np.int32(tn1.shape[0]),
            tn2, np.int32(tn2.shape[0]),
            tn3, np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]), np.int32(args_derham.starts[1]), np.int32(args_derham.starts[2]),
            *d(b2_coeffs[0]), *d(b2_coeffs[1]), *d(b2_coeffs[2]),
            *d(curl_unit_b2_coeffs[0]), *d(curl_unit_b2_coeffs[1]), *d(curl_unit_b2_coeffs[2]),
        ),
    )


def unit_b_1form_gpu(
    markers, alpha, column_nr, comps, first_init_idx, first_shift_idx,
    args_derham, unit_b1_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.unit_b_1form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("unit_b_1form_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(column_nr), np.int32(comps_dev.shape[0]), comps_dev,
            np.int32(first_init_idx), np.int32(first_shift_idx),
            alpha_dev,
            np.int32(args_derham.pn[0]), np.int32(args_derham.pn[1]), np.int32(args_derham.pn[2]),
            tn1, np.int32(tn1.shape[0]),
            tn2, np.int32(tn2.shape[0]),
            tn3, np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]), np.int32(args_derham.starts[1]), np.int32(args_derham.starts[2]),
            *d(unit_b1_coeffs[0]), *d(unit_b1_coeffs[1]), *d(unit_b1_coeffs[2]),
        ),
    )
