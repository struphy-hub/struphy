"""Hand-written CUDA replacements for select 5D guiding-center pusher
kernels in :mod:`~struphy.pic.pushing.pusher_kernels_gc`, used only under
``ARRAY_BACKEND=cupy``.

Scope: this branch's 6D (full-orbit) work ported every real (non-dead-code)
kernel in ``pusher_kernels.py``/``accum_kernels.py``. The 5D guiding-center
family (``pusher_kernels_gc.py``/``accum_kernels_gc.py``) is a separate,
much larger body of kernels -- 15 pushers + 8 accumulators, 21 of them with
real propagator callers -- several of which (the ``*_discrete_gradient_*``
variants) are implicit per-marker Newton solves, not simple explicit RK
stages, and are a substantially bigger porting effort.

Currently covered here: the two *explicit* multistage GC pushers,
:func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_explicit_multistage`
and :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_explicit_multistage`
-- both are plain explicit-RK marker loops (same ``dt*a[stage]``/``dt*b[stage]``/
``last`` structure as ``push_eta_stage`` in the 6D family), and reuse the
existing 0-/1-/2-form spline evaluation and geometry device functions from
:mod:`~struphy.pic.pushing.pusher_kernels_cuda`'s ``_GENERAL_GEOMETRY_SRC``
unchanged. The accompanying accumulation kernel
:func:`~struphy.pic.accumulation.accum_kernels_gc.gc_mag_density_0form` is
ported alongside these in
:mod:`~struphy.pic.accumulation.accum_kernels_gc_cuda` (same
``atomicAdd``-scatter approach as ``charge_density_0form``).
"""

_PUSH_GC_BXESTAR_SRC = r"""
extern "C" __global__
void push_gc_bxEstar_explicit_multistage_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* unit_b1_1, const int ub1_n2, const int ub1_n3,
    const double* unit_b1_2, const int ub2_n2, const int ub2_n3,
    const double* unit_b1_3, const int ub3_n2, const int ub3_n3,
    const double* grad_b_full_1, const int gb1_n2, const int gb1_n3,
    const double* grad_b_full_2, const int gb2_n2, const int gb2_n3,
    const double* grad_b_full_3, const int gb3_n2, const int gb3_n3,
    const double* B_dot_b_coeffs, const int bdb_n2, const int bdb_n3,
    const double* curl_unit_b_dot_b0, const int cub_n2, const int cub_n3,
    const double* e_field_1, const int e1_n2, const int e1_n3,
    const double* e_field_2, const int e2_n2, const int e2_n3,
    const double* e_field_3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt_a, const double dt_b, const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];
    const double mu = row[mu_idx];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double unit_b1[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        unit_b1_1, ub1_n2, ub1_n3, unit_b1_2, ub2_n2, ub2_n3, unit_b1_3, ub3_n2, ub3_n3, unit_b1);

    double e_star[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        grad_b_full_1, gb1_n2, gb1_n3, grad_b_full_2, gb2_n2, gb2_n3, grad_b_full_3, gb3_n2, gb3_n3, e_star);
    e_star[0] *= -epsilon * mu;
    e_star[1] *= -epsilon * mu;
    e_star[2] *= -epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
            start0, start1, start2,
            e_field_1, e1_n2, e1_n3, e_field_2, e2_n2, e2_n3, e_field_3, e3_n2, e3_n3, e_field);
        e_star[0] += e_field[0];
        e_star[1] += e_field[1];
        e_star[2] += e_field[2];
    }

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, B_dot_b_coeffs, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, curl_unit_b_dot_b0, cub_n2, cub_n3);
    b_star_parallel = b_star_parallel * epsilon * v + B_dot_b;
    b_star_parallel *= det_df;

    double Exb[3];
    cross_dev(e_star, unit_b1, Exb);

    double k[3];
    k[0] = Exb[0] / b_star_parallel;
    k[1] = Exb[1] / b_star_parallel;
    k[2] = Exb[2] / b_star_parallel;

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}
"""

_PUSH_GC_BSTAR_SRC = r"""
extern "C" __global__
void push_gc_Bstar_explicit_multistage_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* grad_b_full_1, const int gb1_n2, const int gb1_n3,
    const double* grad_b_full_2, const int gb2_n2, const int gb2_n3,
    const double* grad_b_full_3, const int gb3_n2, const int gb3_n3,
    const double* b2_1, const int b1_n2, const int b1_n3,
    const double* b2_2, const int b2n2, const int b2n3,
    const double* b2_3, const int b3_n2, const int b3_n3,
    const double* curl_unit_b2_1, const int cb1_n2, const int cb1_n3,
    const double* curl_unit_b2_2, const int cb2_n2, const int cb2_n3,
    const double* curl_unit_b2_3, const int cb3_n2, const int cb3_n3,
    const double* B_dot_b_coeffs, const int bdb_n2, const int bdb_n3,
    const double* curl_unit_b_dot_b0, const int cub_n2, const int cub_n3,
    const double* e_field_1, const int e1_n2, const int e1_n3,
    const double* e_field_2, const int e2_n2, const int e2_n3,
    const double* e_field_3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt_a, const double dt_b, const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];
    const double mu = row[mu_idx];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double e_star[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        grad_b_full_1, gb1_n2, gb1_n3, grad_b_full_2, gb2_n2, gb2_n3, grad_b_full_3, gb3_n2, gb3_n3, e_star);
    e_star[0] *= -epsilon * mu;
    e_star[1] *= -epsilon * mu;
    e_star[2] *= -epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
            start0, start1, start2,
            e_field_1, e1_n2, e1_n3, e_field_2, e2_n2, e2_n3, e_field_3, e3_n2, e3_n3, e_field);
        e_star[0] += e_field[0];
        e_star[1] += e_field[1];
        e_star[2] += e_field[2];
    }

    double b2[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        b2_1, b1_n2, b1_n3, b2_2, b2n2, b2n3, b2_3, b3_n2, b3_n3, b2);

    double b_star[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        curl_unit_b2_1, cb1_n2, cb1_n3, curl_unit_b2_2, cb2_n2, cb2_n3, curl_unit_b2_3, cb3_n2, cb3_n3, b_star);
    b_star[0] = b_star[0] * epsilon * v + b2[0];
    b_star[1] = b_star[1] * epsilon * v + b2[1];
    b_star[2] = b_star[2] * epsilon * v + b2[2];

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, B_dot_b_coeffs, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, curl_unit_b_dot_b0, cub_n2, cub_n3);
    b_star_parallel = b_star_parallel * epsilon * v + B_dot_b;
    b_star_parallel *= det_df;

    double k[3];
    k[0] = b_star[0] / b_star_parallel * v;
    k[1] = b_star[1] / b_star_parallel * v;
    k[2] = b_star[2] / b_star_parallel * v;

    double k_v = dot3_dev(b_star, e_star);
    k_v /= b_star_parallel * epsilon;

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];
    row[first_free_idx + 3] += dt_b * k_v;

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
    row[3] = row[first_init_idx + 3] + dt_a * k_v + last * row[first_free_idx + 3];
}
"""


def _push_gc_bxEstar_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PUSH_GC_BXESTAR_SRC


def _push_gc_Bstar_source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

    return _GENERAL_GEOMETRY_SRC + _PUSH_GC_BSTAR_SRC


_push_gc_bxEstar_kernel = None
_push_gc_Bstar_kernel = None


def _get_push_gc_bxEstar_kernel():
    global _push_gc_bxEstar_kernel
    if _push_gc_bxEstar_kernel is None:
        import cupy as cp

        _push_gc_bxEstar_kernel = cp.RawKernel(_push_gc_bxEstar_source(), "push_gc_bxEstar_explicit_multistage_cuda")
    return _push_gc_bxEstar_kernel


def _get_push_gc_Bstar_kernel():
    global _push_gc_Bstar_kernel
    if _push_gc_Bstar_kernel is None:
        import cupy as cp

        _push_gc_Bstar_kernel = cp.RawKernel(_push_gc_Bstar_source(), "push_gc_Bstar_explicit_multistage_cuda")
    return _push_gc_Bstar_kernel


def push_gc_bxEstar_explicit_multistage_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    mu_idx,
    kind_map,
    params_dev,
    epsilon,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    unit_b1_1_dev,
    unit_b1_2_dev,
    unit_b1_3_dev,
    grad_b_full_1_dev,
    grad_b_full_2_dev,
    grad_b_full_3_dev,
    B_dot_b_coeffs_dev,
    curl_unit_b_dot_b0_dev,
    e_field_1_dev,
    e_field_2_dev,
    e_field_3_dev,
    evaluate_e_field: bool,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_explicit_multistage`,
    for any domain in :data:`~struphy.pic.pushing.pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS`.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_push_gc_bxEstar_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.int32(mu_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
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
            *d(unit_b1_1_dev),
            *d(unit_b1_2_dev),
            *d(unit_b1_3_dev),
            *d(grad_b_full_1_dev),
            *d(grad_b_full_2_dev),
            *d(grad_b_full_3_dev),
            *d(B_dot_b_coeffs_dev),
            *d(curl_unit_b_dot_b0_dev),
            *d(e_field_1_dev),
            *d(e_field_2_dev),
            *d(e_field_3_dev),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


def push_gc_Bstar_explicit_multistage_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    mu_idx,
    kind_map,
    params_dev,
    epsilon,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    grad_b_full_1_dev,
    grad_b_full_2_dev,
    grad_b_full_3_dev,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    curl_unit_b2_1_dev,
    curl_unit_b2_2_dev,
    curl_unit_b2_3_dev,
    B_dot_b_coeffs_dev,
    curl_unit_b_dot_b0_dev,
    e_field_1_dev,
    e_field_2_dev,
    e_field_3_dev,
    evaluate_e_field: bool,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_explicit_multistage`,
    for any domain in :data:`~struphy.pic.pushing.pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS`.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_push_gc_Bstar_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.int32(mu_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
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
            *d(grad_b_full_1_dev),
            *d(grad_b_full_2_dev),
            *d(grad_b_full_3_dev),
            *d(b2_1_dev),
            *d(b2_2_dev),
            *d(b2_3_dev),
            *d(curl_unit_b2_1_dev),
            *d(curl_unit_b2_2_dev),
            *d(curl_unit_b2_3_dev),
            *d(B_dot_b_coeffs_dev),
            *d(curl_unit_b_dot_b0_dev),
            *d(e_field_1_dev),
            *d(e_field_2_dev),
            *d(e_field_3_dev),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


# ---------------------------------------------------------------------------
# Discrete-gradient (implicit) guiding-centre pushers.
#
# Despite the name these are NOT internally iterative: each call performs one
# Picard iteration, and the outer fixed-point loop lives in
# Pusher._push (the ``while`` over ``maxiter``/``tol``). So they are just as
# per-marker parallel as the explicit multistage pushers, and the residual
# each marker writes to ``residual_idx`` is what drives the outer loop.
#
# They need no domain Jacobian at all: the Poisson-matrix pieces
# (b_star_parallel, unit_b1 / b_star) are precomputed into marker columns by
# the propagator's init/eval kernels, so only a 1-form spline evaluation of
# grad|B| (and optionally E) at the midpoint is done here.
# ---------------------------------------------------------------------------

_DG_1ST_SRC = r"""
// mod(x, 1.0) matching numpy (result in [0, 1))
__device__ double mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void push_gc_bxEstar_discrete_gradient_1st_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int gb1_n2, const int gb1_n3,
    const double* gb2, const int gb2_n2, const int gb2_n3,
    const double* gb3, const int gb3_n2, const int gb3_n3,
    const double* e1c, const int e1_n2, const int e1_n3,
    const double* e2c, const int e2_n2, const int e2_n3,
    const double* e3c, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int i = 0; i < 3; i++) {
        eta_k[i] = row[i] + row[first_shift_idx + i];
        eta_n[i] = row[first_init_idx + i];
        eta_mid[i] = mod1_dev((eta_k[i] + eta_n[i]) / 2.0);
        eta_diff[i] = eta_k[i] - eta_n[i];
    }

    const double mu = row[mu_idx];
    const double H_n = row[first_free_idx];
    const double b_star_parallel = row[first_free_idx + 1];
    double unit_b1[3] = {
        row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k = row[first_free_idx + 5];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double grad_H[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,gb1_n2,gb1_n3, gb2,gb2_n2,gb2_n3, gb3,gb3_n2,gb3_n3, grad_H);
    for (int i = 0; i < 3; i++) grad_H[i] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            e1c,e1_n2,e1_n3, e2c,e2_n2,e2_n3, e3c,e3_n2,e3_n3, e_field);
        for (int i = 0; i < 3; i++) grad_H[i] += -e_field[i];
    }

    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H);
    const double dZ_squared = dot3_dev(eta_diff, eta_diff);

    double grad_I[3];
    if (dZ_squared == 0.0) {
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i];
    } else {
        const double c = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i] + eta_diff[i] * c;
    }

    double Exb[3];
    cross_dev(unit_b1, grad_I, Exb);

    double k[3];
    for (int i = 0; i < 3; i++) k[i] = Exb[i] / b_star_parallel;

    for (int i = 0; i < 3; i++) row[i] = eta_n[i] + dt * k[i];

    row[residual_idx] = sqrt(
        (row[0] - eta_k[0]) * (row[0] - eta_k[0])
      + (row[1] - eta_k[1]) * (row[1] - eta_k[1])
      + (row[2] - eta_k[2]) * (row[2] - eta_k[2]));
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_1st_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int gb1_n2, const int gb1_n3,
    const double* gb2, const int gb2_n2, const int gb2_n3,
    const double* gb3, const int gb3_n2, const int gb3_n3,
    const double* e1c, const int e1_n2, const int e1_n3,
    const double* e2c, const int e2_n2, const int e2_n3,
    const double* e3c, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int i = 0; i < 3; i++) {
        eta_k[i] = row[i] + row[first_shift_idx + i];
        eta_n[i] = row[first_init_idx + i];
        eta_mid[i] = mod1_dev((eta_k[i] + eta_n[i]) / 2.0);
        eta_diff[i] = eta_k[i] - eta_n[i];
    }

    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_mid = (v_k + v_n) / 2.0;
    const double v_diff = v_k - v_n;

    const double mu = row[mu_idx];
    const double H_n = row[first_free_idx];
    const double b_star_parallel = epsilon * row[first_free_idx + 1];
    double b_star[3] = {
        row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k = row[first_free_idx + 5];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double grad_H[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,gb1_n2,gb1_n3, gb2,gb2_n2,gb2_n3, gb3,gb3_n2,gb3_n3, grad_H);
    for (int i = 0; i < 3; i++) grad_H[i] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            e1c,e1_n2,e1_n3, e2c,e2_n2,e2_n3, e3c,e3_n2,e3_n3, e_field);
        for (int i = 0; i < 3; i++) grad_H[i] += -e_field[i];
    }

    const double grad_H_v = epsilon * v_mid;
    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H) + v_diff * grad_H_v;
    const double dZ_squared = dot3_dev(eta_diff, eta_diff) + v_diff * v_diff;

    double grad_I[3];
    double grad_I_v;
    if (dZ_squared == 0.0) {
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i];
        grad_I_v = grad_H_v;
    } else {
        const double c = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i] + eta_diff[i] * c;
        grad_I_v = grad_H_v + v_diff * c;
    }

    double k[3];
    for (int i = 0; i < 3; i++) k[i] = b_star[i] / b_star_parallel * grad_I_v;

    double k_v = dot3_dev(b_star, grad_I);
    k_v /= -b_star_parallel;

    for (int i = 0; i < 3; i++) row[i] = eta_n[i] + dt * k[i];
    row[3] = v_n + dt * k_v;

    row[residual_idx] = sqrt(
        (row[0] - eta_k[0]) * (row[0] - eta_k[0])
      + (row[1] - eta_k[1]) * (row[1] - eta_k[1])
      + (row[2] - eta_k[2]) * (row[2] - eta_k[2])
      + ((row[3] - v_k) / v_k) * ((row[3] - v_k) / v_k));
}
"""

_dg_kernels = {}


def _get_dg_kernel(name):
    if name not in _dg_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _dg_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_1ST_SRC, name)
    return _dg_kernels[name]


def _dg_launch(
    name,
    markers,
    n_cols,
    first_init_idx,
    first_shift_idx,
    residual_idx,
    first_free_idx,
    mu_idx,
    epsilon,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    grad_b_full,
    e_field,
    evaluate_e_field,
    dt,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_dg_kernel(name)(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(residual_idx),
            np.int32(first_free_idx),
            np.int32(mu_idx),
            np.float64(epsilon),
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
            *d(grad_b_full[0]),
            *d(grad_b_full[1]),
            *d(grad_b_full[2]),
            *d(e_field[0]),
            *d(e_field[1]),
            *d(e_field[2]),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt),
        ),
    )


def push_gc_bxEstar_discrete_gradient_1st_order_gpu(*args, **kwargs):
    """GPU replacement for one Picard iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order`."""
    _dg_launch("push_gc_bxEstar_discrete_gradient_1st_order_cuda", *args, **kwargs)


def push_gc_Bstar_discrete_gradient_1st_order_gpu(*args, **kwargs):
    """GPU replacement for one Picard iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order`."""
    _dg_launch("push_gc_Bstar_discrete_gradient_1st_order_cuda", *args, **kwargs)


# ---------------------------------------------------------------------------
# push_gc_cc_J1_{H1vec,Hcurl,Hdiv}: single-stage (dt, no Butcher coefficients
# -- `stage` is accepted but unused by the CPU kernels too) velocity update
# for CurrentCoupling5DCurlb. All three read the same fields (b, norm_b1,
# curl_norm_b) and differ only in which FEEC space `u` lives in and how it
# is transformed to Cartesian:
#   H1vec: u is already a vector field (eval_vectorfield_dev), no transform
#   Hcurl: u is a 1-form; transform via g^-1 = (DF^T DF)^-1
#   Hdiv:  u is a 2-form (like b); transform by dividing by det(DF)
# ---------------------------------------------------------------------------

_PUSH_GC_CC_J1_SRC = r"""
extern "C" __global__
void push_gc_cc_J1_H1vec_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;

    double b[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_vectorfield_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u, e);
    const double temp = dot3_dev(e, curl_norm_b);

    row[3] += temp / abs_b_star_para * v * dt;
}

extern "C" __global__
void push_gc_cc_J1_Hcurl_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b[3], u_form[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u_form);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    // g_inv = (DF^T DF)^-1, transforms the 1-form u into H1vec components
    double df_t[9] = {
        dfm[0], dfm[3], dfm[6],
        dfm[1], dfm[4], dfm[7],
        dfm[2], dfm[5], dfm[8],
    };
    double g[9], g_inv[9], u0[3];
    matmat_dev(df_t, dfm, g);
    matrix_inv_dev(g, g_inv);
    matvec_dev(g_inv, u_form, u0);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = (b[k] + curl_norm_b[k] * v * epsilon) / det_df;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u0, e);
    const double temp = dot3_dev(e, curl_norm_b) / det_df;

    row[3] += temp / abs_b_star_para * v * dt;
}

extern "C" __global__
void push_gc_cc_J1_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    for (int k = 0; k < 3; k++) u[k] /= det_df;

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u, e);
    const double temp = dot3_dev(e, curl_norm_b);

    row[3] += temp / abs_b_star_para * v * dt;
}
"""

_j1_kernels = {}


def _get_j1_kernel(name):
    if name not in _j1_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _j1_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _PUSH_GC_CC_J1_SRC, name)
    return _j1_kernels[name]


def _j1_launch(
    name,
    markers,
    kind_map,
    params_dev,
    epsilon,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    norm_b1,
    curl_norm_b,
    u,
    dt,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_j1_kernel(name)(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.float64(dt),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
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
            *d(b2[0]),
            *d(b2[1]),
            *d(b2[2]),
            *d(norm_b1[0]),
            *d(norm_b1[1]),
            *d(norm_b1[2]),
            *d(curl_norm_b[0]),
            *d(curl_norm_b[1]),
            *d(curl_norm_b[2]),
            *d(u[0]),
            *d(u[1]),
            *d(u[2]),
        ),
    )


def push_gc_cc_J1_H1vec_gpu(*args, **kwargs):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J1_H1vec`."""
    _j1_launch("push_gc_cc_J1_H1vec_cuda", *args, **kwargs)


def push_gc_cc_J1_Hcurl_gpu(*args, **kwargs):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J1_Hcurl`."""
    _j1_launch("push_gc_cc_J1_Hcurl_cuda", *args, **kwargs)


def push_gc_cc_J1_Hdiv_gpu(*args, **kwargs):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J1_Hdiv`."""
    _j1_launch("push_gc_cc_J1_Hdiv_cuda", *args, **kwargs)


# ---------------------------------------------------------------------------
# push_gc_cc_J2_stage_{H1vec,Hdiv}: multistage (a[stage]/b[stage]/last, same
# first_init_idx/first_free_idx accumulation scheme as
# push_gc_bxEstar_explicit_multistage above) position update for
# CurrentCoupling5DGradB. Both build the same b_prod/norm_b_prod
# cross-product matrices and e = (norm_b_prod @ b_prod @ u) / |B*_para|;
# H1vec evaluates u as a vector field and stops there (its DF/det(DF) are
# computed by the CPU reference but never used); Hdiv evaluates u as a
# 2-form and divides e by det(DF) as well.
# ---------------------------------------------------------------------------

_PUSH_GC_CC_J2_STAGE_SRC = r"""
extern "C" __global__
void push_gc_cc_J2_stage_H1vec_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_free_idx,
    const double dt_a, const double dt_b, const double last,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double bb[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_vectorfield_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    for (int k = 0; k < 3; k++) e[k] /= abs_b_star_para;

    row[first_free_idx + 0] -= dt_b * e[0];
    row[first_free_idx + 1] -= dt_b * e[1];
    row[first_free_idx + 2] -= dt_b * e[2];

    row[0] = row[first_init_idx + 0] - dt_a * e[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] - dt_a * e[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] - dt_a * e[2] + last * row[first_free_idx + 2];
}

extern "C" __global__
void push_gc_cc_J2_stage_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_free_idx,
    const double dt_a, const double dt_b, const double last,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double bb[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    for (int k = 0; k < 3; k++) e[k] /= (abs_b_star_para * det_df);

    row[first_free_idx + 0] -= dt_b * e[0];
    row[first_free_idx + 1] -= dt_b * e[1];
    row[first_free_idx + 2] -= dt_b * e[2];

    row[0] = row[first_init_idx + 0] - dt_a * e[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] - dt_a * e[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] - dt_a * e[2] + last * row[first_free_idx + 2];
}
"""

_j2_stage_kernels = {}


def _get_j2_stage_kernel(name):
    if name not in _j2_stage_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _j2_stage_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _PUSH_GC_CC_J2_STAGE_SRC, name)
    return _j2_stage_kernels[name]


def _j2_stage_launch(
    name,
    markers,
    first_init_idx,
    first_free_idx,
    kind_map,
    params_dev,
    epsilon,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    norm_b1,
    curl_norm_b,
    u,
    dt_a,
    dt_b,
    last,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_j2_stage_kernel(name)(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
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
            *d(b2[0]),
            *d(b2[1]),
            *d(b2[2]),
            *d(norm_b1[0]),
            *d(norm_b1[1]),
            *d(norm_b1[2]),
            *d(curl_norm_b[0]),
            *d(curl_norm_b[1]),
            *d(curl_norm_b[2]),
            *d(u[0]),
            *d(u[1]),
            *d(u[2]),
        ),
    )


def push_gc_cc_J2_stage_H1vec_gpu(*args, **kwargs):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J2_stage_H1vec`."""
    _j2_stage_launch("push_gc_cc_J2_stage_H1vec_cuda", *args, **kwargs)


def push_gc_cc_J2_stage_Hdiv_gpu(*args, **kwargs):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J2_stage_Hdiv`."""
    _j2_stage_launch("push_gc_cc_J2_stage_Hdiv_cuda", *args, **kwargs)


# ---------------------------------------------------------------------------
# push_gc_cc_J2_dg_init_Hdiv / push_gc_cc_J2_dg_Hdiv: the discrete-gradient
# variant of CurrentCoupling5DGradB's position push. Both are single-pass
# marker loops (no per-marker Newton solve -- the outer fixed-point loop in
# CurrentCoupling5DGradB.__call__ is over one global scalar `const` from an
# energy reduction, recomputed and re-applied to all markers each iteration).
#   dg_init: like push_gc_cc_J2_stage_Hdiv's single-stage core, evaluated at
#            the current position, straight `eta -= dt*e`.
#   dg:      evaluated at the midpoint eta_mid = mod((eta+eta_init)/2, 1),
#            with a second U-field `ud` (discrete-gradient correction term,
#            scaled by `const`) added before the same |B*_para|/det(DF)
#            division, then `eta = alpha*(eta_init - dt*e) + (1-alpha)*eta_old`.
# ---------------------------------------------------------------------------

_PUSH_GC_CC_J2_DG_SRC = r"""
extern "C" __global__
void push_gc_cc_J2_dg_init_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double bb[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    for (int k = 0; k < 3; k++) e[k] /= (abs_b_star_para * det_df);

    row[0] -= dt * e[0];
    row[1] -= dt * e[1];
    row[2] -= dt * e[2];
}

extern "C" __global__
void push_gc_cc_J2_dg_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx,
    const double dt, const double const_, const double alpha,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3,
    const double* ud_1, const int ud1_n2, const int ud1_n3,
    const double* ud_2, const int ud2_n2, const int ud2_n3,
    const double* ud_3, const int ud3_n2, const int ud3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta_old0 = row[0], eta_old1 = row[1], eta_old2 = row[2];
    double eta_mid[3];
    eta_mid[0] = mod1_dev((row[0] + row[first_init_idx + 0]) / 2.0);
    eta_mid[1] = mod1_dev((row[1] + row[first_init_idx + 1]) / 2.0);
    eta_mid[2] = mod1_dev((row[2] + row[first_init_idx + 2]) / 2.0);
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double bb[3], u[3], ud[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ud_1,ud1_n2,ud1_n3, ud_2,ud2_n2,ud2_n3, ud_3,ud3_n2,ud3_n3, ud);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3], e2[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    matvec_dev(tmp, ud, e2);
    for (int k = 0; k < 3; k++) e[k] = (e[k] + const_ * e2[k]) / (abs_b_star_para * det_df);

    double eta_new[3];
    eta_new[0] = row[first_init_idx + 0] - dt * e[0];
    eta_new[1] = row[first_init_idx + 1] - dt * e[1];
    eta_new[2] = row[first_init_idx + 2] - dt * e[2];

    row[0] = alpha * eta_new[0] + (1.0 - alpha) * eta_old0;
    row[1] = alpha * eta_new[1] + (1.0 - alpha) * eta_old1;
    row[2] = alpha * eta_new[2] + (1.0 - alpha) * eta_old2;
}
"""

_j2_dg_kernels = {}


def _get_j2_dg_kernel(name):
    if name not in _j2_dg_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _j2_dg_kernels[name] = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _DG_1ST_SRC + _PUSH_GC_CC_J2_DG_SRC, name
        )
    return _j2_dg_kernels[name]


def push_gc_cc_J2_dg_init_Hdiv_gpu(
    markers, first_init_idx, kind_map, params_dev, epsilon,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    b2, norm_b1, curl_norm_b, u, dt,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J2_dg_init_Hdiv`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_j2_dg_kernel("push_gc_cc_J2_dg_init_Hdiv_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx),
            np.float64(dt),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(b2[0]), *d(b2[1]), *d(b2[2]),
            *d(norm_b1[0]), *d(norm_b1[1]), *d(norm_b1[2]),
            *d(curl_norm_b[0]), *d(curl_norm_b[1]), *d(curl_norm_b[2]),
            *d(u[0]), *d(u[1]), *d(u[2]),
        ),
    )


def push_gc_cc_J2_dg_Hdiv_gpu(
    markers, first_init_idx, kind_map, params_dev, epsilon,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    b2, norm_b1, curl_norm_b, u, ud, const, alpha, dt,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_cc_J2_dg_Hdiv`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_j2_dg_kernel("push_gc_cc_J2_dg_Hdiv_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx),
            np.float64(dt), np.float64(const), np.float64(alpha),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(b2[0]), *d(b2[1]), *d(b2[2]),
            *d(norm_b1[0]), *d(norm_b1[1]), *d(norm_b1[2]),
            *d(curl_norm_b[0]), *d(curl_norm_b[1]), *d(curl_norm_b[2]),
            *d(u[0]), *d(u[1]), *d(u[2]),
            *d(ud[0]), *d(ud[1]), *d(ud[2]),
        ),
    )


# ---------------------------------------------------------------------------
# push_gc_bxEstar_discrete_gradient_1st_order_newton /
# push_gc_Bstar_discrete_gradient_1st_order_newton: one Newton iteration
# (per marker, so per-marker parallel like the non-Newton 1st_order variants
# above) for the Itoh-Abe discrete-gradient guiding-centre pushers. Unlike
# the *_1st_order Picard kernels, these read a richer set of pre-evaluated
# marker columns (the Hamiltonian and its gradient at several points along
# the coordinate axes, written by driftkinetic_hamiltonian/
# grad_driftkinetic_hamiltonian eval_kernels -- both already CUDA-ported
# above) and solve one 3x3 (bxEstar) or 4x4 (Bstar, via Schur complement of
# its [[I,B],[C,1]] block structure) Newton step in closed form; no domain
# Jacobian is needed. Purely marker-local, no shared per-marker helper beyond
# what's already in _GENERAL_GEOMETRY_SRC.
# ---------------------------------------------------------------------------

_DG_NEWTON_SRC = r"""
extern "C" __global__
void push_gc_bxEstar_discrete_gradient_1st_order_newton_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const double* phi, const int p_n2, const int p_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        const double eta_k_shifted = row[k] + row[first_shift_idx + k];
        eta_k[k] = row[k];
        eta_diff[k] = eta_k_shifted - row[first_init_idx + k];
    }
    const double v = row[3];
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double b_star_parallel = row[first_free_idx + 1];
    const double unit_b1[3] = {row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k1 = row[first_free_idx + 5];
    const double H_k12 = row[first_free_idx + 6];
    const double grad_H_1 = row[first_free_idx + 7];
    const double grad_H_12[2] = {row[first_free_idx + 8], row[first_free_idx + 9]};

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_k[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_k[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_k[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_k[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_k[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_k[2], span3, bn3, bd3);

    double phi_val = 0.0;
    if (evaluate_e_field) {
        phi_val = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi, p_n2, p_n3);
    }
    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    const double H_k = epsilon * v * v / 2.0 + epsilon * mu * B_dot_b + phi_val;

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

    double grad_I[3];
    grad_I[0] = (eta_diff[0] == 0.0) ? grad_H[0] : (H_k1 - H_n) / eta_diff[0];
    grad_I[1] = (eta_diff[1] == 0.0) ? grad_H[1] : (H_k12 - H_k1) / eta_diff[1];
    grad_I[2] = (eta_diff[2] == 0.0) ? grad_H[2] : (H_k - H_k12) / eta_diff[2];

    double bcross_mat[9] = {
        0.0, -unit_b1[2], unit_b1[1],
        unit_b1[2], 0.0, -unit_b1[0],
        -unit_b1[1], unit_b1[0], 0.0};
    for (int k = 0; k < 9; k++) bcross_mat[k] /= b_star_parallel;

    double func[3];
    matvec_dev(bcross_mat, grad_I, func);
    for (int k = 0; k < 3; k++) func[k] = eta_diff[k] - dt * func[k];

    double Ddg[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (eta_diff[0] != 0.0) Ddg[0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / (eta_diff[0] * eta_diff[0]);
    if (eta_diff[1] != 0.0) {
        Ddg[4] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / (eta_diff[1] * eta_diff[1]);
        Ddg[3] = (grad_H_12[0] - grad_H_1) / eta_diff[1];
    }
    if (eta_diff[2] != 0.0) {
        Ddg[8] = (grad_H[2] * eta_diff[2] - (H_k - H_k12)) / (eta_diff[2] * eta_diff[2]);
        Ddg[6] = (grad_H[0] - grad_H_12[0]) / eta_diff[2];
        Ddg[7] = (grad_H[1] - grad_H_12[1]) / eta_diff[2];
    }

    double Dfunc[9];
    matmat_dev(bcross_mat, Ddg, Dfunc);
    for (int k = 0; k < 9; k++) Dfunc[k] *= -dt;
    Dfunc[0] += 1.0; Dfunc[4] += 1.0; Dfunc[8] += 1.0;

    double Dfunc_inv[9], k_vec[3];
    matrix_inv_dev(Dfunc, Dfunc_inv);
    matvec_dev(Dfunc_inv, func, k_vec);

    row[0] -= k_vec[0];
    row[1] -= k_vec[1];
    row[2] -= k_vec[2];

    row[residual_idx] = sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_1st_order_newton_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const double* phi, const int p_n2, const int p_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        const double eta_k_shifted = row[k] + row[first_shift_idx + k];
        eta_k[k] = row[k];
        eta_diff[k] = eta_k_shifted - row[first_init_idx + k];
    }
    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_diff = v_k - v_n;
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double b_star_parallel = epsilon * row[first_free_idx + 1];
    const double b_star[3] = {row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k1 = row[first_free_idx + 5];
    const double H_k12 = row[first_free_idx + 6];
    const double grad_H_1 = row[first_free_idx + 7];
    const double grad_H_12[2] = {row[first_free_idx + 8], row[first_free_idx + 9]};

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_k[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_k[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_k[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_k[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_k[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_k[2], span3, bn3, bd3);

    double phi_val = 0.0;
    if (evaluate_e_field) {
        phi_val = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi, p_n2, p_n3);
    }
    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    const double H_k = epsilon * v_k * v_k / 2.0 + epsilon * mu * B_dot_b + phi_val;
    const double H_k123 = epsilon * v_n * v_n / 2.0 + epsilon * mu * B_dot_b + phi_val;

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

    const double grad_H_v = epsilon * v_k;

    double grad_I[3];
    grad_I[0] = (eta_diff[0] == 0.0) ? grad_H[0] : (H_k1 - H_n) / eta_diff[0];
    grad_I[1] = (eta_diff[1] == 0.0) ? grad_H[1] : (H_k12 - H_k1) / eta_diff[1];
    grad_I[2] = (eta_diff[2] == 0.0) ? grad_H[2] : (H_k123 - H_k12) / eta_diff[2];
    const double grad_I_v = (v_diff == 0.0) ? grad_H_v : (H_k - H_k123) / v_diff;

    double J_vec[3];
    for (int k = 0; k < 3; k++) J_vec[k] = b_star[k] / b_star_parallel;

    double func[3];
    for (int k = 0; k < 3; k++) func[k] = eta_diff[k] - dt * (J_vec[k] * grad_I_v);
    double func_v = v_diff + dt * dot3_dev(J_vec, grad_I);

    double Ddg[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (eta_diff[0] != 0.0) Ddg[0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / (eta_diff[0] * eta_diff[0]);
    if (eta_diff[1] != 0.0) {
        Ddg[4] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / (eta_diff[1] * eta_diff[1]);
        Ddg[3] = (grad_H_12[0] - grad_H_1) / eta_diff[1];
    }
    if (eta_diff[2] != 0.0) {
        Ddg[8] = (grad_H[2] * eta_diff[2] - (H_k123 - H_k12)) / (eta_diff[2] * eta_diff[2]);
        Ddg[6] = (grad_H[0] - grad_H_12[0]) / eta_diff[2];
        Ddg[7] = (grad_H[1] - grad_H_12[1]) / eta_diff[2];
    }
    const double Ddg_v = (v_diff == 0.0) ? 0.0 : (grad_H_v * v_diff - (H_k - H_k123)) / (v_diff * v_diff);

    // DF = [[I, B], [C^T, 1]], B = -dt*Ddg_v*J_vec, C = dt*Ddg^T @ J_vec
    double Bv[3], Cv[3];
    for (int k = 0; k < 3; k++) Bv[k] = -dt * Ddg_v * J_vec[k];
    double DdgT[9] = {Ddg[0], Ddg[3], Ddg[6], Ddg[1], Ddg[4], Ddg[7], Ddg[2], Ddg[5], Ddg[8]};
    matvec_dev(DdgT, J_vec, Cv);
    for (int k = 0; k < 3; k++) Cv[k] *= dt;

    const double schur = 1.0 - dot3_dev(Cv, Bv);

    double A_inv[9];
    A_inv[0] = Bv[0]*Cv[0]; A_inv[1] = Bv[0]*Cv[1]; A_inv[2] = Bv[0]*Cv[2];
    A_inv[3] = Bv[1]*Cv[0]; A_inv[4] = Bv[1]*Cv[1]; A_inv[5] = Bv[1]*Cv[2];
    A_inv[6] = Bv[2]*Cv[0]; A_inv[7] = Bv[2]*Cv[1]; A_inv[8] = Bv[2]*Cv[2];
    for (int k = 0; k < 9; k++) A_inv[k] /= schur;
    A_inv[0] += 1.0; A_inv[4] += 1.0; A_inv[8] += 1.0;

    double Binv[3], Cinv[3];
    for (int k = 0; k < 3; k++) { Binv[k] = -Bv[k] / schur; Cinv[k] = -Cv[k] / schur; }

    double k_vec[3];
    matvec_dev(A_inv, func, k_vec);
    for (int k = 0; k < 3; k++) k_vec[k] += Binv[k] * func_v;
    double k_v = dot3_dev(Cinv, func) + func_v / schur;

    row[0] -= k_vec[0];
    row[1] -= k_vec[1];
    row[2] -= k_vec[2];
    row[3] -= k_v;

    row[residual_idx] = sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2] + (k_v/v_k)*(k_v/v_k));
}
"""

_dg_newton_kernels = {}


def _get_dg_newton_kernel(name):
    if name not in _dg_newton_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _dg_newton_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_NEWTON_SRC, name)
    return _dg_newton_kernels[name]


def _dg_newton_launch(
    name, markers, first_init_idx, first_shift_idx, residual_idx, first_free_idx,
    mu_idx, epsilon, pn, tn1_dev, tn2_dev, tn3_dev, starts,
    grad_b_full, B_dot_b_coeffs, e_field, phi_coeffs, evaluate_e_field, dt,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_dg_newton_kernel(name)(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(first_shift_idx),
            np.int32(residual_idx), np.int32(first_free_idx), np.int32(mu_idx),
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(grad_b_full[0]), *d(grad_b_full[1]), *d(grad_b_full[2]),
            *d(B_dot_b_coeffs),
            *d(e_field[0]), *d(e_field[1]), *d(e_field[2]),
            *d(phi_coeffs),
            np.int32(bool(evaluate_e_field)), np.float64(dt),
        ),
    )


def push_gc_bxEstar_discrete_gradient_1st_order_newton_gpu(*args, **kwargs):
    """GPU replacement for one Newton iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order_newton`."""
    _dg_newton_launch("push_gc_bxEstar_discrete_gradient_1st_order_newton_cuda", *args, **kwargs)


def push_gc_Bstar_discrete_gradient_1st_order_newton_gpu(*args, **kwargs):
    """GPU replacement for one Newton iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order_newton`."""
    _dg_newton_launch("push_gc_Bstar_discrete_gradient_1st_order_newton_cuda", *args, **kwargs)


# ---------------------------------------------------------------------------
# push_gc_bxEstar_discrete_gradient_2nd_order /
# push_gc_Bstar_discrete_gradient_2nd_order: one Picard iteration (per
# marker, so per-marker parallel like the *_1st_order variants) of the
# Gonzalez discrete-gradient guiding-centre pushers -- unlike *_1st_order_newton
# this evaluates fields at the midpoint eta_mid = mod((eta_k+eta_n)/2, 1) and
# needs the domain Jacobian there (df_dispatch_dev/det3_dev), and only reads
# 2 pre-evaluated marker columns (H_n, H_k) instead of the Itoh-Abe set.
# ---------------------------------------------------------------------------

_DG_2ND_ORDER_SRC = r"""
extern "C" __global__
void push_gc_bxEstar_discrete_gradient_2nd_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* ub1, const int u1_n2, const int u1_n3,
    const double* ub2, const int u2_n2, const int u2_n3,
    const double* ub3, const int u3_n2, const int u3_n3,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_k[k] = row[k] + row[first_shift_idx + k];
        eta_n[k] = row[first_init_idx + k];
        double m = fmod((eta_k[k] + eta_n[k]) / 2.0, 1.0);
        if (m < 0.0) m += 1.0;
        eta_mid[k] = m;
        eta_diff[k] = eta_k[k] - eta_n[k];
    }
    const double v = row[3];
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double H_k = row[first_free_idx + 1];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double unit_b1[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ub1,u1_n2,u1_n3, ub2,u2_n2,u2_n3, ub3,u3_n2,u3_n3, unit_b1);

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

    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H);
    const double dZ_squared = dot3_dev(eta_diff, eta_diff);

    double grad_I[3];
    if (dZ_squared == 0.0) {
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k];
    } else {
        const double s = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k] + eta_diff[k] * s;
    }

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);
    b_star_parallel = (b_star_parallel * epsilon * v + B_dot_b) * det_df;

    double Exb[3];
    cross_dev(unit_b1, grad_I, Exb);

    double k_vec[3];
    for (int k = 0; k < 3; k++) k_vec[k] = Exb[k] / b_star_parallel;

    row[0] = eta_n[0] + dt * k_vec[0];
    row[1] = eta_n[1] + dt * k_vec[1];
    row[2] = eta_n[2] + dt * k_vec[2];

    const double r0 = row[0] - eta_k[0], r1 = row[1] - eta_k[1], r2 = row[2] - eta_k[2];
    row[residual_idx] = sqrt(r0*r0 + r1*r1 + r2*r2);
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_2nd_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* b2_1, const int b1_n2, const int b1_n3,
    const double* b2_2, const int b2_n2, const int b2_n3,
    const double* b2_3, const int b3_n2, const int b3_n3,
    const double* cb1, const int c1_n2, const int c1_n3,
    const double* cb2, const int c2_n2, const int c2_n3,
    const double* cb3, const int c3_n2, const int c3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_k[k] = row[k] + row[first_shift_idx + k];
        eta_n[k] = row[first_init_idx + k];
        double m = fmod((eta_k[k] + eta_n[k]) / 2.0, 1.0);
        if (m < 0.0) m += 1.0;
        eta_mid[k] = m;
        eta_diff[k] = eta_k[k] - eta_n[k];
    }
    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_mid = (v_k + v_n) / 2.0;
    const double v_diff = v_k - v_n;
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double H_k = row[first_free_idx + 1];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

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

    const double grad_H_v = epsilon * v_mid;
    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H) + v_diff * grad_H_v;
    const double dZ_squared = dot3_dev(eta_diff, eta_diff) + v_diff * v_diff;

    double grad_I[3];
    double grad_I_v;
    if (dZ_squared == 0.0) {
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k];
        grad_I_v = grad_H_v;
    } else {
        const double s = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k] + eta_diff[k] * s;
        grad_I_v = grad_H_v + v_diff * s;
    }

    double b2[3], b_star[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b2_1,b1_n2,b1_n3, b2_2,b2_n2,b2_n3, b2_3,b3_n2,b3_n3, b2);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cb1,c1_n2,c1_n3, cb2,c2_n2,c2_n3, cb3,c3_n2,c3_n3, b_star);
    for (int k = 0; k < 3; k++) b_star[k] = b_star[k] * epsilon * v_mid + b2[k];

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);
    b_star_parallel = (b_star_parallel * epsilon * v_mid + B_dot_b) * epsilon * det_df;

    double k_vec[3];
    for (int k = 0; k < 3; k++) k_vec[k] = b_star[k] / b_star_parallel * grad_I_v;
    const double k_v = -dot3_dev(b_star, grad_I) / b_star_parallel;

    row[0] = eta_n[0] + dt * k_vec[0];
    row[1] = eta_n[1] + dt * k_vec[1];
    row[2] = eta_n[2] + dt * k_vec[2];
    row[3] = v_n + dt * k_v;

    const double r0 = row[0] - eta_k[0], r1 = row[1] - eta_k[1], r2 = row[2] - eta_k[2];
    const double rv = (row[3] - v_k) / v_k;
    row[residual_idx] = sqrt(r0*r0 + r1*r1 + r2*r2 + rv*rv);
}
"""

_dg_2nd_order_kernels = {}


def _get_dg_2nd_order_kernel(name):
    if name not in _dg_2nd_order_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _dg_2nd_order_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_2ND_ORDER_SRC, name)
    return _dg_2nd_order_kernels[name]


def push_gc_bxEstar_discrete_gradient_2nd_order_gpu(
    markers, first_init_idx, first_shift_idx, residual_idx, first_free_idx, mu_idx,
    kind_map, params_dev, epsilon, pn, tn1_dev, tn2_dev, tn3_dev, starts,
    unit_b1, grad_b_full, B_dot_b_coeffs, curl_unit_b_dot_b0, e_field, evaluate_e_field, dt,
):
    """GPU replacement for one Picard iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_2nd_order`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_dg_2nd_order_kernel("push_gc_bxEstar_discrete_gradient_2nd_order_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(first_shift_idx),
            np.int32(residual_idx), np.int32(first_free_idx), np.int32(mu_idx),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(unit_b1[0]), *d(unit_b1[1]), *d(unit_b1[2]),
            *d(grad_b_full[0]), *d(grad_b_full[1]), *d(grad_b_full[2]),
            *d(B_dot_b_coeffs), *d(curl_unit_b_dot_b0),
            *d(e_field[0]), *d(e_field[1]), *d(e_field[2]),
            np.int32(bool(evaluate_e_field)), np.float64(dt),
        ),
    )


def push_gc_Bstar_discrete_gradient_2nd_order_gpu(
    markers, first_init_idx, first_shift_idx, residual_idx, first_free_idx, mu_idx,
    kind_map, params_dev, epsilon, pn, tn1_dev, tn2_dev, tn3_dev, starts,
    grad_b_full, b2, curl_unit_b2, B_dot_b_coeffs, curl_unit_b_dot_b0, e_field, evaluate_e_field, dt,
):
    """GPU replacement for one Picard iteration of
    :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_2nd_order`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_dg_2nd_order_kernel("push_gc_Bstar_discrete_gradient_2nd_order_cuda")(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(first_shift_idx),
            np.int32(residual_idx), np.int32(first_free_idx), np.int32(mu_idx),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(grad_b_full[0]), *d(grad_b_full[1]), *d(grad_b_full[2]),
            *d(b2[0]), *d(b2[1]), *d(b2[2]),
            *d(curl_unit_b2[0]), *d(curl_unit_b2[1]), *d(curl_unit_b2[2]),
            *d(B_dot_b_coeffs), *d(curl_unit_b_dot_b0),
            *d(e_field[0]), *d(e_field[1]), *d(e_field[2]),
            np.int32(bool(evaluate_e_field)), np.float64(dt),
        ),
    )
