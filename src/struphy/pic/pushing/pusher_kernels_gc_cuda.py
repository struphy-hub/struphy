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
    markers, n_cols, first_init_idx, first_free_idx, mu_idx,
    kind_map, params_dev, epsilon,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    unit_b1_1_dev, unit_b1_2_dev, unit_b1_3_dev,
    grad_b_full_1_dev, grad_b_full_2_dev, grad_b_full_3_dev,
    B_dot_b_coeffs_dev, curl_unit_b_dot_b0_dev,
    e_field_1_dev, e_field_2_dev, e_field_3_dev,
    evaluate_e_field: bool,
    dt_a: float, dt_b: float, last: float,
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
            dev, np.int32(n_cols), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(first_free_idx), np.int32(mu_idx),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(unit_b1_1_dev), *d(unit_b1_2_dev), *d(unit_b1_3_dev),
            *d(grad_b_full_1_dev), *d(grad_b_full_2_dev), *d(grad_b_full_3_dev),
            *d(B_dot_b_coeffs_dev), *d(curl_unit_b_dot_b0_dev),
            *d(e_field_1_dev), *d(e_field_2_dev), *d(e_field_3_dev),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt_a), np.float64(dt_b), np.float64(last),
        ),
    )


def push_gc_Bstar_explicit_multistage_general_gpu(
    markers, n_cols, first_init_idx, first_free_idx, mu_idx,
    kind_map, params_dev, epsilon,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    grad_b_full_1_dev, grad_b_full_2_dev, grad_b_full_3_dev,
    b2_1_dev, b2_2_dev, b2_3_dev,
    curl_unit_b2_1_dev, curl_unit_b2_2_dev, curl_unit_b2_3_dev,
    B_dot_b_coeffs_dev, curl_unit_b_dot_b0_dev,
    e_field_1_dev, e_field_2_dev, e_field_3_dev,
    evaluate_e_field: bool,
    dt_a: float, dt_b: float, last: float,
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
            dev, np.int32(n_cols), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(first_free_idx), np.int32(mu_idx),
            np.int32(kind_map), params_dev,
            np.float64(epsilon),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(grad_b_full_1_dev), *d(grad_b_full_2_dev), *d(grad_b_full_3_dev),
            *d(b2_1_dev), *d(b2_2_dev), *d(b2_3_dev),
            *d(curl_unit_b2_1_dev), *d(curl_unit_b2_2_dev), *d(curl_unit_b2_3_dev),
            *d(B_dot_b_coeffs_dev), *d(curl_unit_b_dot_b0_dev),
            *d(e_field_1_dev), *d(e_field_2_dev), *d(e_field_3_dev),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt_a), np.float64(dt_b), np.float64(last),
        ),
    )
