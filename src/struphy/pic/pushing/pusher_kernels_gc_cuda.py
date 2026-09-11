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
from struphy.cuda import load_cuda_source

_PUSH_GC_BXESTAR_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_push_gc_bxestar_src.cu")

_PUSH_GC_BSTAR_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_push_gc_bstar_src.cu")


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

_DG_1ST_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_dg_1st_src.cu")

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

_PUSH_GC_CC_J1_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_push_gc_cc_j1_src.cu")

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

_PUSH_GC_CC_J2_STAGE_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_push_gc_cc_j2_stage_src.cu")

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

_PUSH_GC_CC_J2_DG_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_push_gc_cc_j2_dg_src.cu")

_j2_dg_kernels = {}


def _get_j2_dg_kernel(name):
    if name not in _j2_dg_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _j2_dg_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_1ST_SRC + _PUSH_GC_CC_J2_DG_SRC, name)
    return _j2_dg_kernels[name]


def push_gc_cc_J2_dg_init_Hdiv_gpu(
    markers,
    first_init_idx,
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
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
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


def push_gc_cc_J2_dg_Hdiv_gpu(
    markers,
    first_init_idx,
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
    ud,
    const,
    alpha,
    dt,
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
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.float64(dt),
            np.float64(const),
            np.float64(alpha),
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
            *d(ud[0]),
            *d(ud[1]),
            *d(ud[2]),
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

_DG_NEWTON_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_dg_newton_src.cu")

_dg_newton_kernels = {}


def _get_dg_newton_kernel(name):
    if name not in _dg_newton_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _dg_newton_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_NEWTON_SRC, name)
    return _dg_newton_kernels[name]


def _dg_newton_launch(
    name,
    markers,
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
    B_dot_b_coeffs,
    e_field,
    phi_coeffs,
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

    _get_dg_newton_kernel(name)(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
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
            *d(B_dot_b_coeffs),
            *d(e_field[0]),
            *d(e_field[1]),
            *d(e_field[2]),
            *d(phi_coeffs),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt),
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

_DG_2ND_ORDER_SRC = load_cuda_source(__file__, "pusher_kernels_gc_cuda/_dg_2nd_order_src.cu")

_dg_2nd_order_kernels = {}


def _get_dg_2nd_order_kernel(name):
    if name not in _dg_2nd_order_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _dg_2nd_order_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _DG_2ND_ORDER_SRC, name)
    return _dg_2nd_order_kernels[name]


def push_gc_bxEstar_discrete_gradient_2nd_order_gpu(
    markers,
    first_init_idx,
    first_shift_idx,
    residual_idx,
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
    unit_b1,
    grad_b_full,
    B_dot_b_coeffs,
    curl_unit_b_dot_b0,
    e_field,
    evaluate_e_field,
    dt,
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
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(residual_idx),
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
            *d(unit_b1[0]),
            *d(unit_b1[1]),
            *d(unit_b1[2]),
            *d(grad_b_full[0]),
            *d(grad_b_full[1]),
            *d(grad_b_full[2]),
            *d(B_dot_b_coeffs),
            *d(curl_unit_b_dot_b0),
            *d(e_field[0]),
            *d(e_field[1]),
            *d(e_field[2]),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt),
        ),
    )


def push_gc_Bstar_discrete_gradient_2nd_order_gpu(
    markers,
    first_init_idx,
    first_shift_idx,
    residual_idx,
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
    grad_b_full,
    b2,
    curl_unit_b2,
    B_dot_b_coeffs,
    curl_unit_b_dot_b0,
    e_field,
    evaluate_e_field,
    dt,
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
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(residual_idx),
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
            *d(grad_b_full[0]),
            *d(grad_b_full[1]),
            *d(grad_b_full[2]),
            *d(b2[0]),
            *d(b2[1]),
            *d(b2[2]),
            *d(curl_unit_b2[0]),
            *d(curl_unit_b2[1]),
            *d(curl_unit_b2[2]),
            *d(B_dot_b_coeffs),
            *d(curl_unit_b_dot_b0),
            *d(e_field[0]),
            *d(e_field[1]),
            *d(e_field[2]),
            np.int32(bool(evaluate_e_field)),
            np.float64(dt),
        ),
    )
