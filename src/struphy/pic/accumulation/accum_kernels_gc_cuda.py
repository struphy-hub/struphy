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
from struphy.cuda import load_cuda_source

_GC_MAG_DENSITY_0FORM_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_gc_mag_density_0form_src.cu")

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

from struphy.pic.accumulation.accum_kernels_cuda import charge_density_0form_gpu as _charge_density_0form_gpu


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

_CC_LIN_MHD_5D_D_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_cc_lin_mhd_5d_d_src.cu")

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
    markers,
    kind_map,
    params_dev,
    epsilon,
    ep_scale,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    norm_b1,
    curl_norm_b,
    basis_u,
    mat12_dev,
    mat13_dev,
    mat23_dev,
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
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_cc_lin_mhd_5d_D_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.float64(ep_scale),
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
            np.int32(basis_u),
            mat12_dev,
            mat13_dev,
            mat23_dev,
            *dims(mat12_dev),
            *dims(mat13_dev),
            *dims(mat23_dev),
        ),
    )


# ---------------------------------------------------------------------------
# cc_lin_mhd_5d_gradB: vector-only accumulation (no matrix) into V_u, with
# filling  w_p * mu * [B2_x . norm_b_x . grad(PB)] / |B*_para|  (times
# 1/det(DF) for basis_u=2, which additionally adds grad_PBeq to grad_PB).
# Uses fill_vec_dev below -- the vector half of fill_mat_vec_dev, needed on
# its own here since no matrix block is filled.
# ---------------------------------------------------------------------------

# Port of filler_kernels.fill_vec; shared by all vector-filling accumulators
# in this module.
_FILL_VEC_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_fill_vec_src.cu")

_CC_LIN_MHD_5D_GRADB_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_cc_lin_mhd_5d_gradb_src.cu")

_cc_gradB_kernel = None


def _get_cc_lin_mhd_5d_gradB_kernel():
    global _cc_gradB_kernel
    if _cc_gradB_kernel is None:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _cc_gradB_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _FILL_VEC_SRC + _CC_LIN_MHD_5D_GRADB_SRC,
            "cc_lin_mhd_5d_gradB_cuda",
        )
    return _cc_gradB_kernel


def cc_lin_mhd_5d_gradB_gpu(
    markers,
    first_init_idx,
    mu_idx,
    kind_map,
    params_dev,
    epsilon,
    ep_scale,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    norm_b1,
    curl_norm_b,
    grad_PB,
    grad_PBeq,
    basis_u,
    vec1_dev,
    vec2_dev,
    vec3_dev,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_gradB`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_cc_lin_mhd_5d_gradB_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(mu_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.float64(ep_scale),
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
            *d(grad_PB[0]),
            *d(grad_PB[1]),
            *d(grad_PB[2]),
            *d(grad_PBeq[0]),
            *d(grad_PBeq[1]),
            *d(grad_PBeq[2]),
            np.int32(basis_u),
            vec1_dev,
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            vec2_dev,
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            vec3_dev,
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# cc_lin_mhd_5d_curlb: full symmetric 6-block matrix + vector fill, with the
# curvature filling
#
#     M = w_p * v^2 * [B2_x (curl_b (x) curl_b) (-B2_x)] / |B*_para|^2
#     V = w_p * v^2 * [B2_x curl_b]             / |B*_para|
#
# (times 1/det^2 resp. 1/det for basis_u=2). Only basis_u 0 and 2 exist here.
#
# NOTE: the basis_u == 0 branch of the CPU kernel used to accumulate into
# filling_m/filling_v with `+=` across markers, which made it order-dependent
# and inherently sequential; that was a typo and is fixed on this branch --
# see ISSUE_cc_lin_mhd_5d_curlb_order_dependent.md. This port assumes the
# fixed (per-marker) semantics.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_5D_CURLB_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_cc_lin_mhd_5d_curlb_src.cu")

_cc_curlb_kernel = None


def _get_cc_lin_mhd_5d_curlb_kernel():
    global _cc_curlb_kernel
    if _cc_curlb_kernel is None:
        import cupy as cp

        from struphy.pic.accumulation.accum_kernels_cuda import _LINEAR_VLASOV_AMPERE_EXTRA_SRC
        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _cc_curlb_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _LINEAR_VLASOV_AMPERE_EXTRA_SRC + _CC_LIN_MHD_5D_CURLB_SRC,
            "cc_lin_mhd_5d_curlb_cuda",
        )
    return _cc_curlb_kernel


def cc_lin_mhd_5d_curlb_gpu(
    markers,
    kind_map,
    params_dev,
    epsilon,
    ep_scale,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    norm_b1,
    curl_norm_b,
    basis_u,
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
    :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_curlb`."""
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
            np.int32(a.shape[1]),
            np.int32(a.shape[2]),
            np.int32(a.shape[3]),
            np.int32(a.shape[4]),
            np.int32(a.shape[5]),
        )

    _get_cc_lin_mhd_5d_curlb_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.float64(ep_scale),
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
            np.int32(basis_u),
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
# cc_lin_mhd_5d_gradB_dg_init / cc_lin_mhd_5d_gradB_dg
#
# Both are vector-only accumulators of the same shape; they differ only in
#
#   * where they evaluate: `dg_init` at the current position eta, `dg` at the
#     midpoint  eta_mid = mod((eta + eta^n) / 2, 1),
#   * `dg` adds a discrete-gradient correction term proportional to
#     eta_diff = eta - eta^n, scaled by `const`.
#
# They are therefore compiled from one source with an `is_dg` switch, so the
# per-marker geometry/spline work is written once.
#
#   V = sum over {Beq, B} of  w_p mu [X_x b_x] grad(PB_.) / |B*_para|
#       (+ const [X_x b_x] eta_diff / |B*_para|   for `dg`)
#
# (times 1/det for basis_u=2). Only basis_u 0 and 2 exist here.
# ---------------------------------------------------------------------------

_CC_LIN_MHD_5D_GRADB_DG_SRC = load_cuda_source(__file__, "accum_kernels_gc_cuda/_cc_lin_mhd_5d_gradb_dg_src.cu")

_cc_gradB_dg_kernel = None


def _get_cc_lin_mhd_5d_gradB_dg_kernel():
    global _cc_gradB_dg_kernel
    if _cc_gradB_dg_kernel is None:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _cc_gradB_dg_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _FILL_VEC_SRC + _CC_LIN_MHD_5D_GRADB_DG_SRC,
            "cc_lin_mhd_5d_gradB_dg_cuda",
        )
    return _cc_gradB_dg_kernel


def cc_lin_mhd_5d_gradB_dg_gpu(
    markers,
    first_init_idx,
    mu_idx,
    kind_map,
    params_dev,
    epsilon,
    ep_scale,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2,
    beq2,
    norm_b1,
    curl_norm_b,
    grad_PB,
    grad_PBeq,
    basis_u,
    vec1_dev,
    vec2_dev,
    vec3_dev,
    const=0.0,
    is_dg=False,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_gradB_dg`
    (``is_dg=True``, using ``const``) or of
    :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_gradB_dg_init`
    (``is_dg=False``, where ``const`` is unused)."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_cc_lin_mhd_5d_gradB_dg_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(mu_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.float64(ep_scale),
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
            *d(beq2[0]),
            *d(beq2[1]),
            *d(beq2[2]),
            *d(norm_b1[0]),
            *d(norm_b1[1]),
            *d(norm_b1[2]),
            *d(curl_norm_b[0]),
            *d(curl_norm_b[1]),
            *d(curl_norm_b[2]),
            *d(grad_PB[0]),
            *d(grad_PB[1]),
            *d(grad_PB[2]),
            *d(grad_PBeq[0]),
            *d(grad_PBeq[1]),
            *d(grad_PBeq[2]),
            np.int32(basis_u),
            np.float64(const),
            np.int32(bool(is_dg)),
            vec1_dev,
            np.int32(vec1_dev.shape[1]),
            np.int32(vec1_dev.shape[2]),
            vec2_dev,
            np.int32(vec2_dev.shape[1]),
            np.int32(vec2_dev.shape[2]),
            vec3_dev,
            np.int32(vec3_dev.shape[1]),
            np.int32(vec3_dev.shape[2]),
        ),
    )
