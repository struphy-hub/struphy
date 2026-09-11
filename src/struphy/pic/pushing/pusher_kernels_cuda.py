"""Hand-written CUDA replacements for select pusher kernels, used only under
``ARRAY_BACKEND=cupy``.

Unlike the generic Pyccel kernels in :mod:`~struphy.pic.pushing.pusher_kernels`
(which operate on plain host NumPy arrays regardless of backend), the kernels
here are real ``cupy.RawKernel`` CUDA source, executed directly on the GPU.
They are deliberately narrow: each one reproduces the exact arithmetic of one
Pyccel kernel, specialized for one :class:`~struphy.geometry.domains.Domain`
whose Jacobian is cheap enough that hand-specializing pays off.

Currently covered: :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`
for the :class:`~struphy.geometry.domains.Cuboid` domain (``kind_map == 10``),
whose Jacobian ``DF = diag(r1 - l1, r2 - l2, r3 - l3)`` is constant, so the
whole stage update collapses to an elementwise scale-and-accumulate per marker
row with no spline evaluation at all.

Two entry points are provided:

* :func:`push_eta_stage_cuboid_gpu` replaces a single ``push_eta_stage`` call.
  It round-trips the full marker array through the device on every call, which
  is fine at the marker counts used in early testing but becomes the dominant
  cost at a few hundred thousand markers and up (H2D/D2H bandwidth, not
  compute, ends up dominating the run).

* :func:`push_eta_rk_periodic_gpu` additionally fuses in the boundary-condition
  bookkeeping that :meth:`~struphy.pic.base.Particles.apply_kinetic_bc` would
  otherwise do on the host between stages (periodic wrap + shift-column
  bookkeeping), restricted to an all-periodic ``bc``. That lets the whole
  multi-stage RK push run with the marker array resident on the device the
  entire time, doing exactly one H2D and one D2H transfer per
  :meth:`~struphy.pic.pushing.pusher.Pusher.__call__`, instead of one round
  trip per stage per kernel.

Also covered: :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`,
again for the Cuboid domain. Unlike ``push_eta_stage``, this one does need a
real (small-degree) tensor-product B-spline evaluation -- the electric field
is a 1-form FEEC spline, not a constant -- so :func:`push_v_with_efield_cuboid_gpu`
ports ``find_span`` and the combined N-/D-spline basis recursion
(:func:`~struphy.bsplines.bsplines_kernels.b_d_splines_slim`) to device code
alongside the local stencil sum
(:func:`~struphy.bsplines.evaluation_kernels_3d.eval_spline_mpi_kernel`).
Basis arrays are sized to a compile-time ``MAXP`` (spline degree 8), which
comfortably covers Struphy's usual degrees. The FE coefficient arrays
(``e1_1``, ``e1_2``, ``e1_3``) are the raw ``._data`` of the field's
:class:`~feectools.linalg.stencil.StencilVector` components; under the CuPy
backend these already live on the device (``StencilVector`` allocates via
``cunumpy``'s array-backend-aware ``xp``) and are never reassigned after
:meth:`~struphy.propagators.push_vin_efield.PushVinEfield.allocate` runs, so
they are passed straight through with no transfer at all -- only the marker
array round-trips through the device, exactly once per call.
"""
from struphy.cuda import load_cuda_source

_PUSH_ETA_CUBOID_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_push_eta_cuboid_src.cu")

_push_eta_cuboid_kernel = None


def _get_kernel():
    global _push_eta_cuboid_kernel
    if _push_eta_cuboid_kernel is None:
        import cupy as cp

        _push_eta_cuboid_kernel = cp.RawKernel(_PUSH_ETA_CUBOID_SRC, "push_eta_stage_cuboid")
    return _push_eta_cuboid_kernel


def push_eta_stage_cuboid_gpu(
    markers,
    n_cols: int,
    first_init_idx: int,
    first_free_idx: int,
    scale: tuple[float, float, float],
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`, restricted to
    the :class:`~struphy.geometry.domains.Cuboid` domain.

    ``markers`` is the host (pinned-memory) marker array; it is round-tripped
    through the device in full, matching the pattern used by
    :meth:`~struphy.pic.pushing.pusher.Pusher._reset_marker_buffers_gpu`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_kernel()
    n_markers = markers.shape[0]

    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.float64(scale[0]),
            np.float64(scale[1]),
            np.float64(scale[2]),
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


_PUSH_ETA_RK_PERIODIC_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_push_eta_rk_periodic_src.cu")

_push_eta_rk_periodic_kernel = None


def _get_periodic_kernel():
    global _push_eta_rk_periodic_kernel
    if _push_eta_rk_periodic_kernel is None:
        import cupy as cp

        _push_eta_rk_periodic_kernel = cp.RawKernel(_PUSH_ETA_RK_PERIODIC_SRC, "push_eta_rk_periodic")
    return _push_eta_rk_periodic_kernel


def push_eta_rk_periodic_gpu(
    markers,
    n_cols: int,
    vdim: int,
    first_init_idx: int,
    first_shift_idx: int,
    first_free_idx: int,
    scale: tuple[float, float, float],
    dt: float,
    a,
    b,
    n_stages: int,
):
    """Run a full multi-stage RK push of :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`
    plus periodic boundary handling, entirely on the device.

    Restricted to the :class:`~struphy.geometry.domains.Cuboid` domain and an
    all-``"periodic"`` ``bc``. Equivalent to calling
    :func:`push_eta_stage_cuboid_gpu` once per stage followed by the periodic
    branch of :meth:`~struphy.pic.base.Particles.apply_kinetic_bc`, but with a
    single H2D transfer at the start and a single D2H transfer at the end
    instead of one round trip per stage. Holes and ghost particles are
    invariant under a periodic-only push (positions are wrapped mod 1, never
    set to the -1.0 hole sentinel), so :meth:`~struphy.pic.base.Particles.update_holes`
    does not need to be called.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_periodic_kernel()
    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    dev = markers

    # reset: save initial phase-space coords, zero shift/free/residual columns
    # (matches Pusher._reset_marker_buffers_gpu, done once instead of per stage)
    dev[:, first_init_idx : first_init_idx + 3 + vdim] = dev[:, : 3 + vdim]
    dev[:, first_shift_idx:-2] = 0.0

    for stage in range(n_stages):
        last = 1.0 if stage == n_stages - 1 else 0.0
        kernel(
            (blocks,),
            (threads,),
            (
                dev,
                np.int32(n_cols),
                np.int32(n_markers),
                np.int32(first_init_idx),
                np.int32(first_free_idx),
                np.int32(first_shift_idx),
                np.float64(scale[0]),
                np.float64(scale[1]),
                np.float64(scale[2]),
                np.float64(dt * float(a[stage])),
                np.float64(dt * float(b[stage])),
                np.float64(last),
            ),
        )


_PUSH_V_EFIELD_CUBOID_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_push_v_efield_cuboid_src.cu")

_push_v_efield_cuboid_kernel = None


def _get_v_efield_kernel():
    global _push_v_efield_cuboid_kernel
    if _push_v_efield_cuboid_kernel is None:
        import cupy as cp

        _push_v_efield_cuboid_kernel = cp.RawKernel(_PUSH_V_EFIELD_CUBOID_SRC, "push_v_with_efield_cuboid")
    return _push_v_efield_cuboid_kernel


def push_v_with_efield_cuboid_gpu(
    markers,
    n_cols: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    e1_1_dev,
    e1_2_dev,
    e1_3_dev,
    scale: tuple[float, float, float],
    dt_const: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`, restricted
    to the :class:`~struphy.geometry.domains.Cuboid` domain.

    ``markers`` is the host marker array and is round-tripped through the
    device once (matching :func:`push_eta_stage_cuboid_gpu`). ``tn1_dev``,
    ``tn2_dev``, ``tn3_dev`` (knot vectors) and ``e1_1_dev``, ``e1_2_dev``,
    ``e1_3_dev`` (FE coefficients of the 1-form E-field) are expected to
    already be CuPy arrays resident on the device -- callers should cache
    them once rather than converting on every call, see
    :class:`~struphy.pic.pushing.pusher.Pusher`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_v_efield_kernel()
    n_markers = markers.shape[0]

    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            e1_1_dev,
            np.int32(e1_1_dev.shape[1]),
            np.int32(e1_1_dev.shape[2]),
            e1_2_dev,
            np.int32(e1_2_dev.shape[1]),
            np.int32(e1_2_dev.shape[2]),
            e1_3_dev,
            np.int32(e1_3_dev.shape[1]),
            np.int32(e1_3_dev.shape[2]),
            np.float64(scale[0]),
            np.float64(scale[1]),
            np.float64(scale[2]),
            np.float64(dt_const),
        ),
    )


# ============================================================================
# General (non-Cuboid-restricted) domain support
# ============================================================================
#
# The two kernels above hardcode Cuboid's Jacobian (a constant diagonal
# matrix, precomputed on the host as `scale`) directly into the marker
# update, which is what makes them fast but restricts them to `kind_map ==
# 10`. Everything else about them -- the B-spline evaluation in
# push_v_with_efield_cuboid_gpu -- is already fully general (arbitrary
# degree, arbitrary non-uniform knot vector; nothing there assumes Cuboid).
#
# push_eta_stage_general_gpu / push_v_with_efield_general_gpu below drop the
# constant-Jacobian assumption: they evaluate DF(eta) (and its inverse) per
# marker, per call, on the device, matching the general
# struphy.geometry.evaluation_kernels.df / struphy.linear_algebra.linalg_kernels
# dispatch that struphy.pic.pushing.pusher_kernels.push_eta_stage /
# push_v_with_efield use on the CPU. This is genuinely more per-marker work
# (a Jacobian evaluation instead of a lookup), but still embarrassingly
# parallel across markers, so it remains a good GPU fit.
#
# All analytic (closed-form) mappings in struphy.geometry.mappings_kernels
# are implemented: Cuboid (10), Orthogonal (11), Colella (12),
# HollowCylinder (20), PoweredEllipticCylinder (21), HollowTorus (22,
# including both its straight-field-line and equal-angle branches),
# ShafranovShiftCylinder (30), ShafranovSqrtCylinder (31) and
# ShafranovDshapedCylinder (32) -- see SUPPORTED_GENERAL_KIND_MAPS.
#
# NOT implemented: kind_map 0/1/2 (spline_3d / spline_2d_straight /
# spline_2d_torus), where the domain mapping F itself is an IGA B-spline
# volume (control points args.cx/cy/cz) rather than a closed-form function --
# evaluating DF there means differentiating that spline (basis_funs_1st_der /
# a derivative-spline evaluation, not just the tensor-product sum this file
# already has for FEEC fields), which is a separate, larger piece of work.
# Callers must check kind_map themselves (see Pusher._gpu_eta_general /
# _gpu_v_efield_general in pusher.py) and fall back to the host Pyccel kernel
# for anything else -- these functions do not raise on an unsupported
# kind_map, they are simply
# not wired up for one.

_GENERAL_GEOMETRY_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_general_geometry_src.cu")

_push_eta_general_kernel = None
_push_v_efield_general_kernel = None


def _get_eta_general_kernel():
    global _push_eta_general_kernel
    if _push_eta_general_kernel is None:
        import cupy as cp

        _push_eta_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_eta_stage_general")
    return _push_eta_general_kernel


def _get_v_efield_general_kernel():
    global _push_v_efield_general_kernel
    if _push_v_efield_general_kernel is None:
        import cupy as cp

        _push_v_efield_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_v_with_efield_general")
    return _push_v_efield_general_kernel


#: kind_map values df_dispatch_dev supports (Cuboid, Colella). Callers should
#: check membership before dispatching to the *_general_gpu functions below.
SUPPORTED_GENERAL_KIND_MAPS = (10, 11, 12, 20, 21, 22, 30, 31, 32)


def push_eta_stage_general_gpu(
    markers,
    n_cols: int,
    first_init_idx: int,
    first_free_idx: int,
    kind_map: int,
    params_dev,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS` (evaluates DF(eta) per
    marker instead of assuming a constant Jacobian, unlike
    :func:`push_eta_stage_cuboid_gpu`).

    ``markers`` is the host marker array, round-tripped through the device
    once per call. ``params_dev`` is the domain's mapping-parameter array
    (``args_domain.params``), expected to already be a small CuPy array
    (cheap to keep device-resident; callers should cache it once).
    """
    import cupy as cp
    import numpy as np

    kernel = _get_eta_general_kernel()
    n_markers = markers.shape[0]

    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


def push_v_with_efield_general_gpu(
    markers,
    n_cols: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    e1_1_dev,
    e1_2_dev,
    e1_3_dev,
    kind_map: int,
    params_dev,
    dt_const: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. See
    :func:`push_v_with_efield_cuboid_gpu` for the argument conventions
    (``tn*_dev``/``e1_*_dev`` are expected to already be device-resident);
    ``params_dev`` follows :func:`push_eta_stage_general_gpu`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_v_efield_general_kernel()
    n_markers = markers.shape[0]

    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            e1_1_dev,
            np.int32(e1_1_dev.shape[1]),
            np.int32(e1_1_dev.shape[2]),
            e1_2_dev,
            np.int32(e1_2_dev.shape[1]),
            np.int32(e1_2_dev.shape[2]),
            e1_3_dev,
            np.int32(e1_3_dev.shape[1]),
            np.int32(e1_3_dev.shape[2]),
            np.int32(kind_map),
            params_dev,
            np.float64(dt_const),
        ),
    )


_push_vxb_analytic_general_kernel = None
_push_vxb_implicit_general_kernel = None


def _get_vxb_analytic_general_kernel():
    global _push_vxb_analytic_general_kernel
    if _push_vxb_analytic_general_kernel is None:
        import cupy as cp

        _push_vxb_analytic_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_vxb_analytic_general")
    return _push_vxb_analytic_general_kernel


def _get_vxb_implicit_general_kernel():
    global _push_vxb_implicit_general_kernel
    if _push_vxb_implicit_general_kernel is None:
        import cupy as cp

        _push_vxb_implicit_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_vxb_implicit_general")
    return _push_vxb_implicit_general_kernel


def _launch_vxb_general(
    kernel,
    markers,
    n_cols,
    first_init_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    kind_map,
    params_dev,
    dt,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
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
            np.int32(kind_map),
            params_dev,
            np.float64(dt),
        ),
    )


def push_vxb_analytic_general_gpu(
    markers,
    n_cols: int,
    first_init_idx: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    kind_map: int,
    params_dev,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_vxb_analytic`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. Argument conventions match
    :func:`push_v_with_efield_general_gpu` (``tn*_dev``/``b2_*_dev`` already
    device-resident, ``params_dev`` the domain's mapping-parameter array).
    """
    _launch_vxb_general(
        _get_vxb_analytic_general_kernel(),
        markers,
        n_cols,
        first_init_idx,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        b2_1_dev,
        b2_2_dev,
        b2_3_dev,
        kind_map,
        params_dev,
        dt,
    )


def push_vxb_implicit_general_gpu(
    markers,
    n_cols: int,
    first_init_idx: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    kind_map: int,
    params_dev,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_vxb_implicit` (Crank-
    Nicolson rotation), for any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`.
    See :func:`push_vxb_analytic_general_gpu` for argument conventions."""
    _launch_vxb_general(
        _get_vxb_implicit_general_kernel(),
        markers,
        n_cols,
        first_init_idx,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        b2_1_dev,
        b2_2_dev,
        b2_3_dev,
        kind_map,
        params_dev,
        dt,
    )


_push_bxu_hdiv_general_kernel = None
_push_bxu_hcurl_general_kernel = None
_push_bxu_h1vec_general_kernel = None


def _get_bxu_hdiv_general_kernel():
    global _push_bxu_hdiv_general_kernel
    if _push_bxu_hdiv_general_kernel is None:
        import cupy as cp

        _push_bxu_hdiv_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_bxu_Hdiv_general")
    return _push_bxu_hdiv_general_kernel


def _get_bxu_hcurl_general_kernel():
    global _push_bxu_hcurl_general_kernel
    if _push_bxu_hcurl_general_kernel is None:
        import cupy as cp

        _push_bxu_hcurl_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_bxu_Hcurl_general")
    return _push_bxu_hcurl_general_kernel


def _get_bxu_h1vec_general_kernel():
    global _push_bxu_h1vec_general_kernel
    if _push_bxu_h1vec_general_kernel is None:
        import cupy as cp

        _push_bxu_h1vec_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_bxu_H1vec_general")
    return _push_bxu_h1vec_general_kernel


def _launch_bxu_general(
    kernel,
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    u_1_dev,
    u_2_dev,
    u_3_dev,
    kind_map,
    params_dev,
    boundary_cut,
    dt,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            u_1_dev,
            np.int32(u_1_dev.shape[1]),
            np.int32(u_1_dev.shape[2]),
            u_2_dev,
            np.int32(u_2_dev.shape[1]),
            np.int32(u_2_dev.shape[2]),
            u_3_dev,
            np.int32(u_3_dev.shape[1]),
            np.int32(u_3_dev.shape[2]),
            np.int32(kind_map),
            params_dev,
            np.float64(boundary_cut),
            np.float64(dt),
        ),
    )


def push_bxu_Hdiv_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    u2_1_dev,
    u2_2_dev,
    u2_3_dev,
    kind_map: int,
    params_dev,
    boundary_cut: float,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_bxu_Hdiv`, for any domain
    in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``u2_*_dev`` is the U-field's
    2-form FE coefficients (same evaluation as ``b2_*_dev``)."""
    _launch_bxu_general(
        _get_bxu_hdiv_general_kernel(),
        markers,
        n_cols,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        b2_1_dev,
        b2_2_dev,
        b2_3_dev,
        u2_1_dev,
        u2_2_dev,
        u2_3_dev,
        kind_map,
        params_dev,
        boundary_cut,
        dt,
    )


def push_bxu_Hcurl_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    u1_1_dev,
    u1_2_dev,
    u1_3_dev,
    kind_map: int,
    params_dev,
    boundary_cut: float,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_bxu_Hcurl`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``u1_*_dev`` is the
    U-field's 1-form FE coefficients."""
    _launch_bxu_general(
        _get_bxu_hcurl_general_kernel(),
        markers,
        n_cols,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        b2_1_dev,
        b2_2_dev,
        b2_3_dev,
        u1_1_dev,
        u1_2_dev,
        u1_3_dev,
        kind_map,
        params_dev,
        boundary_cut,
        dt,
    )


def push_bxu_H1vec_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    b2_1_dev,
    b2_2_dev,
    b2_3_dev,
    uv_1_dev,
    uv_2_dev,
    uv_3_dev,
    kind_map: int,
    params_dev,
    boundary_cut: float,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_bxu_H1vec`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``uv_*_dev`` is the
    U-field's (H^1)^3 vector-field FE coefficients."""
    _launch_bxu_general(
        _get_bxu_h1vec_general_kernel(),
        markers,
        n_cols,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        b2_1_dev,
        b2_2_dev,
        b2_3_dev,
        uv_1_dev,
        uv_2_dev,
        uv_3_dev,
        kind_map,
        params_dev,
        boundary_cut,
        dt,
    )


_push_pc_gxu_full_general_kernel = None
_push_pc_gxu_general_kernel = None


def _get_pc_gxu_full_general_kernel():
    global _push_pc_gxu_full_general_kernel
    if _push_pc_gxu_full_general_kernel is None:
        import cupy as cp

        _push_pc_gxu_full_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_pc_GXu_full_general")
    return _push_pc_gxu_full_general_kernel


def _get_pc_gxu_general_kernel():
    global _push_pc_gxu_general_kernel
    if _push_pc_gxu_general_kernel is None:
        import cupy as cp

        _push_pc_gxu_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_pc_GXu_general")
    return _push_pc_gxu_general_kernel


def push_pc_GXu_full_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    g11_dev,
    g12_dev,
    g13_dev,
    g21_dev,
    g22_dev,
    g23_dev,
    g31_dev,
    g32_dev,
    g33_dev,
    kind_map: int,
    params_dev,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_pc_GXu_full`, for any
    domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``g{i}{j}_dev`` is the FE
    coefficients of :math:`\\nabla_j(\\mathcal X \\cdot u)_i`, each row
    ``i`` a 1-form (same evaluation as ``push_v_with_efield_general_gpu``'s
    ``e1_*``)."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    g = (g11_dev, g12_dev, g13_dev, g21_dev, g22_dev, g23_dev, g31_dev, g32_dev, g33_dev)
    _get_pc_gxu_full_general_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            *g,
            np.int32(g11_dev.shape[1]),
            np.int32(g11_dev.shape[2]),
            np.int32(g12_dev.shape[1]),
            np.int32(g12_dev.shape[2]),
            np.int32(g13_dev.shape[1]),
            np.int32(g13_dev.shape[2]),
            np.int32(kind_map),
            params_dev,
            np.float64(dt),
        ),
    )


def push_pc_GXu_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    g11_dev,
    g12_dev,
    g13_dev,
    g21_dev,
    g22_dev,
    g23_dev,
    kind_map: int,
    params_dev,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_pc_GXu` (the 2-row
    variant of :func:`push_pc_GXu_full_general_gpu`), for any domain in
    :data:`SUPPORTED_GENERAL_KIND_MAPS`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    g = (g11_dev, g12_dev, g13_dev, g21_dev, g22_dev, g23_dev)
    _get_pc_gxu_general_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            *g,
            np.int32(g11_dev.shape[1]),
            np.int32(g11_dev.shape[2]),
            np.int32(g12_dev.shape[1]),
            np.int32(g12_dev.shape[2]),
            np.int32(g13_dev.shape[1]),
            np.int32(g13_dev.shape[2]),
            np.int32(kind_map),
            params_dev,
            np.float64(dt),
        ),
    )


_push_pc_eta_hcurl_general_kernel = None
_push_pc_eta_hdiv_general_kernel = None
_push_pc_eta_h1vec_general_kernel = None


def _get_pc_eta_hcurl_general_kernel():
    global _push_pc_eta_hcurl_general_kernel
    if _push_pc_eta_hcurl_general_kernel is None:
        import cupy as cp

        _push_pc_eta_hcurl_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_pc_eta_stage_Hcurl_general")
    return _push_pc_eta_hcurl_general_kernel


def _get_pc_eta_hdiv_general_kernel():
    global _push_pc_eta_hdiv_general_kernel
    if _push_pc_eta_hdiv_general_kernel is None:
        import cupy as cp

        _push_pc_eta_hdiv_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_pc_eta_stage_Hdiv_general")
    return _push_pc_eta_hdiv_general_kernel


def _get_pc_eta_h1vec_general_kernel():
    global _push_pc_eta_h1vec_general_kernel
    if _push_pc_eta_h1vec_general_kernel is None:
        import cupy as cp

        _push_pc_eta_h1vec_general_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC, "push_pc_eta_stage_H1vec_general")
    return _push_pc_eta_h1vec_general_kernel


def _launch_pc_eta_general(
    kernel,
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    u_1_dev,
    u_2_dev,
    u_3_dev,
    use_perp_model,
    kind_map,
    params_dev,
    dt_a,
    dt_b,
    last,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
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
            u_1_dev,
            np.int32(u_1_dev.shape[1]),
            np.int32(u_1_dev.shape[2]),
            u_2_dev,
            np.int32(u_2_dev.shape[1]),
            np.int32(u_2_dev.shape[2]),
            u_3_dev,
            np.int32(u_3_dev.shape[1]),
            np.int32(u_3_dev.shape[2]),
            np.int32(1 if use_perp_model else 0),
            np.int32(kind_map),
            params_dev,
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


def push_pc_eta_stage_Hcurl_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    u_1_dev,
    u_2_dev,
    u_3_dev,
    use_perp_model: bool,
    kind_map: int,
    params_dev,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_pc_eta_stage_Hcurl`, for
    any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``u_*_dev`` is the
    U-field's 1-form FE coefficients."""
    _launch_pc_eta_general(
        _get_pc_eta_hcurl_general_kernel(),
        markers,
        n_cols,
        first_init_idx,
        first_free_idx,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        u_1_dev,
        u_2_dev,
        u_3_dev,
        use_perp_model,
        kind_map,
        params_dev,
        dt_a,
        dt_b,
        last,
    )


def push_pc_eta_stage_Hdiv_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    u_1_dev,
    u_2_dev,
    u_3_dev,
    use_perp_model: bool,
    kind_map: int,
    params_dev,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_pc_eta_stage_Hdiv`, for
    any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``u_*_dev`` is the
    U-field's 2-form FE coefficients."""
    _launch_pc_eta_general(
        _get_pc_eta_hdiv_general_kernel(),
        markers,
        n_cols,
        first_init_idx,
        first_free_idx,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        u_1_dev,
        u_2_dev,
        u_3_dev,
        use_perp_model,
        kind_map,
        params_dev,
        dt_a,
        dt_b,
        last,
    )


def push_pc_eta_stage_H1vec_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    u_1_dev,
    u_2_dev,
    u_3_dev,
    use_perp_model: bool,
    kind_map: int,
    params_dev,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_pc_eta_stage_H1vec`, for
    any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``u_*_dev`` is the
    U-field's (H^1)^3 vector-field FE coefficients."""
    _launch_pc_eta_general(
        _get_pc_eta_h1vec_general_kernel(),
        markers,
        n_cols,
        first_init_idx,
        first_free_idx,
        pn,
        tn1_dev,
        tn2_dev,
        tn3_dev,
        starts,
        u_1_dev,
        u_2_dev,
        u_3_dev,
        use_perp_model,
        kind_map,
        params_dev,
        dt_a,
        dt_b,
        last,
    )


_push_weights_efield_lin_va_general_kernel = None


def _get_weights_efield_lin_va_general_kernel():
    global _push_weights_efield_lin_va_general_kernel
    if _push_weights_efield_lin_va_general_kernel is None:
        import cupy as cp

        _push_weights_efield_lin_va_general_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC, "push_weights_with_efield_lin_va_general"
        )
    return _push_weights_efield_lin_va_general_kernel


def push_weights_with_efield_lin_va_general_gpu(
    markers,
    n_cols,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    e1_1_dev,
    e1_2_dev,
    e1_3_dev,
    f0_values,
    kappa: float,
    vth: float,
    kind_map: int,
    params_dev,
    dt: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_weights_with_efield_lin_va`,
    for any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``f0_values`` is
    allocated via ``xp.zeros`` by the caller (EfieldWeightsCoupling) and
    updated in place every step, so under CuPy it is already device-resident
    -- passed straight through here, like ``e1_*_dev``."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    f0_dev = cp.ascontiguousarray(f0_values)
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_weights_efield_lin_va_general_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
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
            e1_1_dev,
            np.int32(e1_1_dev.shape[1]),
            np.int32(e1_1_dev.shape[2]),
            e1_2_dev,
            np.int32(e1_2_dev.shape[1]),
            np.int32(e1_2_dev.shape[2]),
            e1_3_dev,
            np.int32(e1_3_dev.shape[1]),
            np.int32(e1_3_dev.shape[2]),
            f0_dev,
            np.float64(kappa),
            np.float64(vth),
            np.int32(kind_map),
            params_dev,
            np.float64(dt),
        ),
    )


_push_deterministic_diffusion_general_kernel = None


def _get_deterministic_diffusion_general_kernel():
    global _push_deterministic_diffusion_general_kernel
    if _push_deterministic_diffusion_general_kernel is None:
        import cupy as cp

        _push_deterministic_diffusion_general_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC, "push_deterministic_diffusion_stage_general"
        )
    return _push_deterministic_diffusion_general_kernel


def push_deterministic_diffusion_stage_general_gpu(
    markers,
    n_cols,
    first_init_idx,
    first_free_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    pi_u_dev,
    pi_grad_u1_dev,
    pi_grad_u2_dev,
    pi_grad_u3_dev,
    diffusion_coeff: float,
    kind_map: int,
    params_dev,
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_deterministic_diffusion_stage`,
    for any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`. ``pi_u_dev`` is
    the 0-form FE coefficients of the (fixed-in-time) density, ``pi_grad_u{1,2,3}_dev``
    its gradient as a 1-form."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_deterministic_diffusion_general_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
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
            pi_u_dev,
            np.int32(pi_u_dev.shape[1]),
            np.int32(pi_u_dev.shape[2]),
            pi_grad_u1_dev,
            np.int32(pi_grad_u1_dev.shape[1]),
            np.int32(pi_grad_u1_dev.shape[2]),
            pi_grad_u2_dev,
            np.int32(pi_grad_u2_dev.shape[1]),
            np.int32(pi_grad_u2_dev.shape[2]),
            pi_grad_u3_dev,
            np.int32(pi_grad_u3_dev.shape[1]),
            np.int32(pi_grad_u3_dev.shape[2]),
            np.float64(diffusion_coeff),
            np.int32(kind_map),
            params_dev,
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )


# push_random_diffusion_stage does not touch geometry at all (a pure additive
# noise kick, no Jacobian, no field evaluation), so it gets its own minimal,
# domain-independent RawKernel source instead of living in
# _GENERAL_GEOMETRY_SRC -- it applies to every domain, not just
# SUPPORTED_GENERAL_KIND_MAPS.
_RANDOM_DIFFUSION_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_random_diffusion_src.cu")

_push_random_diffusion_kernel = None


def _get_random_diffusion_kernel():
    global _push_random_diffusion_kernel
    if _push_random_diffusion_kernel is None:
        import cupy as cp

        _push_random_diffusion_kernel = cp.RawKernel(_RANDOM_DIFFUSION_SRC, "push_random_diffusion_stage")
    return _push_random_diffusion_kernel


def push_random_diffusion_stage_gpu(markers, n_cols, noise, diffusion_coeff: float, dt: float):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_random_diffusion_stage`.
    Domain-independent (no geometry involved), so unlike the other
    ``*_general_gpu`` functions this one has no ``kind_map`` restriction.
    ``noise`` may be a host or a device array:
    :class:`~struphy.propagators.push_random_diffusion.PushRandomDiffusion` draws it
    through ``xp.random``, so under CuPy it is already on the device and no transfer
    happens here; a host array is accepted (and copied) so the signature stays
    backend-agnostic."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    dev = markers
    # cp.asarray takes host or device input; np.ascontiguousarray would raise on a
    # device array rather than transferring it.
    noise_dev = cp.ascontiguousarray(cp.asarray(noise, dtype=cp.float64))
    scale = float(np.sqrt(2.0 * dt * diffusion_coeff))
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    _get_random_diffusion_kernel()(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            noise_dev,
            np.float64(scale),
        ),
    )
