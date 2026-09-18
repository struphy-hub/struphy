"""Hand-written CUDA replacements for select pusher kernels, used only under
``ARRAY_BACKEND=cupy``.

Unlike the generic Pyccel kernels in :mod:`~struphy.pic.pushing.pusher_kernels`
(which operate on plain host NumPy arrays regardless of backend), the kernels
here are real ``cupy.RawKernel`` CUDA source, executed directly on the GPU.
They are deliberately narrow: each one reproduces the exact arithmetic of one
Pyccel kernel, specialized for one :class:`~struphy.geometry.domains.Domain`
whose Jacobian is cheap enough that hand-specializing pays off.

Currently covered: :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`,
for the Cuboid domain (:func:`push_v_with_efield_cuboid_gpu`) and, more
generally, for any domain in :data:`SUPPORTED_GENERAL_KIND_MAPS`
(:func:`push_v_with_efield_general_gpu`). This one does need a real
(small-degree) tensor-product B-spline evaluation -- the electric field is a
1-form FEEC spline, not a constant -- so these functions port ``find_span``
and the combined N-/D-spline basis recursion
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

Every ``*_gpu`` function here is a thin wrapper around exactly one
:class:`~struphy.cuda.CudaKernel` (declared at module level, right above the
function that launches it): the kernel is compiled once, on first use, and
the function itself only builds the argument tuple and calls
:func:`~struphy.cuda.launch_1d`. If a function in this module does *not* sit
next to a ``CudaKernel``, it does not touch the GPU.
"""
from struphy.cuda import CudaKernel, launch_1d, load_cuda_source

_PUSH_V_EFIELD_CUBOID_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_push_v_efield_cuboid_src.cu")
_push_v_efield_cuboid_kernel = CudaKernel(_PUSH_V_EFIELD_CUBOID_SRC, "push_v_with_efield_cuboid")


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
    device once. ``tn1_dev``, ``tn2_dev``, ``tn3_dev`` (knot vectors) and
    ``e1_1_dev``, ``e1_2_dev``, ``e1_3_dev`` (FE coefficients of the 1-form
    E-field) are expected to already be CuPy arrays resident on the device --
    callers should cache them once rather than converting on every call, see
    :class:`~struphy.pic.pushing.pusher.Pusher`.
    """
    import numpy as np

    n_markers = markers.shape[0]
    launch_1d(
        _push_v_efield_cuboid_kernel,
        n_markers,
        (
            markers,
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
# push_v_with_efield_cuboid_gpu above hardcodes Cuboid's Jacobian (a constant
# diagonal matrix, precomputed on the host as `scale`) directly into the
# marker update, which is what makes it fast but restricts it to
# `kind_map == 10`. Everything else about it -- the B-spline evaluation -- is
# already fully general (arbitrary degree, arbitrary non-uniform knot vector;
# nothing there assumes Cuboid).
#
# push_v_with_efield_general_gpu below drops the constant-Jacobian
# assumption: it evaluates DF(eta) (and its inverse) per marker, per call, on
# the device, matching the general struphy.geometry.evaluation_kernels.df /
# struphy.linear_algebra.linalg_kernels dispatch that
# struphy.pic.pushing.pusher_kernels.push_v_with_efield uses on the CPU. This
# is genuinely more per-marker work (a Jacobian evaluation instead of a
# lookup), but still embarrassingly parallel across markers, so it remains a
# good GPU fit.
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
# Callers must check kind_map themselves (see Pusher._gpu_v_efield_general in
# pusher.py) and fall back to the host Pyccel kernel for anything else --
# this function does not raise on an unsupported kind_map, it is simply not
# wired up for one.

_GENERAL_GEOMETRY_SRC = load_cuda_source(__file__, "pusher_kernels_cuda/_general_geometry_src.cu")

_push_v_efield_general_kernel = CudaKernel(_GENERAL_GEOMETRY_SRC, "push_v_with_efield_general")

#: kind_map values df_dispatch_dev supports. Callers should check membership
#: before dispatching to push_v_with_efield_general_gpu.
SUPPORTED_GENERAL_KIND_MAPS = (10, 11, 12, 20, 21, 22, 30, 31, 32)


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
    ``params_dev`` is the domain's mapping-parameter array
    (``args_domain.params``), expected to already be a small CuPy array
    (cheap to keep device-resident; callers should cache it once).
    """
    import numpy as np

    n_markers = markers.shape[0]
    launch_1d(
        _push_v_efield_general_kernel,
        n_markers,
        (
            markers,
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
