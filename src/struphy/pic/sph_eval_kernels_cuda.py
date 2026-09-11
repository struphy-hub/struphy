"""Hand-written CUDA replacement for the box-based SPH kernel-density evaluation
in :mod:`~struphy.pic.sph_eval_kernels`, used only under ``ARRAY_BACKEND=cupy``.

:func:`~struphy.pic.sph_eval_kernels.box_based_evaluation_flat` (called from
:meth:`~struphy.pic.base.Particles._eval_sph`, in turn used by
:meth:`~struphy.pic.base.Particles.eval_density` and
:meth:`~struphy.pic.base.Particles.eval_velocity`) is the actual SPH
kernel-density-estimation sum -- the defining operation of "smoothed particle
hydrodynamics": reconstruct a continuous field at a set of evaluation points
by summing a smoothing kernel over every marker in the 27 sorting boxes
neighbouring each point. It is embarrassingly parallel across evaluation
points (unlike the pusher kernels, there is no per-marker output to race on),
which makes it a clean fit for one CUDA thread per evaluation point.

:func:`box_based_evaluation_flat_gpu` ports :func:`~struphy.pic.sorting_kernels.find_box`,
the 27-neighbour box loop of :func:`~struphy.pic.sph_eval_kernels.box_based_kernel`,
and every smoothing kernel in :mod:`~struphy.pic.sph_smoothing_kernels` (all of
them: they are cheap closed-form tensor products of one-dimensional
trigonometric/Gaussian/linear kernels, or -- for ``linear_isotropic_3d`` -- a
simple radial one, so there is no reason to port only the default kernel
type). ``markers``/``boxes``/``neighbours``/``holes`` are the same
host-resident arrays used everywhere else in this backend (see
``ISSUE_cupy_particles_never_pushed.md``); this function round-trips them
through the device once per call, matching :func:`push_v_with_efield_cuboid_gpu`
in :mod:`~struphy.pic.pushing.pusher_kernels_cuda` -- ``_eval_sph`` is a
diagnostics/reconstruction entry point, not a per-step hot loop, so there is
no benefit to caching device buffers across calls the way the pushers do.
"""
from struphy.cuda import load_cuda_source

_SPH_EVAL_FLAT_SRC = load_cuda_source(__file__, "sph_eval_kernels_cuda/_sph_eval_flat_src.cu")

_box_based_evaluation_flat_kernel = None
_box_based_evaluation_meshgrid_kernel = None


def _get_kernel():
    global _box_based_evaluation_flat_kernel
    if _box_based_evaluation_flat_kernel is None:
        import cupy as cp

        _box_based_evaluation_flat_kernel = cp.RawKernel(_SPH_EVAL_FLAT_SRC, "box_based_evaluation_flat_cuda")
    return _box_based_evaluation_flat_kernel


def _get_meshgrid_kernel():
    global _box_based_evaluation_meshgrid_kernel
    if _box_based_evaluation_meshgrid_kernel is None:
        import cupy as cp

        _box_based_evaluation_meshgrid_kernel = cp.RawKernel(_SPH_EVAL_FLAT_SRC, "box_based_evaluation_meshgrid_cuda")
    return _box_based_evaluation_meshgrid_kernel


def box_based_evaluation_flat_gpu(
    markers,
    eta1,
    eta2,
    eta3,
    nx: int,
    ny: int,
    nz: int,
    domain_array,
    boxes,
    neighbours,
    holes,
    periodic1: bool,
    periodic2: bool,
    periodic3: bool,
    index: int,
    kernel_type: int,
    h1: float,
    h2: float,
    h3: float,
    out,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.sph_eval_kernels.box_based_evaluation_flat`.

    All inputs are host arrays (``markers``, ``domain_array``, ``boxes``,
    ``neighbours``, ``holes``, matching the rest of the CuPy backend) except
    ``eta1``/``eta2``/``eta3``/``out``, which may already be device-resident
    (the caller passes whatever backend it's using for evaluation points).
    Everything is round-tripped through the device once for this call.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_kernel()
    n_cols = markers.shape[1]
    n_eval = eta1.shape[0]
    n_box_cols = boxes.shape[1]

    dev_markers = cp.asarray(markers)
    # ascontiguousarray, not asarray: eta1/eta2/eta3 may be arbitrary (e.g.
    # strided/sliced) views, and the kernel indexes them as dense 1-D buffers
    # -- asarray is a no-op on an already-CuPy, already-float64 view and
    # would silently pass the RawKernel a pointer with the wrong stride.
    dev_eta1 = cp.ascontiguousarray(eta1, dtype=cp.float64)
    dev_eta2 = cp.ascontiguousarray(eta2, dtype=cp.float64)
    dev_eta3 = cp.ascontiguousarray(eta3, dtype=cp.float64)
    dev_domain = cp.asarray(domain_array, dtype=cp.float64)
    dev_boxes = cp.asarray(boxes, dtype=cp.int32)
    dev_neighbours = cp.asarray(neighbours, dtype=cp.int32)
    dev_holes = cp.asarray(holes, dtype=cp.int32)
    dev_out = cp.zeros(n_eval, dtype=cp.float64)

    threads = 256
    blocks = (n_eval + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(n_cols),
            dev_eta1,
            dev_eta2,
            dev_eta3,
            np.int32(n_eval),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            dev_domain,
            dev_boxes,
            np.int32(n_box_cols),
            dev_neighbours,
            dev_holes,
            np.int32(1 if periodic1 else 0),
            np.int32(1 if periodic2 else 0),
            np.int32(1 if periodic3 else 0),
            np.int32(index),
            np.int32(kernel_type),
            np.float64(h1),
            np.float64(h2),
            np.float64(h3),
            dev_out,
        ),
    )
    if isinstance(out, cp.ndarray):
        out[:] = dev_out
    else:
        dev_out.get(out=out)


def box_based_evaluation_meshgrid_gpu(
    markers,
    eta1,
    eta2,
    eta3,
    nx: int,
    ny: int,
    nz: int,
    domain_array,
    boxes,
    neighbours,
    holes,
    periodic1: bool,
    periodic2: bool,
    periodic3: bool,
    index: int,
    kernel_type: int,
    h1: float,
    h2: float,
    h3: float,
    out,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.sph_eval_kernels.box_based_evaluation_meshgrid`.

    ``eta1``, ``eta2``, ``eta3`` are the full 3-D meshgrid arrays (as produced
    by ``xp.meshgrid(..., indexing="ij")``); only their distinct 1-D axis
    vectors are transferred to the device, see the CUDA source. Otherwise
    behaves like :func:`box_based_evaluation_flat_gpu`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_meshgrid_kernel()
    n_cols = markers.shape[1]
    n1_eval, n2_eval, n3_eval = eta1.shape[0], eta2.shape[1], eta3.shape[2]
    n_box_cols = boxes.shape[1]

    dev_markers = cp.asarray(markers)
    # ascontiguousarray, not asarray: eta1[:,0,0] etc. are strided views into
    # the full meshgrid (stride = the *other* axes' extents, not 1 element),
    # and the kernel indexes them as dense 1-D buffers -- asarray is a no-op
    # on an already-CuPy view and would silently pass the RawKernel a pointer
    # with the wrong stride (this was a real bug: mismatched against the
    # already-validated flat kernel on identical points until fixed).
    dev_eta1 = cp.ascontiguousarray(eta1[:, 0, 0], dtype=cp.float64)
    dev_eta2 = cp.ascontiguousarray(eta2[0, :, 0], dtype=cp.float64)
    dev_eta3 = cp.ascontiguousarray(eta3[0, 0, :], dtype=cp.float64)
    dev_domain = cp.asarray(domain_array, dtype=cp.float64)
    dev_boxes = cp.asarray(boxes, dtype=cp.int32)
    dev_neighbours = cp.asarray(neighbours, dtype=cp.int32)
    dev_holes = cp.asarray(holes, dtype=cp.int32)
    dev_out = cp.zeros((n1_eval, n2_eval, n3_eval), dtype=cp.float64)

    n_total = n1_eval * n2_eval * n3_eval
    threads = 256
    blocks = (n_total + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(n_cols),
            dev_eta1,
            dev_eta2,
            dev_eta3,
            np.int32(n1_eval),
            np.int32(n2_eval),
            np.int32(n3_eval),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            dev_domain,
            dev_boxes,
            np.int32(n_box_cols),
            dev_neighbours,
            dev_holes,
            np.int32(1 if periodic1 else 0),
            np.int32(1 if periodic2 else 0),
            np.int32(1 if periodic3 else 0),
            np.int32(index),
            np.int32(kernel_type),
            np.float64(h1),
            np.float64(h2),
            np.float64(h3),
            dev_out,
        ),
    )
    if isinstance(out, cp.ndarray):
        out[:] = dev_out
    else:
        dev_out.get(out=out)


# ---------------------------------------------------------------------------
# naive_evaluation_flat / naive_evaluation_meshgrid: the O(N) reference
# implementation of the SPH kernel-density sum above (sums over every marker
# instead of the 27 neighbouring boxes), used only for testing/verification
# per the CPU docstring -- not a hot loop, so like box_based_evaluation_*
# this round-trips its (host-resident) inputs through the device once per
# call rather than caching device buffers. Reuses distance_dev/
# smoothing_kernel_dev from _SPH_EVAL_FLAT_SRC above; unlike the box-based
# kernels the result is divided by Np, matching
# :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_kernel`.
# ---------------------------------------------------------------------------

_SPH_EVAL_NAIVE_SRC = load_cuda_source(__file__, "sph_eval_kernels_cuda/_sph_eval_naive_src.cu")

_naive_evaluation_flat_kernel = None
_naive_evaluation_meshgrid_kernel = None


def _get_naive_flat_kernel():
    global _naive_evaluation_flat_kernel
    if _naive_evaluation_flat_kernel is None:
        import cupy as cp

        _naive_evaluation_flat_kernel = cp.RawKernel(
            _SPH_EVAL_FLAT_SRC + _SPH_EVAL_NAIVE_SRC, "naive_evaluation_flat_cuda"
        )
    return _naive_evaluation_flat_kernel


def _get_naive_meshgrid_kernel():
    global _naive_evaluation_meshgrid_kernel
    if _naive_evaluation_meshgrid_kernel is None:
        import cupy as cp

        _naive_evaluation_meshgrid_kernel = cp.RawKernel(
            _SPH_EVAL_FLAT_SRC + _SPH_EVAL_NAIVE_SRC, "naive_evaluation_meshgrid_cuda"
        )
    return _naive_evaluation_meshgrid_kernel


def naive_evaluation_flat_gpu(
    markers,
    Np: float,
    eta1,
    eta2,
    eta3,
    holes,
    periodic1: bool,
    periodic2: bool,
    periodic3: bool,
    index: int,
    kernel_type: int,
    h1: float,
    h2: float,
    h3: float,
    out,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_flat`."""
    import cupy as cp
    import numpy as np

    kernel = _get_naive_flat_kernel()
    n_cols = markers.shape[1]
    n_markers = markers.shape[0]
    n_eval = eta1.shape[0]

    dev_markers = cp.asarray(markers)
    dev_eta1 = cp.ascontiguousarray(eta1, dtype=cp.float64)
    dev_eta2 = cp.ascontiguousarray(eta2, dtype=cp.float64)
    dev_eta3 = cp.ascontiguousarray(eta3, dtype=cp.float64)
    dev_holes = cp.asarray(holes, dtype=cp.int32)
    dev_out = cp.zeros(n_eval, dtype=cp.float64)

    threads = 256
    blocks = (n_eval + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(n_cols),
            np.int32(n_markers),
            np.float64(Np),
            dev_eta1,
            dev_eta2,
            dev_eta3,
            np.int32(n_eval),
            dev_holes,
            np.int32(1 if periodic1 else 0),
            np.int32(1 if periodic2 else 0),
            np.int32(1 if periodic3 else 0),
            np.int32(index),
            np.int32(kernel_type),
            np.float64(h1),
            np.float64(h2),
            np.float64(h3),
            dev_out,
        ),
    )
    if isinstance(out, cp.ndarray):
        out[:] = dev_out
    else:
        dev_out.get(out=out)


def naive_evaluation_meshgrid_gpu(
    markers,
    Np: float,
    eta1,
    eta2,
    eta3,
    holes,
    periodic1: bool,
    periodic2: bool,
    periodic3: bool,
    index: int,
    kernel_type: int,
    h1: float,
    h2: float,
    h3: float,
    out,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.sph_eval_kernels.naive_evaluation_meshgrid`.

    Like :func:`box_based_evaluation_meshgrid_gpu`, ``eta1``/``eta2``/``eta3``
    are the 3 distinct 1-D axis vectors of the meshgrid, not the broadcast
    arrays -- the CPU kernel this ports only ever reads
    ``eta1[i,0,0]``/``eta2[0,j,0]``/``eta3[0,0,k]``.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_naive_meshgrid_kernel()
    n_cols = markers.shape[1]
    n_markers = markers.shape[0]
    n1_eval, n2_eval, n3_eval = eta1.shape[0], eta2.shape[1], eta3.shape[2]

    dev_markers = cp.asarray(markers)
    dev_eta1 = cp.ascontiguousarray(eta1[:, 0, 0], dtype=cp.float64)
    dev_eta2 = cp.ascontiguousarray(eta2[0, :, 0], dtype=cp.float64)
    dev_eta3 = cp.ascontiguousarray(eta3[0, 0, :], dtype=cp.float64)
    dev_holes = cp.asarray(holes, dtype=cp.int32)
    dev_out = cp.zeros((n1_eval, n2_eval, n3_eval), dtype=cp.float64)

    threads = 256
    n_total = n1_eval * n2_eval * n3_eval
    blocks = (n_total + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev_markers,
            np.int32(n_cols),
            np.int32(n_markers),
            np.float64(Np),
            dev_eta1,
            dev_eta2,
            dev_eta3,
            np.int32(n1_eval),
            np.int32(n2_eval),
            np.int32(n3_eval),
            dev_holes,
            np.int32(1 if periodic1 else 0),
            np.int32(1 if periodic2 else 0),
            np.int32(1 if periodic3 else 0),
            np.int32(index),
            np.int32(kernel_type),
            np.float64(h1),
            np.float64(h2),
            np.float64(h3),
            dev_out,
        ),
    )
    if isinstance(out, cp.ndarray):
        out[:] = dev_out
    else:
        dev_out.get(out=out)
