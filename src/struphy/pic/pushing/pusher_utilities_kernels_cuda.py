"""Hand-written CUDA replacement for
:func:`~struphy.pic.pushing.pusher_utilities_kernels.reflect`, used only
under ``ARRAY_BACKEND=cupy``.

``reflect`` is called from
:meth:`~struphy.pic.base.Particles.apply_kinetic_bc` -- i.e. once per
Runge-Kutta stage, inside the per-step hot path -- whenever a species has a
reflecting boundary. With markers device-resident it was the last thing in
that path still forcing a host<->device round trip of the whole marker
array, so it is ported here.

It reuses the geometry device functions (``df_dispatch_dev``,
``matrix_inv_dev``, ``matvec_dev``) from
:mod:`~struphy.pic.pushing.pusher_kernels_cuda`'s ``_GENERAL_GEOMETRY_SRC``.
Only the markers listed in ``outside_inds`` are touched, so the kernel is
launched over that index array rather than over all markers.
"""
from struphy.cuda import load_cuda_source

_REFLECT_SRC = load_cuda_source(__file__, "pusher_utilities_kernels_cuda/_reflect_src.cu")

_reflect_kernel = None


def _get_reflect_kernel():
    global _reflect_kernel
    if _reflect_kernel is None:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _reflect_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _REFLECT_SRC, "reflect_cuda")
    return _reflect_kernel


def reflect_gpu(markers, kind_map, params_dev, outside_inds, axis):
    """GPU replacement for
    :func:`~struphy.pic.pushing.pusher_utilities_kernels.reflect`, for any
    domain in
    :data:`~struphy.pic.pushing.pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS`.

    ``markers`` and ``outside_inds`` are device-resident; markers are updated
    in place.
    """
    import cupy as cp
    import numpy as np

    n_outside = int(outside_inds.shape[0])
    if n_outside == 0:
        return

    inds = cp.ascontiguousarray(outside_inds, dtype=cp.int64)
    threads = 256
    blocks = (n_outside + threads - 1) // threads
    _get_reflect_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            inds,
            np.int32(n_outside),
            np.int32(axis),
            np.int32(kind_map),
            params_dev,
        ),
    )
