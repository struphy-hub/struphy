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

_REFLECT_SRC = r"""
extern "C" __global__
void reflect_cuda(
    double* markers, const int n_cols,
    const long long* outside_inds, const int n_outside,
    const int axis,
    const int kind_map, const double* params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outside) return;

    const long long ip = outside_inds[i];
    double* row = markers + (size_t)ip * n_cols;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;

    double dfinv[9], v_logical[3];
    matrix_inv_dev(dfm, dfinv);

    // pull back of the velocity
    matvec_dev(dfinv, v, v_logical);

    // reverse the velocity component along `axis`
    v_logical[axis] *= -1.0;

    // push forward of the velocity
    matvec_dev(dfm, v_logical, v);

    row[3] = v[0];
    row[4] = v[1];
    row[5] = v[2];
}
"""

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
