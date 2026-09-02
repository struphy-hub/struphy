"""Hand-written CUDA replacements for the SPH marker-column kernels in
:mod:`~struphy.pic.pushing.eval_kernels_sph`, used only under
``ARRAY_BACKEND=cupy``.

Like the SPH velocity pushers in
:mod:`~struphy.pic.pushing.pusher_kernels_sph_cuda`, these are per-marker
loops whose inner work is one or more box-neighbourhood SPH sums, i.e. the
same computation as :func:`~struphy.pic.sph_eval_kernels.box_based_kernel`.
Rather than duplicating that device function, this module reuses
``box_based_kernel_dev`` from :mod:`~struphy.pic.pushing.pusher_kernels_sph_cuda`
(which itself reuses ``distance_dev``/``smoothing_kernel_dev`` from
:mod:`~struphy.pic.sph_eval_kernels_cuda` and ``df_dispatch_dev``/
``matrix_inv_dev`` from :mod:`~struphy.pic.pushing.pusher_kernels_cuda`,
though the two geometry helpers are unused here since none of these three
kernels touch the domain Jacobian).
"""
from struphy.cuda import load_cuda_source

_SPH_MARKER_COLUMN_SRC = load_cuda_source(__file__, "eval_kernels_sph_cuda/_sph_marker_column_src.cu")

_sph_marker_column_kernels = {}


def _get_sph_marker_column_kernel(name):
    if name not in _sph_marker_column_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC
        from struphy.pic.pushing.pusher_kernels_sph_cuda import _SPH_PUSHER_SRC
        from struphy.pic.sph_eval_kernels_cuda import _SPH_EVAL_FLAT_SRC

        _sph_marker_column_kernels[name] = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _SPH_EVAL_FLAT_SRC + _SPH_PUSHER_SRC + _SPH_MARKER_COLUMN_SRC,
            name,
        )
    return _sph_marker_column_kernels[name]


def _sph_marker_column_launch(
    name,
    markers,
    valid_mks,
    column_nr,
    weight_idx,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    *,
    first_free_idx=None,
    mu=None,
):
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    dev_valid = cp.ascontiguousarray(cp.asarray(valid_mks).astype(cp.int32, copy=False))
    dev_boxes = cp.ascontiguousarray(cp.asarray(boxes).astype(cp.int32, copy=False))
    dev_neigh = cp.ascontiguousarray(cp.asarray(neighbours).astype(cp.int32, copy=False))
    dev_holes = cp.ascontiguousarray(cp.asarray(holes).astype(cp.int32, copy=False))

    args = [
        markers,
        np.int32(markers.shape[1]),
        np.int32(n_markers),
        np.int32(column_nr),
        np.int32(weight_idx),
    ]
    if first_free_idx is not None:
        args.append(np.int32(first_free_idx))
    args += [
        dev_valid,
        dev_boxes,
        np.int32(dev_boxes.shape[1]),
        dev_neigh,
        dev_holes,
        np.int32(bool(periodic[0])),
        np.int32(bool(periodic[1])),
        np.int32(bool(periodic[2])),
        np.int32(kernel_type),
        np.float64(h[0]),
        np.float64(h[1]),
        np.float64(h[2]),
    ]
    if mu is not None:
        args.append(np.float64(mu))

    _get_sph_marker_column_kernel(name)((blocks,), (threads,), tuple(args))


def sph_pressure_coeffs_gpu(
    markers, valid_mks, column_nr, weight_idx, boxes, neighbours, holes, periodic, kernel_type, h
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.eval_kernels_sph.sph_pressure_coeffs`."""
    _sph_marker_column_launch(
        "sph_pressure_coeffs_cuda",
        markers,
        valid_mks,
        column_nr,
        weight_idx,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
    )


def sph_mean_velocity_coeffs_gpu(
    markers, valid_mks, column_nr, weight_idx, boxes, neighbours, holes, periodic, kernel_type, h
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.eval_kernels_sph.sph_mean_velocity_coeffs`."""
    _sph_marker_column_launch(
        "sph_mean_velocity_coeffs_cuda",
        markers,
        valid_mks,
        column_nr,
        weight_idx,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
    )


def sph_viscosity_tensor_gpu(
    markers,
    valid_mks,
    column_nr,
    weight_idx,
    first_free_idx,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    mu,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.eval_kernels_sph.sph_viscosity_tensor`."""
    _sph_marker_column_launch(
        "sph_viscosity_tensor_cuda",
        markers,
        valid_mks,
        column_nr,
        weight_idx,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
        first_free_idx=first_free_idx,
        mu=mu,
    )
