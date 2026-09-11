"""Hand-written CUDA replacements for the SPH velocity pushers in
:mod:`~struphy.pic.pushing.pusher_kernels_sph`, used only under
``ARRAY_BACKEND=cupy``.

All three are per-marker loops whose inner work is a box-neighbourhood SPH
sum, i.e. the same computation as
:func:`~struphy.pic.sph_eval_kernels.box_based_kernel`. That sum is factored
out here into the ``box_based_kernel_dev`` device function, which mirrors the
already-validated accumulation loop in
:mod:`~struphy.pic.sph_eval_kernels_cuda`'s
``box_based_evaluation_flat_cuda`` -- the only difference is that the
marker's own box index is read from its ``n_cols - 2`` column instead of
being looked up with ``find_box_dev``.

The smoothing-kernel evaluation (``smoothing_kernel_dev``) and the periodic
distance helper (``distance_dev``) are reused from that module's source
string, and the geometry (``df_dispatch_dev``, ``matrix_inv_dev``) from
:mod:`~struphy.pic.pushing.pusher_kernels_cuda`.

Note on ``df_inv``: the CPU kernels call
:func:`~struphy.geometry.evaluation_kernels.df_inv` with
``avoid_round_off=False``, which is exactly ``matrix_inv(df(eta))`` -- the
manual zeroing of analytically-zero entries is skipped -- so
``matrix_inv_dev(df_dispatch_dev(...))`` reproduces it exactly.
"""
from struphy.cuda import load_cuda_source

_SPH_PUSHER_SRC = load_cuda_source(__file__, "pusher_kernels_sph_cuda/_sph_pusher_src.cu")

_kernels = {}


def _source():
    from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC
    from struphy.pic.sph_eval_kernels_cuda import _SPH_EVAL_FLAT_SRC

    # _SPH_EVAL_FLAT_SRC brings distance_dev/smoothing_kernel_dev (and its own
    # __global__ entry points, which are simply unused here);
    # _GENERAL_GEOMETRY_SRC brings df_dispatch_dev/matrix_inv_dev/matvecT_dev.
    return _GENERAL_GEOMETRY_SRC + _SPH_EVAL_FLAT_SRC + _SPH_PUSHER_SRC


def _get_kernel(name):
    if name not in _kernels:
        import cupy as cp

        _kernels[name] = cp.RawKernel(_source(), name)
    return _kernels[name]


def _launch(
    name,
    markers,
    valid_mks,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    kind_map,
    params_dev,
    dt,
    *,
    weight_idx=None,
    first_free_idx=None,
    gravity=None,
    kappa=None,
):
    """Shared launch path for the three SPH velocity pushers."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    # valid_mks/holes are marker-row-indexed and therefore already device
    # arrays; boxes/neighbours belong to SortingBoxes and are host-owned, so
    # cp.asarray does the (small, box-sized) upload.
    dev_valid = cp.asarray(valid_mks).astype(cp.int32, copy=False)
    dev_boxes = cp.asarray(boxes).astype(cp.int32, copy=False)
    dev_neigh = cp.asarray(neighbours).astype(cp.int32, copy=False)
    dev_holes = cp.asarray(holes).astype(cp.int32, copy=False)
    dev_boxes = cp.ascontiguousarray(dev_boxes)
    dev_neigh = cp.ascontiguousarray(dev_neigh)

    args = [
        markers,
        np.int32(markers.shape[1]),
        np.int32(n_markers),
        dev_valid,
    ]
    if weight_idx is not None:
        args.append(np.int32(weight_idx))
    args.append(np.int32(first_free_idx))
    args += [
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
    if gravity is not None:
        args.append(cp.ascontiguousarray(gravity, dtype=cp.float64))
        args.append(np.float64(kappa))
    args += [np.int32(kind_map), params_dev, np.float64(dt)]

    _get_kernel(name)((blocks,), (threads,), tuple(args))


def push_v_sph_pressure_gpu(
    markers,
    valid_mks,
    weight_idx,
    first_free_idx,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    gravity,
    kappa,
    kind_map,
    params_dev,
    dt,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.pusher_kernels_sph.push_v_sph_pressure`."""
    _launch(
        "push_v_sph_pressure_cuda",
        markers,
        valid_mks,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
        kind_map,
        params_dev,
        dt,
        weight_idx=weight_idx,
        first_free_idx=first_free_idx,
        gravity=gravity,
        kappa=kappa,
    )


def push_v_sph_pressure_ideal_gas_gpu(
    markers,
    valid_mks,
    weight_idx,
    first_free_idx,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    gravity,
    kappa,
    kind_map,
    params_dev,
    dt,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.pusher_kernels_sph.push_v_sph_pressure_ideal_gas`."""
    _launch(
        "push_v_sph_pressure_ideal_gas_cuda",
        markers,
        valid_mks,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
        kind_map,
        params_dev,
        dt,
        weight_idx=weight_idx,
        first_free_idx=first_free_idx,
        gravity=gravity,
        kappa=kappa,
    )


def push_v_viscosity_gpu(
    markers,
    valid_mks,
    first_free_idx,
    boxes,
    neighbours,
    holes,
    periodic,
    kernel_type,
    h,
    kind_map,
    params_dev,
    dt,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.pusher_kernels_sph.push_v_viscosity`."""
    _launch(
        "push_v_viscosity_cuda",
        markers,
        valid_mks,
        boxes,
        neighbours,
        holes,
        periodic,
        kernel_type,
        h,
        kind_map,
        params_dev,
        dt,
        first_free_idx=first_free_idx,
    )
