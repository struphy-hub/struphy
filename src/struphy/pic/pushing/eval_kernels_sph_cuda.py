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

_SPH_MARKER_COLUMN_SRC = r"""
extern "C" __global__
void sph_pressure_coeffs_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);

    const double weight = row[weight_idx];
    const double gamma = 5.0 / 3.0;

    row[column_nr] = n_at_eta;
    row[column_nr + 1] = weight / n_at_eta;
    row[column_nr + 2] = weight * pow(n_at_eta, gamma - 2.0);
}

extern "C" __global__
void sph_mean_velocity_coeffs_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);

    const double weight = row[weight_idx];
    const double scale = weight / n_at_eta;

    row[column_nr + 0] = scale * row[3];
    row[column_nr + 1] = scale * row[4];
    row[column_nr + 2] = scale * row[5];
}

extern "C" __global__
void sph_viscosity_tensor_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx, const int first_free_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3,
    const double mu)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);
    const double weight = row[weight_idx];

    double grad_v[3][3];
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            grad_v[j][k] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
                neighbours, holes, periodic1, periodic2, periodic3,
                first_free_idx + j, kernel_type + 1 + k, h1, h2, h3);
        }
    }

    double d_dev[3][3];
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++)
            d_dev[j][k] = 0.5 * (grad_v[j][k] + grad_v[k][j]);

    const double mean_trace = (d_dev[0][0] + d_dev[1][1] + d_dev[2][2]) / 3.0;
    d_dev[0][0] -= mean_trace;
    d_dev[1][1] -= mean_trace;
    d_dev[2][2] -= mean_trace;

    const double scale = -2.0 * mu * (weight / n_at_eta);
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            row[column_nr + 3 * j + k] = d_dev[j][k] * scale;
        }
    }
}
"""

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
