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

_SPH_PUSHER_SRC = r"""
// Port of struphy.pic.sph_eval_kernels.box_based_kernel: SPH sum over the 27
// neighbouring boxes of the marker's own box.
__device__ double box_based_kernel_dev(
    const double* markers, const int n_cols,
    double e1, double e2, double e3,
    int loc_box,
    const int* boxes, const int n_box_cols,
    const int* neighbours,
    const int* holes,
    int periodic1, int periodic2, int periodic3,
    int index, int kernel_type,
    double h1, double h2, double h3)
{
    if (loc_box == -1) return 0.0;

    double acc = 0.0;
    for (int neigh = 0; neigh < 27; neigh++) {
        int box_to_search = neighbours[loc_box * 27 + neigh];
        int c = 0;
        while (boxes[(size_t)box_to_search * n_box_cols + c] != -1) {
            int p = boxes[(size_t)box_to_search * n_box_cols + c];
            c++;
            if (!holes[p]) {
                double r1 = distance_dev(e1, markers[(size_t)p * n_cols + 0], (bool)periodic1);
                double r2 = distance_dev(e2, markers[(size_t)p * n_cols + 1], (bool)periodic2);
                double r3 = distance_dev(e3, markers[(size_t)p * n_cols + 2], (bool)periodic3);
                acc += markers[(size_t)p * n_cols + index]
                     * smoothing_kernel_dev(kernel_type, r1, r2, r3, h1, h2, h3);
            }
        }
    }
    return acc;
}

// Shared tail of all three pushers: pull the logical-space force back to
// Cartesian with DF^-T and apply it to the marker velocity.
__device__ void apply_force_dev(
    double* row, double e1, double e2, double e3,
    int kind_map, const double* params,
    const double* force_logical, const double* gravity,
    double dt)
{
    double dfm[9], dfinv[9], force_cart[3];
    if (!df_dispatch_dev(kind_map, e1, e2, e3, params, dfm)) return;
    matrix_inv_dev(dfm, dfinv);
    // dfinvT @ force_logical == matvecT(dfinv, force_logical)
    matvecT_dev(dfinv, force_logical, force_cart);

    row[3] -= dt * (force_cart[0] - gravity[0]);
    row[4] -= dt * (force_cart[1] - gravity[1]);
    row[5] -= dt * (force_cart[2] - gravity[2]);
}

// --- push_v_sph_pressure (isothermal closure) ---
extern "C" __global__
void push_v_sph_pressure_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int* valid_mks,
    const int weight_idx, const int first_free_idx,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3,
    const double* gravity, const double kappa,
    const int kind_map, const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const double n_at_eta = row[first_free_idx];
    const int loc_box = (int)row[n_cols - 2];

    double grad_u[3] = {0.0, 0.0, 0.0};

    grad_u[0] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 1, h1, h2, h3);
    grad_u[0] *= kappa / n_at_eta;
    grad_u[0] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 1, kernel_type + 1, h1, h2, h3);

    if (kernel_type >= 340) {
        grad_u[1] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 2, h1, h2, h3);
        grad_u[1] *= kappa / n_at_eta;
        grad_u[1] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 1, kernel_type + 2, h1, h2, h3);
    }

    if (kernel_type >= 670) {
        grad_u[2] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 3, h1, h2, h3);
        grad_u[2] *= kappa / n_at_eta;
        grad_u[2] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 1, kernel_type + 3, h1, h2, h3);
    }

    apply_force_dev(row, e1, e2, e3, kind_map, params, grad_u, gravity, dt);
}

// --- push_v_sph_pressure_ideal_gas (polytropic closure, gamma = 5/3) ---
extern "C" __global__
void push_v_sph_pressure_ideal_gas_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int* valid_mks,
    const int weight_idx, const int first_free_idx,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3,
    const double* gravity, const double kappa,
    const int kind_map, const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    const double gamma = 5.0 / 3.0;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const double n_at_eta = row[first_free_idx];
    const int loc_box = (int)row[n_cols - 2];

    const double pref = kappa * pow(n_at_eta, gamma - 2.0);
    double grad_u[3] = {0.0, 0.0, 0.0};

    grad_u[0] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 1, h1, h2, h3);
    grad_u[0] *= pref;
    grad_u[0] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 2, kernel_type + 1, h1, h2, h3);

    if (kernel_type >= 340) {
        grad_u[1] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 2, h1, h2, h3);
        grad_u[1] *= pref;
        grad_u[1] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 2, kernel_type + 2, h1, h2, h3);
    }

    if (kernel_type >= 670) {
        grad_u[2] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type + 3, h1, h2, h3);
        grad_u[2] *= pref;
        grad_u[2] += kappa * box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
            neighbours, holes, periodic1, periodic2, periodic3, first_free_idx + 2, kernel_type + 3, h1, h2, h3);
    }

    apply_force_dev(row, e1, e2, e3, kind_map, params, grad_u, gravity, dt);
}

// --- push_v_viscosity (deviatoric strain-rate tensor) ---
extern "C" __global__
void push_v_viscosity_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int* valid_mks,
    const int first_free_idx,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3,
    const int kind_map, const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    double f_visc[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            const int coeff_idx = first_free_idx + 3 * (j + 1) + k;
            f_visc[j] += box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
                neighbours, holes, periodic1, periodic2, periodic3,
                coeff_idx, kernel_type + 1 + k, h1, h2, h3);
        }
    }

    const double no_gravity[3] = {0.0, 0.0, 0.0};
    apply_force_dev(row, e1, e2, e3, kind_map, params, f_visc, no_gravity, dt);
}
"""

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
