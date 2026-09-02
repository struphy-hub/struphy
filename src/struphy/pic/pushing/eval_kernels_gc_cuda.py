"""Hand-written CUDA replacement for
:func:`~struphy.pic.pushing.eval_kernels_gc.driftkinetic_hamiltonian`, used
only under ``ARRAY_BACKEND=cupy``.

This is the ``eval_kernel`` of the discrete-gradient guiding-centre
propagators: it is re-run on *every* Picard iteration of every RK stage (89
calls in a 3-step ``LinearMHDDriftkineticCC`` run), writing the Hamiltonian
at the weighted evaluation point into one marker column. With markers
device-resident, leaving it on the host would cost a full marker round trip
per iteration -- by far the most frequent host crossing left in that model.

It is a plain per-marker 0-form spline evaluation, so it reuses the shared
``find_span_dev``/``b_splines_dev``/``eval_0form_dev`` device functions.
"""
from struphy.cuda import load_cuda_source

_DK_HAMILTONIAN_SRC = load_cuda_source(__file__, "eval_kernels_gc_cuda/_dk_hamiltonian_src.cu")

_dk_kernel = None


def _get_dk_kernel():
    global _dk_kernel
    if _dk_kernel is None:
        import cupy as cp

        _dk_kernel = cp.RawKernel(_DK_HAMILTONIAN_SRC, "driftkinetic_hamiltonian_cuda")
    return _dk_kernel


def driftkinetic_hamiltonian_gpu(
    markers,
    alpha,
    column_nr,
    first_init_idx,
    first_shift_idx,
    mu_idx,
    args_derham,
    epsilon,
    B_dot_b_coeffs,
    phi_coeffs,
    evaluate_e_field,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.driftkinetic_hamiltonian`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    bdb = cp.ascontiguousarray(B_dot_b_coeffs)
    phi = cp.ascontiguousarray(phi_coeffs)
    a = [float(x) for x in (alpha[0], alpha[1], alpha[2], alpha[3])]

    _get_dk_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(mu_idx),
            np.float64(a[0]),
            np.float64(a[1]),
            np.float64(a[2]),
            np.float64(a[3]),
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            bdb,
            np.int32(bdb.shape[1]),
            np.int32(bdb.shape[2]),
            phi,
            np.int32(phi.shape[1]),
            np.int32(phi.shape[2]),
            np.int32(bool(evaluate_e_field)),
        ),
    )


# ---------------------------------------------------------------------------
# grad_driftkinetic_hamiltonian / bstar_parallel_3form / bstar_2form /
# unit_b_1form: the remaining marker-column init/eval kernels of the
# discrete-gradient guiding-centre propagators. Unlike driftkinetic_hamiltonian
# above (self-contained 0-form-only source), these also need 1-/2-form
# evaluation and (for bstar_parallel_3form) the domain Jacobian, so they are
# built from pusher_kernels_cuda._GENERAL_GEOMETRY_SRC instead. All four
# share the same alpha-weighted evaluation point
#   eta_i = mod(alpha_i * (eta_i + shift_i) + (1 - alpha_i) * eta_i^n, 1)
# (and, for the two that need v_parallel, the same alpha-weighted v), factored
# into one device helper.
# ---------------------------------------------------------------------------

_GC_MARKER_COLUMN_SRC = load_cuda_source(__file__, "eval_kernels_gc_cuda/_gc_marker_column_src.cu")

_gc_marker_column_kernels = {}


def _get_gc_marker_column_kernel(name):
    if name not in _gc_marker_column_kernels:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _gc_marker_column_kernels[name] = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _GC_MARKER_COLUMN_SRC, name)
    return _gc_marker_column_kernels[name]


def grad_driftkinetic_hamiltonian_gpu(
    markers,
    alpha,
    column_nr,
    comps,
    first_init_idx,
    first_shift_idx,
    mu_idx,
    args_derham,
    epsilon,
    grad_b_full_coeffs,
    e_field_coeffs,
    evaluate_e_field,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.grad_driftkinetic_hamiltonian`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("grad_driftkinetic_hamiltonian_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(comps_dev.shape[0]),
            comps_dev,
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            np.int32(mu_idx),
            alpha_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            *d(grad_b_full_coeffs[0]),
            *d(grad_b_full_coeffs[1]),
            *d(grad_b_full_coeffs[2]),
            *d(e_field_coeffs[0]),
            *d(e_field_coeffs[1]),
            *d(e_field_coeffs[2]),
            np.int32(bool(evaluate_e_field)),
        ),
    )


def bstar_parallel_3form_gpu(
    markers,
    alpha,
    column_nr,
    first_init_idx,
    first_shift_idx,
    kind_map,
    params_dev,
    args_derham,
    epsilon,
    B_dot_b_coeffs,
    curl_unit_b_dot_b0_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.bstar_parallel_3form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("bstar_parallel_3form_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            alpha_dev,
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            *d(B_dot_b_coeffs),
            *d(curl_unit_b_dot_b0_coeffs),
        ),
    )


def bstar_2form_gpu(
    markers,
    alpha,
    column_nr,
    comps,
    first_init_idx,
    first_shift_idx,
    args_derham,
    epsilon,
    b2_coeffs,
    curl_unit_b2_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.bstar_2form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("bstar_2form_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(comps_dev.shape[0]),
            comps_dev,
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            alpha_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            *d(b2_coeffs[0]),
            *d(b2_coeffs[1]),
            *d(b2_coeffs[2]),
            *d(curl_unit_b2_coeffs[0]),
            *d(curl_unit_b2_coeffs[1]),
            *d(curl_unit_b2_coeffs[2]),
        ),
    )


def unit_b_1form_gpu(
    markers,
    alpha,
    column_nr,
    comps,
    first_init_idx,
    first_shift_idx,
    args_derham,
    unit_b1_coeffs,
):
    """GPU replacement for
    :func:`~struphy.pic.pushing.eval_kernels_gc.unit_b_1form`."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    alpha_dev = cp.asarray(np.asarray(alpha, dtype=np.float64))
    comps_dev = cp.asarray(np.asarray(comps, dtype=np.int32))
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)

    _get_gc_marker_column_kernel("unit_b_1form_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(column_nr),
            np.int32(comps_dev.shape[0]),
            comps_dev,
            np.int32(first_init_idx),
            np.int32(first_shift_idx),
            alpha_dev,
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            *d(unit_b1_coeffs[0]),
            *d(unit_b1_coeffs[1]),
            *d(unit_b1_coeffs[2]),
        ),
    )
