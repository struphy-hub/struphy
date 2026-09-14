"""Hand-written CUDA replacements for the per-marker *diagnostics* kernels in
:mod:`~struphy.pic.utilities_kernels`, used only under ``ARRAY_BACKEND=cupy``.

These run every time step (they back the scalar quantities a model saves,
e.g. ``en_fB`` in :class:`~struphy.models.guiding_center.GuidingCenter`), and
each one writes a diagnostics column of the marker array in place. With
markers now device-resident (see :class:`~struphy.pic.base.Particles`), the
compiled host-only versions were the last thing forcing a host<->device
round trip of the whole marker array in the per-step path -- porting them
removes it.

Both kernels here are plain per-marker 0-form spline evaluations, so they
reuse the ``find_span_dev``/``b_splines_dev``/``eval_0form_dev`` device
functions rather than defining their own.

Every real GPU kernel this module launches is a :class:`~struphy.cuda.CudaKernel`
(or a :class:`~struphy.cuda.CudaKernelSet` entry) declared at module scope; if
a function here doesn't sit next to one, it does not touch the GPU.
"""
from struphy.cuda import CudaKernel, CudaKernelSet, launch_1d, load_cuda_source
from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

_UTILITIES_SRC = load_cuda_source(__file__, "utilities_kernels_cuda/_utilities_src.cu")

_kernels = CudaKernelSet(_UTILITIES_SRC)


def _launch_0form_diag(kernel_name, markers, args_derham, first_diagnostics_idx, mu_idx, coeffs):
    """Shared launch path for the two 0-form diagnostics kernels above."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]

    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    coeffs = cp.ascontiguousarray(coeffs)

    launch_1d(
        _kernels[kernel_name],
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
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
            coeffs,
            np.int32(coeffs.shape[1]),
            np.int32(coeffs.shape[2]),
        ),
    )


def eval_magnetic_background_energy_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, abs_B0):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_background_energy`.
    ``markers`` is device-resident and written in place.
    """
    _launch_0form_diag(
        "eval_magnetic_background_energy_cuda",
        markers,
        args_derham,
        first_diagnostics_idx,
        mu_idx,
        abs_B0,
    )


def eval_energy_5d_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, absB):
    """GPU replacement for :func:`~struphy.pic.utilities_kernels.eval_energy_5d`.
    ``markers`` is device-resident and written in place.
    """
    _launch_0form_diag(
        "eval_energy_5d_cuda",
        markers,
        args_derham,
        first_diagnostics_idx,
        mu_idx,
        absB,
    )


def eval_canonical_toroidal_moment_5d_gpu(
    markers, args_derham, first_diagnostics_idx, mu_idx, idx_can_momentum, epsilon, B0, R0, absB
):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_canonical_toroidal_moment_5d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    launch_1d(
        _kernels["eval_canonical_toroidal_moment_5d_cuda"],
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
            np.int32(idx_can_momentum),
            np.float64(epsilon),
            np.float64(B0),
            np.float64(R0),
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
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_canonical_toroidal_moment_6d_gpu(markers, args_derham, first_diagnostics_idx, epsilon, B0, R0, absB):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_canonical_toroidal_moment_6d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    launch_1d(
        _kernels["eval_canonical_toroidal_moment_6d_cuda"],
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.float64(epsilon),
            np.float64(B0),
            np.float64(R0),
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
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_magnetic_moment_5d_gpu(markers, args_derham, first_diagnostics_idx, absB):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_moment_5d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    launch_1d(
        _kernels["eval_magnetic_moment_5d_cuda"],
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
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
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_magnetic_energy_PBb_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, abs_B0, PBb):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_energy_PBb`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    abs_B0 = cp.ascontiguousarray(abs_B0)
    PBb = cp.ascontiguousarray(PBb)
    launch_1d(
        _kernels["eval_magnetic_energy_PBb_cuda"],
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
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
            abs_B0,
            np.int32(abs_B0.shape[1]),
            np.int32(abs_B0.shape[2]),
            PBb,
            np.int32(PBb.shape[1]),
            np.int32(PBb.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# eval_guiding_center_from_6d needs the domain Jacobian and a 2-form (magnetic
# field) evaluation, so unlike the pure 0-form diagnostics above it is built
# on top of pusher_kernels_cuda's shared geometry/spline device functions
# rather than the small self-contained source in this module.
# ---------------------------------------------------------------------------

_GC_FROM_6D_SRC = load_cuda_source(__file__, "utilities_kernels_cuda/_gc_from_6d_src.cu")
_gc6d_kernel = CudaKernel(_GENERAL_GEOMETRY_SRC + _GC_FROM_6D_SRC, "eval_guiding_center_from_6d_cuda")


def eval_guiding_center_from_6d_gpu(
    markers, args_derham, kind_map, params_dev, first_diagnostics_idx, epsilon, b21, b22, b23, absB
):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_guiding_center_from_6d`, for any
    domain in :data:`~struphy.pic.pushing.pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    b21 = cp.ascontiguousarray(b21)
    b22 = cp.ascontiguousarray(b22)
    b23 = cp.ascontiguousarray(b23)
    absB = cp.ascontiguousarray(absB)
    launch_1d(
        _gc6d_kernel,
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
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
            b21,
            np.int32(b21.shape[1]),
            np.int32(b21.shape[2]),
            b22,
            np.int32(b22.shape[1]),
            np.int32(b22.shape[2]),
            b23,
            np.int32(b23.shape[1]),
            np.int32(b23.shape[2]),
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# eval_gradB_ediff: writes markers[:, idx] = mu * dot(eta_diff, gradB +
# grad_PB_b), evaluated at the midpoint eta_mid = mod((eta+eta_init)/2, 1).
# Called once per fixed-point iteration by CurrentCoupling5DGradB's
# discrete-gradient algorithm. Needs 1-form spline evaluation (unlike the
# 0-form diagnostics above), so this is built from
# pusher_kernels_cuda._GENERAL_GEOMETRY_SRC instead of the private
# find_span_dev/b_splines_dev/eval_0form_dev helpers used by _UTILITIES_SRC.
# ---------------------------------------------------------------------------

_GRADB_EDIFF_SRC = load_cuda_source(__file__, "utilities_kernels_cuda/_gradb_ediff_src.cu")
_gradb_ediff_kernel = CudaKernel(_GENERAL_GEOMETRY_SRC + _GRADB_EDIFF_SRC, "eval_gradB_ediff_cuda")


def eval_gradB_ediff_gpu(
    markers,
    first_init_idx,
    mu_idx,
    pn,
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts,
    gradB1_dev,
    grad_PB_b1_dev,
    idx,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.utilities_kernels.eval_gradB_ediff`.

    ``gradB1_dev``/``grad_PB_b1_dev`` are each a 3-tuple of device arrays
    (the 1-form's 3 components), matching the (unpacked) ``gradB1, gradB2,
    gradB3`` / ``grad_PB_b1, grad_PB_b2, grad_PB_b3`` arguments of the CPU
    kernel.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    launch_1d(
        _gradb_ediff_kernel,
        n_markers,
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(mu_idx),
            np.int32(idx),
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
            *d(gradB1_dev[0]),
            *d(gradB1_dev[1]),
            *d(gradB1_dev[2]),
            *d(grad_PB_b1_dev[0]),
            *d(grad_PB_b1_dev[1]),
            *d(grad_PB_b1_dev[2]),
        ),
    )
