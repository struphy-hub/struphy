"""CUDA kernels for fused variational grid evaluations."""

from struphy.cuda import CudaKernel, launch_1d, load_cuda_source

_KINETIC_ENERGY_SOURCE = load_cuda_source(__file__, "variational_kernels_cuda/kinetic_energy_grid.cu")

_kinetic_energy_kernel = CudaKernel(_KINETIC_ENERGY_SOURCE, "kinetic_energy_grid_cuda")


def prepare_kinetic_energy_kernel():
    """Force the fused kinetic-energy CUDA kernel to compile now, during
    model setup, rather than lazily on the first timed propagation step.

    Idempotent (see :meth:`~struphy.cuda.CudaKernel.compile`): every actual
    invocation still goes through the normal
    ``launch_1d(_kinetic_energy_kernel, ...)`` call in
    :func:`kinetic_energy_grid_gpu` below, which is a no-op past compilation
    once this has run.
    """
    _kinetic_energy_kernel.compile()


def kinetic_energy_grid_gpu(
    spans,
    bases,
    degree,
    starts,
    coefficients,
    coefficients1,
    metric,
    out,
    values,
    values1,
):
    """Evaluate both H1-vector splines and their metric product in one launch."""
    import cupy as cp
    import numpy as np

    prepare_kinetic_energy_kernel()
    spans = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.int64)) for value in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64)) for value in bases)
    coefficients = tuple(cp.ascontiguousarray(value) for value in coefficients)
    coefficients1 = tuple(cp.ascontiguousarray(value) for value in coefficients1)
    metric = cp.ascontiguousarray(metric)
    total = out.size
    launch_1d(
        _kinetic_energy_kernel,
        total,
        (
            *spans,
            *bases,
            *(np.int32(value.size) for value in spans),
            *(np.int32(value) for value in degree),
            *(np.int32(value) for value in starts),
            *coefficients,
            *coefficients1,
            np.int32(coefficients[0].shape[1]),
            np.int32(coefficients[0].shape[2]),
            metric,
            out,
            *values,
            *values1,
        ),
    )
    return out
