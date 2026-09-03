"""CUDA kernels for fused variational grid evaluations."""

from struphy.cuda import load_cuda_source

_KINETIC_ENERGY_KERNEL = None

_KINETIC_ENERGY_SOURCE = load_cuda_source(__file__, "variational_kernels_cuda/kinetic_energy_grid.cu")


def prepare_kinetic_energy_kernel():
    """Compile and cache the fused kinetic-energy CUDA kernel."""
    import cupy as cp

    global _KINETIC_ENERGY_KERNEL
    if _KINETIC_ENERGY_KERNEL is None:
        _KINETIC_ENERGY_KERNEL = cp.RawKernel(
            _KINETIC_ENERGY_SOURCE,
            "kinetic_energy_grid_cuda",
        )
        # Force NVRTC compilation during model setup rather than the first
        # timed propagation step.
        _KINETIC_ENERGY_KERNEL.compile()
    return _KINETIC_ENERGY_KERNEL


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

    kernel = prepare_kinetic_energy_kernel()
    spans = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.int64)) for value in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64)) for value in bases)
    coefficients = tuple(cp.ascontiguousarray(value) for value in coefficients)
    coefficients1 = tuple(cp.ascontiguousarray(value) for value in coefficients1)
    metric = cp.ascontiguousarray(metric)
    total = out.size
    threads = 256
    kernel(
        ((total + threads - 1) // threads,),
        (threads,),
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
