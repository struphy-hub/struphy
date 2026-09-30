"""Pairs of pyccel and CUDA kernels, selected at runtime from the cunumpy backend."""

import math

import numpy as np
from cunumpy import PyccelKernel
from cunumpy.xp import array_backend


def is_cuda_backend() -> bool:
    """Whether the active cunumpy backend is CuPy."""
    return array_backend.backend == "cupy"


class CudaKernel:
    """A ``cupy.RawKernel`` called with the 1:1 corresponding arguments of its pyccel counterpart.

    Argument classes must already be the CUDA versions (:mod:`struphy.utils.cuda_arguments`);
    no arrays are converted or copied at call time. One thread is launched per marker.
    """

    def __init__(self, source: str, name: str, block_size: int = 128):
        self.name = name
        self._source = source
        self._block_size = block_size
        self._raw_kernel = None

    def __call__(self, *args):
        if self._raw_kernel is None:
            import cupy as cp

            self._raw_kernel = cp.RawKernel(self._source, self.name)

        values = []
        n_threads = None
        for arg in args:
            if hasattr(arg, "values"):
                values += arg.values
                n_threads = n_threads or getattr(arg, "n_markers", None)
            elif isinstance(arg, bool):
                values.append(np.bool_(arg))
            elif isinstance(arg, int):
                values.append(np.int32(arg))
            elif isinstance(arg, float):
                values.append(np.float64(arg))
            else:
                values.append(arg)

        if n_threads is None:
            raise ValueError(f"{self.name}: no CudaMarkerArguments passed, cannot set the number of threads.")

        grid = (math.ceil(n_threads / self._block_size),)
        self._raw_kernel(grid, (self._block_size,), tuple(values))


class Kernel:
    """A pyccel kernel and its CUDA counterpart; calls the one matching the cunumpy backend."""

    def __init__(self, pyccel_kernel: PyccelKernel, cuda_kernel: CudaKernel):
        self.pyccel_kernel = pyccel_kernel
        self.cuda_kernel = cuda_kernel

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        return self.cuda_kernel if is_cuda_backend() else self.pyccel_kernel

    def __call__(self, *args):
        return self.get_kernel()(*args)
