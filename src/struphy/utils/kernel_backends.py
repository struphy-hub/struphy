"""Pairs of pyccel and CUDA kernels, selected at runtime from the cunumpy backend.

Each Struphy kernel has a pyccel version (wrapped in :class:`cunumpy.PyccelKernel`) and,
optionally, a CUDA version (:class:`CudaKernel`) with a 1:1 corresponding signature.
:class:`Kernel` holds both and dispatches to the CUDA kernel when the cunumpy backend is
``"cupy"``, see :func:`is_cuda_backend`. The pyccelized argument classes are transformed
once (at setup) into their CUDA counterparts by :func:`~struphy.utils.kernel_transform.transform`.

Example
-------
>>> kernel = Kernel(
...     pyccel_kernel=PyccelKernel(demo_kernels.push_eta_linear),
...     cuda_kernel=CudaKernel(PUSH_ETA_LINEAR_SRC, "push_eta_linear"),
... )
>>> catalog.register(kernel)
>>> catalog.get("push_eta_linear")(dt, stage, args_markers, args_domain)  # NumPy backend
>>> # CuPy backend, transform once at setup:
>>> cuda_markers, cuda_domain = transform(args_markers), transform(args_domain)
>>> catalog.get("push_eta_linear")(dt, stage, cuda_markers, cuda_domain)
"""

import math

import numpy as np
from cunumpy import PyccelKernel
from cunumpy.xp import array_backend

from struphy.utils.kernel_transform import CudaArguments


def is_cuda_backend() -> bool:
    """Whether the active cunumpy backend is CuPy."""
    return array_backend.backend == "cupy"


class CudaKernel:
    """Call a ``cupy.RawKernel`` with the 1:1 corresponding arguments of its pyccel counterpart.

    The pyccelized argument classes (``MarkerArguments`` etc.) must be transformed beforehand
    (once, at setup) with :func:`~struphy.utils.kernel_transform.transform`; no arrays are
    converted or copied at call time. Arrays must be CuPy arrays (``cupy`` raises otherwise).

    Parameters
    ----------
    source : str
        CUDA C source code containing an ``extern "C" __global__`` function ``name``.

    name : str
        Name of the kernel function in ``source``.

    block_size : int
        Number of threads per block.
    """

    def __init__(self, source: str, name: str, block_size: int = 128):
        self._source = source
        self._name = name
        self._block_size = block_size
        self._raw_kernel = None

    def __repr__(self):
        return f"CudaKernel(name={self.name!r}, block_size={self._block_size})"

    @property
    def name(self) -> str:
        """Name of the CUDA kernel."""
        return self._name

    @property
    def raw_kernel(self):
        """The compiled ``cupy.RawKernel`` (compiled lazily on first access)."""
        if self._raw_kernel is None:
            import cupy as cp

            self._raw_kernel = cp.RawKernel(self._source, self._name)
        return self._raw_kernel

    def __call__(self, *args):
        values = []
        n_threads = None
        for arg in args:
            if isinstance(arg, CudaArguments):
                values += arg.values
                if n_threads is None:
                    n_threads = arg.n_threads
            else:
                values.append(_cuda_scalar(arg))

        if n_threads is None:
            raise ValueError(f"{self.name}: no argument defines the number of CUDA threads (e.g. CudaMarkerArguments).")

        grid = (max(1, math.ceil(n_threads / self._block_size)),)
        self.raw_kernel(grid, (self._block_size,), tuple(values))


def _cuda_scalar(arg):
    """Python scalars as NumPy scalars with the C type of the kernel signature (int -> int, float -> double)."""
    if isinstance(arg, bool):
        return np.bool_(arg)
    if isinstance(arg, int):
        return np.int32(arg)
    if isinstance(arg, float):
        return np.float64(arg)
    return arg


class Kernel:
    """A pyccel kernel and its 1:1 corresponding CUDA kernel.

    Parameters
    ----------
    pyccel_kernel : PyccelKernel
        The pyccel kernel, used on the NumPy backend (and on CuPy if there is no CUDA kernel).

    cuda_kernel : CudaKernel | None
        The CUDA kernel, used on the CuPy backend.
    """

    def __init__(self, pyccel_kernel: PyccelKernel, cuda_kernel: CudaKernel | None = None):
        assert isinstance(pyccel_kernel, PyccelKernel), f"{pyccel_kernel} is not of type PyccelKernel"
        assert cuda_kernel is None or isinstance(cuda_kernel, CudaKernel), f"{cuda_kernel} is not of type CudaKernel"
        self._pyccel_kernel = pyccel_kernel
        self._cuda_kernel = cuda_kernel

    def __repr__(self):
        return f"Kernel(pyccel_kernel={self.pyccel_kernel!r}, cuda_kernel={self.cuda_kernel!r})"

    @property
    def pyccel_kernel(self) -> PyccelKernel:
        return self._pyccel_kernel

    @property
    def cuda_kernel(self) -> CudaKernel | None:
        return self._cuda_kernel

    @property
    def name(self) -> str:
        return self.pyccel_kernel.name

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """The kernel for the active cunumpy backend."""
        if is_cuda_backend() and self.cuda_kernel is not None:
            return self.cuda_kernel
        return self.pyccel_kernel

    def __call__(self, *args, **kwargs):
        return self.get_kernel()(*args, **kwargs)


class KernelCatalog:
    """Registry of :class:`Kernel` objects by name."""

    def __init__(self):
        self._kernels: dict[str, Kernel] = {}

    def register(self, kernel: Kernel, name: str | None = None) -> Kernel:
        """Register ``kernel`` under ``name`` (default: name of the pyccel kernel)."""
        name = kernel.name if name is None else name
        assert name not in self._kernels, f"Kernel {name!r} is already registered."
        self._kernels[name] = kernel
        return kernel

    def get(self, name: str) -> Kernel:
        return self._kernels[name]

    def __contains__(self, name: str) -> bool:
        return name in self._kernels

    @property
    def names(self) -> list[str]:
        return list(self._kernels)


catalog = KernelCatalog()
