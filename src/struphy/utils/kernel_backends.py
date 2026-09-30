"""Pairs of pyccel and CUDA kernels, selected at runtime from the cunumpy backend.

A :class:`Kernel` holds a pyccel kernel and its 1:1 corresponding CUDA kernel (:class:`CudaKernel`)
and calls the one matching the active cunumpy backend, see :func:`is_cuda_backend`.

The arguments must already be in the format of the kernel that is called; nothing is converted
at call time. For the CUDA kernel, this is the flat tuple of a ``cupy.RawKernel``: CuPy arrays and
NumPy scalars of the C types in the kernel signature (e.g. ``np.float64`` for ``double``,
``np.int32`` for ``int``). The argument classes are flattened with
:attr:`~struphy.utils.cuda_arguments.CudaMarkerArguments.values` etc. This tuple is meant to be
built once, at setup, not before every call.
"""

import math

from cunumpy import PyccelKernel
from cunumpy.xp import array_backend


def is_cuda_backend() -> bool:
    """Whether the active cunumpy backend is CuPy.

    Returns
    -------
    bool
        True if the backend is ``"cupy"``, False if it is ``"numpy"``.
    """
    return array_backend.backend == "cupy"


class CudaKernel:
    """A ``cupy.RawKernel``, the CUDA counterpart of a pyccel kernel.

    The kernel is compiled on the first call. Arguments are passed to the ``cupy.RawKernel`` as they are.

    Parameters
    ----------
    source : str
        CUDA C source code containing the ``extern "C" __global__`` function ``name``.

    name : str
        Name of the kernel function in ``source``.

    block_size : int
        Number of threads per block.
    """

    def __init__(self, source: str, name: str, block_size: int = 128):
        self.name = name
        self._source = source
        self._block_size = block_size
        self._raw_kernel = None

    def __call__(self, *args, n_threads: int):
        """Launch the kernel.

        Parameters
        ----------
        *args
            Kernel arguments in the format of a ``cupy.RawKernel``: CuPy arrays and NumPy scalars
            with the C types of the kernel signature.

        n_threads : int
            Number of threads to launch (e.g. number of markers); rounded up to a multiple of the block size.
        """
        if self._raw_kernel is None:
            import cupy as cp

            self._raw_kernel = cp.RawKernel(self._source, self.name)

        grid = (math.ceil(n_threads / self._block_size),)
        self._raw_kernel(grid, (self._block_size,), args)


class Kernel:
    """A pyccel kernel and its CUDA counterpart; calls the one matching the cunumpy backend.

    Parameters
    ----------
    pyccel_kernel : PyccelKernel
        The pyccel kernel, called on the NumPy backend.

    cuda_kernel : CudaKernel
        The CUDA kernel, called on the CuPy backend.
    """

    def __init__(self, pyccel_kernel: PyccelKernel, cuda_kernel: CudaKernel):
        self.pyccel_kernel = pyccel_kernel
        self.cuda_kernel = cuda_kernel

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """The kernel for the active cunumpy backend."""
        return self.cuda_kernel if is_cuda_backend() else self.pyccel_kernel

    def __call__(self, *args, n_threads: int | None = None):
        """Call the kernel for the active cunumpy backend.

        Parameters
        ----------
        *args
            Kernel arguments, already in the format of the kernel for the active backend.

        n_threads : int | None
            Number of CUDA threads; required on the CuPy backend, ignored on the NumPy backend.
        """
        if is_cuda_backend():
            if n_threads is None:
                raise ValueError(f"{self.cuda_kernel.name}: n_threads is required on the CuPy backend.")
            return self.cuda_kernel(*args, n_threads=n_threads)
        return self.pyccel_kernel(*args)
