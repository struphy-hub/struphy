"""Pairs of pyccel and CUDA kernels, selected at runtime from the cunumpy backend.

A :class:`Kernel` holds a pyccel kernel and its 1:1 corresponding CUDA kernel (:class:`CudaKernel`)
and calls the one matching the active cunumpy backend, see :func:`is_cuda_backend`.

Both kernels are called with the same arguments; the CUDA kernel takes the CUDA versions of the
argument classes (:mod:`struphy.utils.cuda_arguments`), which reference arrays on the device, and
the number of threads ``n_threads``. No arrays are converted or copied at call time.
"""

import math
from pathlib import Path

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

    The kernel is compiled on the first call. At each call, the CUDA argument classes are replaced by their
    ``values``; all other arguments are passed to the ``cupy.RawKernel`` as they are. Arrays must be CuPy
    arrays (``cupy`` raises otherwise); they are never converted or copied. Python ``int`` and ``float``
    arrive correctly in ``int`` and ``double`` parameters; scalars are not checked against the kernel signature.

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

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None, block_size: int = 128) -> "CudaKernel":
        """Load the CUDA source from a ``<name>_cuda.cu`` file.

        Parameters
        ----------
        path : str | Path
            Path of the ``.cu`` file.

        name : str | None
            Name of the kernel function; defaults to the file name without ``_cuda.cu``.

        block_size : int
            Number of threads per block.

        Returns
        -------
        CudaKernel
            The (not yet compiled) CUDA kernel.
        """
        path = Path(path)
        if name is None:
            assert path.name.endswith("_cuda.cu"), f"{path.name} does not follow the naming convention <name>_cuda.cu"
            name = path.name[: -len("_cuda.cu")]
        return cls(path.read_text(), name, block_size=block_size)

    def __call__(self, *args, n_threads: int):
        """Launch the kernel.

        Parameters
        ----------
        *args
            The arguments of the pyccel kernel, with the CUDA versions of the argument classes
            (:mod:`struphy.utils.cuda_arguments`), CuPy arrays and Python or NumPy scalars.

        n_threads : int
            Number of threads to launch (e.g. number of markers); rounded up to a multiple of the block size.
        """
        if self._raw_kernel is None:
            import cupy as cp

            self._raw_kernel = cp.RawKernel(self._source, self.name)

        values = []
        for arg in args:
            if hasattr(arg, "values"):
                values += arg.values
            else:
                values.append(arg)

        grid = (math.ceil(n_threads / self._block_size),)
        self._raw_kernel(grid, (self._block_size,), tuple(values))


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
            Kernel arguments; on the CuPy backend with the CUDA versions of the argument classes.

        n_threads : int | None
            Number of CUDA threads; required on the CuPy backend, ignored on the NumPy backend.
        """
        if is_cuda_backend():
            if n_threads is None:
                raise ValueError(f"{self.cuda_kernel.name}: n_threads is required on the CuPy backend.")
            return self.cuda_kernel(*args, n_threads=n_threads)
        return self.pyccel_kernel(*args)
