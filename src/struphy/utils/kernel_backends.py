"""Pairs of pyccel and CUDA kernels, selected at runtime from the cunumpy backend.

A :class:`Kernel` holds a pyccel kernel and its 1:1 corresponding CUDA kernel (:class:`CudaKernel`)
and calls the one matching the active cunumpy backend, see :func:`is_cuda_backend`.

Both kernels are called with the same arguments; the CUDA kernel takes the CUDA versions of the
argument classes (:mod:`struphy.utils.cuda_arguments`), which reference arrays on the device, and
the number of threads ``n_threads``. No arrays are converted or copied at call time.

CUDA sources can include struphy headers relative to the parent folder of the ``struphy`` package,
e.g. ``#include "struphy/kernel_arguments/pusher_args.cuh"`` for the argument structs.
"""

import importlib
import math
import re

import numpy as np
from pathlib import Path

import cunumpy
from cunumpy import PyccelKernel

from struphy.utils.cuda_arguments import Argument

INCLUDE_DIR = Path(__file__).resolve().parents[2]
"""NVRTC include path of the CUDA kernels: the folder that contains the ``struphy`` package."""


def is_cuda_backend() -> bool:
    """Whether the active cunumpy backend is CuPy.

    Returns
    -------
    bool
        True if the backend is ``"cupy"``, False if it is ``"numpy"``.
    """
    return cunumpy.get_backend() == "cupy"


class CudaKernel:
    """A ``cupy.RawKernel``, the CUDA counterpart of a pyccel kernel.

    The kernel is compiled on the first call, with :data:`INCLUDE_DIR` on the include path. At each call, the
    CUDA argument classes are replaced by their structs; all other arguments are passed to the ``cupy.RawKernel``
    as they are. Arrays must be CuPy
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
        signature = re.search(r"void\s+" + re.escape(name) + r"\s*\((.*?)\)", source, re.S)
        self._view_indices = {i for i, arg in enumerate(signature.group(1).split(",")) if "Array3D<double>" in arg} if signature else set()

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
        if n_threads < 0:
            raise ValueError("n_threads must be nonnegative")
        if n_threads == 0:
            return
        if self._raw_kernel is None:
            import cupy as cp

            self._raw_kernel = cp.RawKernel(self._source, self.name, options=(f"-I{INCLUDE_DIR}",))

        values = []
        for index, arg in enumerate(args):
            if index in self._view_indices:
                import cupy as cp

                if not isinstance(arg, cp.ndarray) or arg.ndim != 3 or arg.dtype != np.float64:
                    raise TypeError("Array3D<double> requires a three-dimensional float64 device array")
                if arg.device.id != cp.cuda.runtime.getDevice():
                    raise ValueError("Array view must be on the current CUDA device")
                view = np.zeros((), dtype=np.dtype([("data", np.uint64), ("shape", np.int64, 3), ("strides", np.int64, 3)], align=True))
                view["data"] = arg.data.ptr
                view["shape"] = arg.shape
                view["strides"] = tuple(s // arg.itemsize for s in arg.strides)
                values.append(view[()])
            elif isinstance(arg, Argument):
                values.extend(arg.get_cuda_args())
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

    cuda_kernel : CudaKernel | None
        The CUDA kernel, called on the CuPy backend; None if it has not been ported yet
        (then the kernel cannot run on the CuPy backend).

    cuda_path : Path | None
        Where the CUDA kernel is expected, used in the error message if it is missing.
    """

    def __init__(
        self,
        pyccel_kernel: PyccelKernel,
        cuda_kernel: CudaKernel | None = None,
        cuda_path: Path | None = None,
    ):
        self.pyccel_kernel = pyccel_kernel
        self.cuda_kernel = cuda_kernel
        self.cuda_path = cuda_path

    @property
    def name(self) -> str:
        """Name of the kernel (the name of the pyccel kernel)."""
        return self.pyccel_kernel.name

    def get_kernel(self) -> PyccelKernel | CudaKernel:
        """The kernel for the active cunumpy backend.

        Raises
        ------
        NotImplementedError
            On the CuPy backend, if there is no CUDA version of the kernel.
        """
        if not is_cuda_backend():
            return self.pyccel_kernel
        if self.cuda_kernel is None:
            expected = "" if self.cuda_path is None else f" (expected {self.cuda_path})"
            raise NotImplementedError(f"No CUDA version of kernel {self.name!r}{expected}.")
        return self.cuda_kernel

    def __call__(self, *args, n_threads: int | None = None):
        """Call the kernel for the active cunumpy backend.

        Parameters
        ----------
        *args
            Kernel arguments; on the CuPy backend with the CUDA versions of the argument classes.

        n_threads : int | None
            Number of CUDA threads; required on the CuPy backend, ignored on the NumPy backend.
        """
        kernel = self.get_kernel()
        if not is_cuda_backend():
            return kernel(*args)
        if n_threads is None:
            raise ValueError(f"{kernel.name}: n_threads is required on the CuPy backend.")
        return kernel(*args, n_threads=n_threads)


class KernelCatalog:
    """The kernels of a package with one folder per kernel.

    For each subfolder ``<name>`` containing ``<name>_kernels.py`` (pyccel), the function ``<name>`` in that
    module is the pyccel kernel, and ``<name>_cuda.cu`` in the same folder, if present, is the CUDA kernel::

        catalog = KernelCatalog.from_package(__name__)  # in the __init__.py of the package
        kernel = catalog["push_eta_stage"]

    Parameters
    ----------
    kernels : dict[str, Kernel]
        The kernels by name.
    """

    def __init__(self, kernels: dict[str, Kernel]):
        self._kernels = kernels

    @classmethod
    def from_package(cls, package: str) -> "KernelCatalog":
        """Collect the kernels in the subfolders of a package.

        Parameters
        ----------
        package : str
            Full name of the package, e.g. ``__name__`` in its ``__init__.py``.

        Returns
        -------
        KernelCatalog
            One :class:`Kernel` per subfolder ``<name>`` with a file ``<name>_kernels.py``.
        """
        root = Path(importlib.import_module(package).__file__).parent
        kernels = {}
        for folder in sorted(p for p in root.iterdir() if (p / f"{p.name}_kernels.py").is_file()):
            name = folder.name
            module = importlib.import_module(f"{package}.{name}.{name}_kernels")
            cuda_path = folder / f"{name}_cuda.cu"
            cuda_kernel = CudaKernel.from_file(cuda_path) if cuda_path.is_file() else None
            kernels[name] = Kernel(PyccelKernel(getattr(module, name)), cuda_kernel, cuda_path)
        return cls(kernels)

    def __getitem__(self, name: str) -> Kernel:
        return self._kernels[name]

    def __contains__(self, name: str) -> bool:
        return name in self._kernels

    @property
    def names(self) -> list[str]:
        """Names of all kernels."""
        return list(self._kernels)

    @property
    def missing_cuda(self) -> list[str]:
        """Names of the kernels without a CUDA version."""
        return [name for name, kernel in self._kernels.items() if kernel.cuda_kernel is None]
