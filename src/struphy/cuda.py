"""Helpers for loading CUDA C sources used by CuPy RawKernel wrappers."""

from functools import lru_cache
from pathlib import Path
from typing import Sequence


@lru_cache(maxsize=None)
def load_cuda_source(module_file: str, source_name: str) -> str:
    """Load a CUDA C source fragment stored alongside its Python wrapper."""
    path = Path(module_file).with_name("cuda") / source_name
    return path.read_text(encoding="utf-8")


class CudaKernel:
    """A lazily-compiled, cached ``cupy.RawKernel``.

    Every hand-written CUDA replacement in ``struphy.pic.*_cuda`` /
    ``struphy.feec.*_cuda`` used to repeat the same boilerplate at each call
    site: a module-level ``_foo_kernel = None`` sentinel, a
    ``_get_foo_kernel()`` function that imports ``cupy`` and compiles the
    ``RawKernel`` the first time it's needed (so importing these modules
    under ``ARRAY_BACKEND=numpy`` never touches CuPy), and caches it back into
    the global. This class replaces that boilerplate with one declaration:

        _foo_kernel = CudaKernel(_FOO_SRC, "foo_cuda")

    made once at module level, right next to the ``*_gpu`` function it backs.
    Compilation is still deferred to first call (``import cupy`` only happens
    inside :meth:`__call__`), and the compiled kernel is cached on the
    instance -- identical behavior to the old pattern, but now every real GPU
    kernel a module launches shows up as a ``CudaKernel(...)`` at module
    scope, so `grep -n "CudaKernel("` (or just reading the top of the file)
    tells you exactly which functions do device work and which don't.
    """

    __slots__ = ("_source", "_name", "_kernel")

    def __init__(self, source: str, name: str) -> None:
        self._source = source
        self._name = name
        self._kernel = None

    def __call__(self, grid, block, args) -> None:
        self._compiled()(grid, block, args)

    def _compiled(self):
        if self._kernel is None:
            import cupy as cp

            kernel = cp.RawKernel(self._source, self._name)
            kernel.compile()
            self._kernel = kernel
        return self._kernel


def launch_1d(kernel: CudaKernel, n: int, args: Sequence, threads: int = 256) -> None:
    """Launch ``kernel`` over a 1-D grid with one thread per element of a
    length-``n`` array (markers, indices, quadrature points, ...) -- the
    launch geometry shared by every kernel in ``struphy.pic.*_cuda`` /
    ``struphy.feec.*_cuda``."""
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), tuple(args))
