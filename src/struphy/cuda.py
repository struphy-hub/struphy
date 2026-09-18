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

    ``source`` may be a string, or a zero-argument callable that builds one
    (matching the old per-module ``_source()`` helpers some of these files
    used to defer assembling/importing a source shared with another module
    until first use).
    """

    __slots__ = ("_source", "_name", "_kernel")

    def __init__(self, source, name: str) -> None:
        self._source = source
        self._name = name
        self._kernel = None

    def __call__(self, grid, block, args) -> None:
        self._compiled()(grid, block, args)

    def compile(self) -> None:
        """Force NVRTC compilation now instead of on first launch.

        For a kernel on the hot path of the first timed step (e.g. a
        model-setup routine that wants compile latency paid during setup, not
        during the first measured propagation step): call this eagerly:
        ``compile()`` is idempotent, so it composes fine with the normal
        lazy-on-first-launch path -- whichever happens first wins, and later
        calls (from either path) are no-ops.
        """
        self._compiled()

    def _compiled(self):
        if self._kernel is None:
            import cupy as cp

            source = self._source() if callable(self._source) else self._source
            kernel = cp.RawKernel(source, self._name)
            kernel.compile()
            self._kernel = kernel
        return self._kernel


class CudaKernelSet:
    """A cache of :class:`CudaKernel` instances sharing one CUDA source,
    keyed by kernel name.

    For modules exposing a *family* of kernel entry points compiled from the
    same source (typically a device-function library plus several
    ``__global__`` entry points), where the entry point needed depends on a
    runtime string (``u_space``, ``algo``, a diffusion variant, ...) rather
    than being fixed at import time. Replaces the old ``_kernels = {}`` dict +
    ``_get_kernel(name)`` function pattern -- ``kernels[name]`` compiles and
    caches lazily, exactly like the old lookup did.

    ``source`` may be a string (shared eagerly, e.g. one ``load_cuda_source``
    result) or a zero-argument callable that builds it (for sources
    concatenated from several fragments -- matching the old per-module
    ``_source()`` helper -- so that assembly, and any ``cupy`` import inside
    it, stays deferred to first use).
    """

    __slots__ = ("_source", "_cache")

    def __init__(self, source) -> None:
        self._source = source
        self._cache: dict[str, CudaKernel] = {}

    def __getitem__(self, name: str) -> CudaKernel:
        if name not in self._cache:
            source = self._source() if callable(self._source) else self._source
            self._cache[name] = CudaKernel(source, name)
        return self._cache[name]


def launch_1d(kernel: CudaKernel, n: int, args: Sequence, threads: int = 256) -> None:
    """Launch ``kernel`` over a 1-D grid with one thread per element of a
    length-``n`` array (markers, indices, quadrature points, ...) -- the
    launch geometry shared by every kernel in ``struphy.pic.*_cuda`` /
    ``struphy.feec.*_cuda``."""
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), tuple(args))
