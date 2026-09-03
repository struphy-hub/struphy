"""Helpers for loading CUDA C sources used by CuPy RawKernel wrappers."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_cuda_source(module_file: str, source_name: str) -> str:
    """Load a CUDA C source fragment stored alongside its Python wrapper."""
    path = Path(module_file).with_name("cuda") / source_name
    return path.read_text(encoding="utf-8")
