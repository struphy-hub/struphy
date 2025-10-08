import array
import os

# TODO: Make this configurable via environment variable or config file.
# Default to numpy
_backend = os.getenv("ARRAY_BACKEND", "numpy").lower()


class ArrayBackend:
    def __init__(self, backend: str = "numpy") -> None:
        self._backend = backend
        pass

    def _import_numpy(self):
        import numpy as np

        return np

    def _import_cupy(self):
        # print("importing cupy...")
        try:
            import cupy as cp

            return cp
        except ImportError:
            print("CuPy not available, falling back to NumPy.")
            return self._import_numpy()

    @property
    def backend(self):
        return self._backend

    @property
    def xp(self):
        # Import numpy/cupy
        if _backend == "cupy":
            return self._import_cupy()
        else:
            return self._import_numpy()


array_backend = ArrayBackend(_backend)

xp = array_backend.xp

print(f"Using {xp} backend.")
