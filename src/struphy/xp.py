import os


class ArrayBackend:

    def __init__(self, backend: str = "numpy") -> None:
        self._backend = backend

        # Import numpy/cupy
        if self.backend == "cupy":
            try:
                import cupy as cp

                return cp
            except ImportError:
                print("CuPy not available, falling back to NumPy.")
                self._backend = "numpy"

        if self.backend == "numpy":
            import numpy as np

            self._xp = np
            print(f"{self._xp = }")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def xp(self):
        return self._xp


# TODO: Make this configurable via environment variable or config file.
array_backend = ArrayBackend(
    backend=os.getenv("ARRAY_BACKEND", "numpy").lower(),
)

xp = array_backend.xp

print(f"Using {xp.__name__} backend.")
