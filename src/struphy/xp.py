import os
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from numpy import isin

BackendType = Literal["numpy", "cupy"]


class ArrayBackend:
    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        self._backend = backend

        # Import numpy/cupy
        if self.backend == "cupy":
            try:
                import cupy as cp

                self._xp = cp
            except ImportError:
                if verbose:
                    print("CuPy not available, falling back to NumPy.")
                self._backend = "numpy"

        if self.backend == "numpy":
            import numpy as np

            self._xp = np

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        assert isinstance(self._xp, ModuleType)
        return self._xp


# TODO: Make this configurable via environment variable or config file.
array_backend = ArrayBackend(
    backend=os.getenv("ARRAY_BACKEND", "numpy").lower(),
)

if TYPE_CHECKING:
    import numpy as xp
else:
    xp = array_backend.xp

print(f"Using {xp.__name__} backend.")
