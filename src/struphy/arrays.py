import os
from types import ModuleType
from typing import TYPE_CHECKING, Literal

BackendType = Literal["numpy", "cupy"]


class ArrayBackend:
    def __init__(
        self,
        backend: BackendType = "numpy",
        verbose: bool = False,
    ) -> None:
        assert backend.lower() in ["numpy", "cupy"], "Array backend must be either 'numpy' or 'cupy'."

        self._backend: BackendType = "cupy" if backend.lower() == "cupy" else "numpy"

        # Import numpy/cupy
        if self.backend == "cupy":
            try:
                import cupy as cp

                self._xp = cp
            except ImportError:
                if verbose:
                    print("CuPy not available.")
                self._backend = "numpy"

        if self.backend == "numpy":
            import numpy as np

            self._xp = np

        assert isinstance(self.xp, ModuleType)

        if verbose:
            print(f"Using {self.xp.__name__} backend.")

    @property
    def backend(self) -> BackendType:
        return self._backend

    @property
    def xp(self) -> ModuleType:
        return self._xp


# TODO: Make this configurable via environment variable or config file.
array_backend = ArrayBackend(
    backend="cupy" if os.getenv("ARRAY_BACKEND", "numpy").lower() == "cupy" else "numpy",
    verbose=True,
)

# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    import numpy as xp
else:
    xp = array_backend.xp
