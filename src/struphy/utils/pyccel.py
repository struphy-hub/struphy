import copy
from typing import Any, Callable

import cunumpy as xp
import numpy as np


class Pyccelkernel:
    def __init__(self, kernel: Callable[..., Any], use_cupy: bool = False) -> None:
        self._kernel = kernel
        self._use_cupy = use_cupy
        if "cupy" in xp.__name__ or "cupy" in xp.ndarray.__module__:
            self._use_cupy = True

    @staticmethod
    def _convert_to_numpy(value: Any, converted_arrays: list[tuple[Any, np.ndarray]]) -> Any:
        if isinstance(value, xp.ndarray):
            value_np = xp.asnumpy(value)
            converted_arrays.append((value, value_np))
            return value_np

        if isinstance(value, tuple):
            return tuple(Pyccelkernel._convert_to_numpy(item, converted_arrays) for item in value)

        if isinstance(value, list):
            return [Pyccelkernel._convert_to_numpy(item, converted_arrays) for item in value]

        if hasattr(value, "__dict__") and value.__class__.__module__.startswith(("struphy.", "feectools.")):
            value_np = copy.copy(value)
            for name, attr in vars(value).items():
                setattr(value_np, name, Pyccelkernel._convert_to_numpy(attr, converted_arrays))
            return value_np

        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_cupy:
            # Convert all args from CuPy to NumPy
            converted_args = []
            args_np = [self._convert_to_numpy(x, converted_args) for x in args]

            # Convert all kwargs from CuPy to NumPy
            converted_kwargs = []
            kwargs_np = {k: self._convert_to_numpy(v, converted_kwargs) for k, v in kwargs.items()}

            # Call kernel
            result = self._kernel(*args_np, **kwargs_np)

            # Copy in-place kernel updates back to CuPy arrays.
            for x, x_np in converted_args:
                x[...] = xp.asarray(x_np)
            for v, v_np in converted_kwargs:
                v[...] = xp.asarray(v_np)

            # Convert NumPy arrays back to CuPy
            if result is None:
                return None
            if isinstance(result, tuple):
                return tuple(xp.asarray(r) if isinstance(r, np.ndarray) else r for r in result)
            if isinstance(result, np.ndarray):
                return xp.asarray(result)
            return result

        else:
            return self._kernel(*args, **kwargs)

    @property
    def name(self) -> str:
        return self.kernel.__name__

    @property
    def kernel(self) -> Callable[..., Any]:
        return self._kernel

    @property
    def use_cupy(self) -> bool:
        return self._use_cupy
