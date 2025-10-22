from typing import Any, Callable

import cunumpy as xp


class Pyccelkernel:
    def __init__(self, kernel: Callable[..., Any], use_cupy: bool = False) -> None:
        self._kernel = kernel
        self._use_cupy = use_cupy
        if "cupy" in xp.__name__:
            self._use_cupy = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_cupy:
            # Convert all args from CuPy to NumPy
            args_np = [x.get() if isinstance(x, xp.ndarray) else x for x in args]
            # Convert all kwargs from CuPy to NumPy
            kwargs_np = {k: v.get() if isinstance(v, xp.ndarray) else v for k, v in kwargs.items()}
            return self._kernel(*args_np, **kwargs_np)
        else:
            return self._kernel(*args, **kwargs)

    @property
    def name(self):
        return self.kernel.__name__

    @property
    def kernel(self) -> Callable[..., Any]:
        return self._kernel

    @property
    def use_cupy(self):
        return self._use_cupy
