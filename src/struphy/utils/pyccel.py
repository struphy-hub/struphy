from typing import Any, Callable


class Pyccelkernel:
    def __init__(self, kernel: Callable[..., Any], use_cupy: bool = False) -> None:
        self._kernel = kernel
        self._use_cupy = use_cupy

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_cupy:
            raise NotImplementedError
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
