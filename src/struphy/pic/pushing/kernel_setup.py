"""Explicit setup for kernels that evaluate into particle marker columns."""

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral, Real

import cunumpy as xp
from cunumpy import PyccelKernel


@dataclass(frozen=True, kw_only=True, eq=False)
class KernelSetup:
    """A marker evaluation kernel and all of its configuration.

    ``output_indices`` gives the destination marker column for each result
    component: one index for a scalar, three for a vector, or nine in row-major
    order for a tensor. Use ``None`` to skip a component. For example,
    ``(20, None, 24)`` writes vector components 0 and 2 to columns 20 and 24.

    ``alpha`` selects the evaluation state, coordinate by coordinate:
    ``alpha * current + (1 - alpha) * initial``. A scalar applies to all six
    phase-space coordinates; a tuple supplies three to six weights explicitly,
    with omitted velocity weights set to zero.
    Initialization kernels use zero weights. The arrays needed by compiled
    kernels are prepared once; ``args`` retains references to mutable field data.

    Kernels follow the signature
    ``kernel(alpha, output_indices, args_markers, args_domain, *args)``.
    At this boundary, a skipped output is represented by the integer -1.
    """

    kernel: Callable
    args: tuple = field(default=(), repr=False)
    output_indices: tuple[int | None, ...]
    alpha: float | tuple[float, ...] = 0.0
    _output_indices_array: xp.ndarray = field(init=False, repr=False)
    _alpha_array: xp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not callable(self.kernel):
            raise TypeError("kernel must be callable")
        if not isinstance(self.args, tuple):
            raise TypeError("args must be a tuple of kernel arguments")

        indices = tuple(self.output_indices)
        if len(indices) not in (1, 3, 9):
            raise ValueError("output_indices must contain 1, 3, or 9 component destinations")
        for index in indices:
            if index is not None and (isinstance(index, bool) or not isinstance(index, Integral) or index < 0):
                raise ValueError("output indices must be nonnegative integers or None")
        destinations = tuple(int(index) for index in indices if index is not None)
        if not destinations or len(set(destinations)) != len(destinations):
            raise ValueError("output_indices must contain distinct destinations and at least one output")

        alpha = (float(self.alpha),) * 6 if isinstance(self.alpha, Real) else tuple(self.alpha)
        if not 3 <= len(alpha) <= 6 or any(not 0.0 <= value <= 1.0 for value in alpha):
            raise ValueError("alpha must supply three to six weights between 0 and 1")
        alpha += (0.0,) * (6 - len(alpha))
        object.__setattr__(self, "output_indices", indices)
        object.__setattr__(self, "alpha", tuple(float(value) for value in alpha))
        object.__setattr__(
            self, "_output_indices_array", xp.array([-1 if i is None else i for i in indices], dtype=int)
        )
        object.__setattr__(self, "_alpha_array", xp.array(alpha, dtype=float))
        if not isinstance(self.kernel, PyccelKernel):
            object.__setattr__(self, "kernel", PyccelKernel(self.kernel))

    @property
    def name(self) -> str:
        """Kernel name used in profiling and validation messages."""
        return self.kernel.name

    @property
    def sorting_alpha(self):
        """Spatial evaluation weights used for MPI sorting before execution."""
        return self._alpha_array[:3]

    def validate_outputs(self, n_cols: int):
        """Check that every destination fits in the marker array."""
        for index in self.output_indices:
            if index is not None and index >= n_cols:
                raise ValueError(f"{self.name}: output column {index} is outside a marker array with {n_cols} columns")

    def evaluate(self, args_markers, args_domain):
        """Evaluate into the configured marker columns."""
        self.kernel(self._alpha_array, self._output_indices_array, args_markers, args_domain, *self.args)
