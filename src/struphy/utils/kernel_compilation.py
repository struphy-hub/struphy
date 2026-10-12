"""Collect the kernels a simulation calls and compile their CUDA versions before the time stepping.

A CUDA kernel is otherwise compiled on its first call, which then shows up in the profile of the first
time step. The objects that keep and call :class:`cunumpy.kernels.Kernel` objects (``Pusher``,
``KernelSetup``, ``Accumulator``, ``AccumulatorVector``, ``Particles``, ``Domain``, ``Derham``, the
propagators) list them with a ``kernels()`` method; :func:`collect_kernels` gathers them and
:meth:`~struphy.simulation.sim.Simulation.compile_cuda_kernels` compiles them once at setup.
"""

from collections.abc import Iterable, Iterator, Mapping

from cunumpy.kernels import Kernel


def _walk(value, seen: dict[int, object]) -> Iterator[Kernel]:
    # `seen` keeps the visited objects alive, so that the id of a temporary (e.g. a tuple returned
    # by ``kernels()``) is not reused by another object during the walk
    if id(value) in seen:
        return
    seen[id(value)] = value
    if isinstance(value, Kernel):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk(item, seen)
    elif not isinstance(value, type) and callable(getattr(value, "kernels", None)):
        for item in value.kernels():
            yield from _walk(item, seen)


def collect_kernels(*owners) -> tuple[Kernel, ...]:
    """The kernels of `owners`, each once, in the order they are found.

    An owner is a :class:`~cunumpy.kernels.Kernel`, an object with a ``kernels()`` method (whose
    result is collected the same way), or a list, tuple, set or dict of owners. Anything else is
    ignored, so ``collect_kernels(*vars(obj).values())`` collects the kernels of the attributes of
    ``obj`` without descending into other objects.
    """
    seen: dict[int, object] = {}
    kernels: dict[int, Kernel] = {}
    for owner in owners:
        for kernel in _walk(owner, seen):
            kernels.setdefault(id(kernel), kernel)
    return tuple(kernels.values())


def missing_cuda(kernels: Iterable[Kernel]) -> list[Kernel]:
    """The kernels that would raise on the CuPy backend because they have no CUDA version."""
    return [k for k in kernels if not k.has_cuda and k.missing_cuda == "raise"]
