"""Small, empirical domain-decomposition optimization helpers.

The current Struphy grid API exposes decomposition choices through
``mpi_dims_mask``.  This module measures those choices without knowing
anything about the model: the caller supplies a function which performs one
step for a given mask.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from feectools.ddm.mpi import mpi as MPI
from feectools.ddm.partition import compute_dims

Mask = tuple[bool, bool, bool]


@dataclass(frozen=True)
class DomainDecompositionTiming:
    """Measured wall time for one decomposition mask."""

    mask: Mask
    seconds: float


@dataclass(frozen=True)
class DomainDecompositionOptimization:
    """Result of an empirical decomposition search."""

    best_mask: Mask
    timings: tuple[DomainDecompositionTiming, ...]


def candidate_masks(
    num_elements: tuple[int, int, int],
    comm_size: int,
    *,
    min_local_elements: int = 1,
) -> tuple[Mask, ...]:
    """Return valid non-empty ``mpi_dims_mask`` candidates.

    A candidate is retained when :func:`feectools.ddm.partition.compute_dims`
    can construct a process grid for it.  The order is stable and starts with
    one-dimensional decompositions, followed by two- and three-dimensional
    decompositions.
    """

    if len(num_elements) != 3:
        raise ValueError("num_elements must contain exactly three dimensions")
    if comm_size < 1:
        raise ValueError("comm_size must be positive")
    if min_local_elements < 1:
        raise ValueError("min_local_elements must be positive")
    if any(int(n) <= 0 for n in num_elements):
        raise ValueError("num_elements must contain positive values")
    if int(comm_size) > num_elements[0] * num_elements[1] * num_elements[2]:
        raise ValueError("comm_size cannot exceed the number of grid elements")

    masks: list[Mask] = []
    for mask in itertools.product((False, True), repeat=3):
        if not any(mask):
            continue
        try:
            nprocs, blocksizes = compute_dims(comm_size, list(num_elements), mpi_dims_mask=list(mask))
        except (AssertionError, ValueError):
            continue
        # ``compute_dims`` accepts a process grid whose local block would have
        # zero cells in a short direction.  FEEC discretization cannot use
        # such a grid, so reject it here before invoking the user callback.
        if any(
            p > n or block < min_local_elements
            for p, n, block in zip(nprocs, num_elements, blocksizes)
        ):
            continue
        masks.append(mask)

    return tuple(sorted(masks, key=lambda mask: (sum(mask), mask)))


def optimize_domain_decomposition(
    num_elements: tuple[int, int, int],
    step: Callable[[Mask], object],
    *,
    comm=None,
    masks: Iterable[Mask] | None = None,
    min_local_elements: int = 1,
    warmups: int = 1,
    repetitions: int = 3,
) -> DomainDecompositionOptimization:
    """Select the fastest decomposition by timing a supplied one-step call.

    Parameters
    ----------
    num_elements
        Global grid resolution, used to reject invalid masks.
    step
        Callable that constructs/configures the candidate decomposition and
        performs exactly one timestep.  It is called collectively by all MPI
        ranks for every candidate.
    comm
        MPI communicator.  Defaults to ``MPI.COMM_WORLD``.
    masks
        Optional explicit subset of masks.  By default all valid masks are
        measured.
    warmups, repetitions
        Number of discarded and measured calls per candidate.  The reported
        time is the average of the communicator-wide maximum time per call.

    Returns
    -------
    DomainDecompositionOptimization
        All measured times and the mask with the lowest average timestep time.

    Notes
    -----
    This function does not reuse simulations between candidates.  A practical
    ``step`` callback should therefore build a fresh simulation for its mask,
    call ``sim.run(one_time_step=True)``, and clean up its output if needed.
    """

    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")

    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    valid_masks = set(
        candidate_masks(
            num_elements,
            comm_size,
            min_local_elements=min_local_elements,
        )
    )
    selected_masks = tuple(valid_masks if masks is None else _validate_masks(masks, valid_masks))
    if not selected_masks:
        raise ValueError("no valid decomposition masks were supplied")
    selected_masks = tuple(sorted(selected_masks, key=lambda mask: (sum(mask), mask)))

    timings: list[DomainDecompositionTiming] = []
    for mask in selected_masks:
        for _ in range(warmups):
            _barrier(comm)
            step(mask)
        samples = []
        for _ in range(repetitions):
            _barrier(comm)
            start = time.perf_counter()
            step(mask)
            elapsed = time.perf_counter() - start
            samples.append(_global_max(comm, elapsed))
        _barrier(comm)
        timings.append(DomainDecompositionTiming(mask=mask, seconds=sum(samples) / len(samples)))

    best = min(timings, key=lambda timing: timing.seconds)
    return DomainDecompositionOptimization(best_mask=best.mask, timings=tuple(timings))


def _validate_masks(masks: Iterable[Mask], valid_masks: set[Mask]) -> set[Mask]:
    selected = set()
    for mask in masks:
        normalized = tuple(mask)
        if len(normalized) != 3 or not all(isinstance(value, bool) for value in normalized):
            raise ValueError(f"invalid decomposition mask: {mask!r}")
        if normalized not in valid_masks:
            raise ValueError(f"decomposition mask is not valid for this grid/MPI size: {mask!r}")
        selected.add(normalized)
    return selected


def _barrier(comm) -> None:
    if comm.Get_size() > 1:
        comm.Barrier()


def _global_max(comm, value: float) -> float:
    if comm.Get_size() == 1:
        return value
    return float(comm.allreduce(value, op=MPI.MAX))
