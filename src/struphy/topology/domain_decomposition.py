"""Small, empirical domain-decomposition optimization helpers.

The current Struphy grid API exposes decomposition choices through
``mpi_dims_mask``.  This module measures those choices without knowing
anything about the model: the caller supplies a function which performs one
step for a given mask.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from feectools.ddm.mpi import mpi as MPI
from feectools.ddm.partition import compute_dims

Mask = tuple[bool, bool, bool]
LocalMinimum = int | tuple[int, int, int]
MaskPattern = tuple[
    bool | int | Literal["auto"],
    bool | int | Literal["auto"],
    bool | int | Literal["auto"],
]


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
    min_local_elements: LocalMinimum = 1,
    mask_pattern: MaskPattern | None = None,
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
    if isinstance(min_local_elements, int):
        local_minimum = (min_local_elements,) * 3
    else:
        if len(min_local_elements) != 3:
            raise ValueError("min_local_elements must contain exactly three dimensions")
        local_minimum = tuple(int(value) for value in min_local_elements)
    if any(value < 1 for value in local_minimum):
        raise ValueError("min_local_elements must be positive")
    if any(int(n) <= 0 for n in num_elements):
        raise ValueError("num_elements must contain positive values")
    if int(comm_size) > num_elements[0] * num_elements[1] * num_elements[2]:
        raise ValueError("comm_size cannot exceed the number of grid elements")
    pattern = _normalize_mask_pattern(mask_pattern)

    masks: list[Mask] = []
    for mask in itertools.product((False, True), repeat=3):
        if not any(mask):
            continue
        if pattern is not None and any(
            isinstance(expected, bool) and value != expected
            for value, expected in zip(mask, pattern)
        ):
            continue
        if pattern is not None and any(
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and ((expected > 1 and not value) or (expected == 1 and value))
            for value, expected in zip(mask, pattern)
        ):
            continue
        try:
            nprocs, blocksizes = compute_dims(comm_size, list(num_elements), mpi_dims_mask=list(mask))
        except (AssertionError, ValueError):
            continue
        # ``compute_dims`` accepts a process grid whose local block is too
        # small in a short direction.  FEEC discretization cannot use such a
        # grid (the required minimum is normally the spline degree), so reject
        # it here before invoking the user callback.
        if any(
            p > n or block < minimum
            for p, n, block, minimum in zip(nprocs, num_elements, blocksizes, local_minimum)
        ):
            continue
        if pattern is not None and any(
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and nproc != expected
            for nproc, expected in zip(nprocs, pattern)
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
    min_local_elements: LocalMinimum = 1,
    mask_pattern: MaskPattern | None = None,
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
    min_local_elements
        Minimum number of elements in every local block. This can be one
        integer for all directions or a three-tuple for direction-specific
        requirements. For FEEC grids, pass the spline degree tuple.
    mask_pattern
        Optional per-direction constraint. Boolean entries fix whether a
        direction may be decomposed, positive integers require an exact
        process-grid extent, and ``"auto"`` lets the optimizer vary it. For
        example, ``(1, "auto", "auto")`` requires one process in the first
        direction.
    mask_pattern
        Optional per-direction constraint. Use ``True`` or ``False`` to fix a
        direction and ``"auto"`` to let the optimizer vary it, e.g.
        ``(True, "auto", "auto")``.

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
            mask_pattern=mask_pattern,
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


def _normalize_mask_pattern(mask_pattern: MaskPattern | None) -> MaskPattern | None:
    if mask_pattern is None:
        return None
    if len(mask_pattern) != 3:
        raise ValueError("mask_pattern must contain exactly three dimensions")
    if not all(
        value == "auto" or isinstance(value, bool) or isinstance(value, int) and value >= 1
        for value in mask_pattern
    ):
        raise ValueError("mask_pattern entries must be True, False, a positive integer, or 'auto'")
    return mask_pattern


def _barrier(comm) -> None:
    if comm.Get_size() > 1:
        comm.Barrier()


def _global_max(comm, value: float) -> float:
    if comm.Get_size() == 1:
        return value
    return float(comm.allreduce(value, op=MPI.MAX))
