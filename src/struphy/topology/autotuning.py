"""Small empirical autotuning helpers for runtime configuration choices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from feectools.ddm.mpi import mpi as MPI


@dataclass(frozen=True)
class ParameterTiming:
    value: int
    seconds: float


@dataclass(frozen=True)
class ParameterOptimization:
    best_value: int
    timings: tuple[ParameterTiming, ...]


def optimize_integer_parameter(
    values: Iterable[int],
    step: Callable[[int], object],
    *,
    comm=None,
    warmups: int = 1,
    repetitions: int = 3,
) -> ParameterOptimization:
    """Select the fastest integer parameter using communicator-wide timings.

    ``step(value)`` must be called collectively and should perform the same
    fixed amount of simulation work for each candidate value.
    """
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    selected = tuple(dict.fromkeys(int(value) for value in values))
    if not selected:
        raise ValueError("at least one parameter value is required")
    if any(value < 0 for value in selected):
        raise ValueError("parameter values must be non-negative")
    if comm is None:
        comm = MPI.COMM_WORLD

    timings = []
    for value in selected:
        for _ in range(warmups):
            _barrier(comm)
            step(value)
        samples = []
        for _ in range(repetitions):
            _barrier(comm)
            start = time.perf_counter()
            step(value)
            elapsed = time.perf_counter() - start
            samples.append(_global_max(comm, elapsed))
        _barrier(comm)
        timings.append(ParameterTiming(value=value, seconds=sum(samples) / len(samples)))

    best = min(timings, key=lambda timing: timing.seconds)
    return ParameterOptimization(best_value=best.value, timings=tuple(timings))


def optimize_sorting_frequency(
    frequencies: Iterable[int],
    step: Callable[[int], object],
    *,
    comm=None,
    warmups: int = 1,
    repetitions: int = 3,
) -> ParameterOptimization:
    """Select the fastest particle-sorting frequency.

    Frequency ``0`` is supported and means that no periodic sorting is done;
    the callback remains responsible for performing any initial sort required
    by the simulation.
    """
    return optimize_integer_parameter(
        frequencies,
        step,
        comm=comm,
        warmups=warmups,
        repetitions=repetitions,
    )


def search_integer_parameter(
    lower: int,
    upper: int,
    step: Callable[[int], object],
    *,
    coarse_step: int = 4,
    refinement_radius: int = 2,
    comm=None,
    warmups: int = 1,
    repetitions: int = 3,
) -> ParameterOptimization:
    """Search an integer interval with a coarse pass and local refinement."""
    if lower < 0 or upper < lower:
        raise ValueError("expected 0 <= lower <= upper")
    if coarse_step < 1 or refinement_radius < 0:
        raise ValueError("coarse_step must be positive and refinement_radius non-negative")

    coarse_values = list(range(lower, upper + 1, coarse_step))
    if coarse_values[-1] != upper:
        coarse_values.append(upper)
    coarse_result = optimize_integer_parameter(
        coarse_values,
        step,
        comm=comm,
        warmups=warmups,
        repetitions=repetitions,
    )

    fine_lower = max(lower, coarse_result.best_value - refinement_radius)
    fine_upper = min(upper, coarse_result.best_value + refinement_radius)
    fine_values = range(fine_lower, fine_upper + 1)
    fine_result = optimize_integer_parameter(
        fine_values,
        step,
        comm=comm,
        warmups=warmups,
        repetitions=repetitions,
    )

    timings = coarse_result.timings + tuple(
        timing for timing in fine_result.timings if timing.value not in coarse_values
    )
    best = min(timings, key=lambda timing: timing.seconds)
    return ParameterOptimization(best_value=best.value, timings=timings)


def search_sorting_frequency(
    upper: int,
    step: Callable[[int], object],
    *,
    lower: int = 0,
    coarse_step: int = 4,
    refinement_radius: int = 2,
    comm=None,
    warmups: int = 1,
    repetitions: int = 3,
) -> ParameterOptimization:
    """Adaptively search a sorting-frequency interval."""
    return search_integer_parameter(
        lower,
        upper,
        step,
        coarse_step=coarse_step,
        refinement_radius=refinement_radius,
        comm=comm,
        warmups=warmups,
        repetitions=repetitions,
    )


def _barrier(comm) -> None:
    if comm.Get_size() > 1:
        comm.Barrier()


def _global_max(comm, value: float) -> float:
    if comm.Get_size() == 1:
        return value
    return float(comm.allreduce(value, op=MPI.MAX))
