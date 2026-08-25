import pytest

import struphy.topology.domain_decomposition as dd
import struphy.topology.autotuning as tuning
from struphy.topology.domain_decomposition import (
    candidate_clone_counts,
    candidate_masks,
    optimize_domain_decomposition,
    optimize_parallel_configuration,
)
from struphy.topology.autotuning import optimize_integer_parameter, search_integer_parameter


def test_candidate_clone_counts_are_divisors():
    assert candidate_clone_counts(8) == (1, 2, 4, 8)


def test_integer_parameter_optimizer_selects_fastest_value(monkeypatch):
    clock = [0.0]
    costs = {0: 0.03, 1: 0.01, 2: 0.02}
    monkeypatch.setattr(tuning.time, "perf_counter", lambda: clock[0])

    def step(value):
        clock[0] += costs[value]

    result = optimize_integer_parameter((0, 1, 2), step, warmups=0, repetitions=1)

    assert result.best_value == 1


def test_integer_parameter_search_refines_coarse_optimum(monkeypatch):
    costs = {value: float((value - 5) ** 2) for value in range(11)}
    clock = [0.0]
    monkeypatch.setattr(tuning.time, "perf_counter", lambda: clock[0])

    result = search_integer_parameter(
        0,
        10,
        lambda value: clock.__setitem__(0, clock[0] + costs[value]),
        coarse_step=5,
        refinement_radius=2,
        warmups=0,
        repetitions=1,
    )

    assert result.best_value == 5
    assert {timing.value for timing in result.timings} == {0, 3, 4, 5, 6, 7, 10}


def test_candidate_masks_are_valid_and_stable():
    masks = candidate_masks((8, 8, 8), 4)

    assert masks == (
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    )


def test_candidate_masks_can_fix_one_direction_and_auto_tune_the_rest():
    masks = candidate_masks((8, 8, 8), 4, mask_pattern=(True, "auto", "auto"))

    assert masks == (
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    )


def test_candidate_masks_can_fix_process_count_in_one_direction():
    masks = candidate_masks((8, 8, 8), 4, mask_pattern=(1, "auto", "auto"))

    assert masks == (
        (False, False, True),
        (False, True, False),
        (False, True, True),
    )


def test_candidate_masks_reject_invalid_pattern():
    with pytest.raises(ValueError, match="positive integer"):
        candidate_masks((8, 8, 8), 4, mask_pattern=(True, "sometimes", "auto"))

    with pytest.raises(ValueError, match="positive integer"):
        candidate_masks((8, 8, 8), 4, mask_pattern=(0, "auto", "auto"))


def test_optimizer_selects_fastest_candidate(monkeypatch):
    calls = []
    costs = {
        (False, False, True): 0.003,
        (True, True, True): 0.001,
    }
    clock = [0.0]

    def fake_perf_counter():
        return clock[0]

    def step(mask):
        calls.append(mask)
        clock[0] += costs[mask]

    monkeypatch.setattr(dd.time, "perf_counter", fake_perf_counter)

    result = optimize_domain_decomposition(
        (8, 8, 8),
        step,
        masks=costs,
        warmups=0,
        repetitions=1,
    )

    assert result.best_mask == (True, True, True)
    assert len(result.timings) == 2
    assert set(calls) == set(costs)

    baseline = next(t.seconds for t in result.timings if t.mask == (False, False, True))
    best = next(t.seconds for t in result.timings if t.mask == (True, True, True))
    assert baseline / best == pytest.approx(3.0)


def test_eight_rank_anisotropic_case_can_leave_one_direction_undecomposed(monkeypatch):
    """An anisotropic 3D case can prefer a 2D process grid."""
    costs = {mask: 0.01 for mask in candidate_masks((48, 24, 8), 8)}
    costs[(True, True, False)] = 0.001
    clock = [0.0]

    monkeypatch.setattr(dd.time, "perf_counter", lambda: clock[0])

    def step(mask):
        clock[0] += costs[mask]

    result = optimize_domain_decomposition(
        (48, 24, 8),
        step,
        comm=dd.MPI.COMM_WORLD,
        warmups=0,
        repetitions=1,
    )

    assert result.best_mask == (True, True, False)
    assert result.best_mask != (True, True, True)


def test_optimizer_rejects_invalid_mask():
    with pytest.raises(ValueError, match="not valid"):
        optimize_domain_decomposition(
            (8, 8, 8),
            lambda mask: None,
            masks=((False, False, False),),
            warmups=0,
            repetitions=1,
        )


def test_candidate_masks_reject_too_many_ranks():
    with pytest.raises(ValueError, match="cannot exceed"):
        candidate_masks((2, 2, 2), 9)


def test_parallel_optimizer_selects_clone_count_and_mask(monkeypatch):
    class Comm:
        def Get_size(self):
            return 8

        def Barrier(self):
            pass

        def allreduce(self, value, op=None):
            return value

    costs = {
        (1, (True, True, True)): 0.01,
        (2, (True, True, False)): 0.002,
        (4, (True, False, False)): 0.004,
    }
    clock = [0.0]
    monkeypatch.setattr(dd.time, "perf_counter", lambda: clock[0])

    def step(num_clones, mask):
        clock[0] += costs.get((num_clones, mask), 0.02)

    result = optimize_parallel_configuration(
        (8, 8, 8),
        step,
        comm=Comm(),
        clone_counts=(1, 2, 4),
        warmups=0,
        repetitions=1,
    )

    assert result.best_num_clones == 2
    assert result.best_mask == (True, True, False)


def test_candidate_masks_reject_empty_short_direction():
    masks = candidate_masks((16, 64, 1), 8)

    assert masks == (
        (False, True, False),
        (True, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    )


def test_candidate_masks_can_enforce_feec_minimum_local_size():
    masks = candidate_masks((128, 16, 1), 16, min_local_elements=(2, 2, 1))

    assert masks == (
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    )
