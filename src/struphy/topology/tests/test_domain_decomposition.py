import pytest

import struphy.topology.domain_decomposition as dd
from struphy.topology.domain_decomposition import candidate_masks, optimize_domain_decomposition


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
    costs = {
        mask: 0.01
        for mask in candidate_masks((48, 24, 8), 8)
    }
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
