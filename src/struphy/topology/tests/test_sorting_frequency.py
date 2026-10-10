"""Example of selecting a particle sorting frequency empirically."""

import struphy.topology.autotuning as tuning
from struphy.topology import optimize_sorting_frequency


def test_sorting_frequency_tuner_finds_fastest_candidate(monkeypatch):
    """A fixed workload with a known sorting tradeoff selects frequency five."""
    # The callback stands in for one fixed simulation segment. In a real
    # benchmark it constructs a fresh Simulation with
    # SortingParameters(sorting_frequency=frequency) and runs that segment.
    measured_costs = {0: 0.030, 1: 0.080, 2: 0.045, 5: 0.020, 10: 0.025}
    clock = [0.0]
    monkeypatch.setattr(tuning.time, "perf_counter", lambda: clock[0])

    def run_segment(frequency):
        clock[0] += measured_costs[frequency]

    result = optimize_sorting_frequency(
        (0, 1, 2, 5, 10),
        run_segment,
        warmups=0,
        repetitions=1,
    )

    assert result.best_value == 5
    assert min(result.timings, key=lambda timing: timing.seconds).value == 5
