"""Space charge in 2D: expansion of a uniform sheet beam (coarse, fast version)."""

import numpy as np
from sheet_beam_expansion import build_simulation, fit_growth, live_beam, normalized


def test_sheet_beam_expands_as_analytic_slab(tmp_path):
    sim = build_simulation(tmp_path, num_elements=(60, 8), dt=0.1, end_time=26.0, rate=60.0)
    sim.run()
    p = normalized()
    x, y = live_beam(sim)
    coefficient, sigma = fit_growth(x, y, p)
    analytic = p["current"] / (4.0 * p["a0"] * p["v0"] ** 3)
    assert abs(coefficient / analytic - 1.0) < max(3 * sigma / analytic, 0.2)
    assert sim.model.ledger.lost_markers["plates"] == 0
