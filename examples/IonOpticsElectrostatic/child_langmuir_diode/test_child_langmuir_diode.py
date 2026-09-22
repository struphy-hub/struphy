"""Space charge: planar diode potential below, and current limitation above, the Child–Langmuir limit."""

import numpy as np
from child_langmuir_diode import J_CL, build_simulation, reference_potential, steady_state


def _run(tmp_path, fraction):
    sim = build_simulation(tmp_path, fraction, num_elements=32, end_time=6.0, markers_per_time=1000.0, dt=0.02)
    sim.run()
    return steady_state(sim, t_start=4.0)


def test_diode_below_child_langmuir_matches_steady_state_ode(tmp_path):
    x, phi, fractions, currents, *_ = _run(tmp_path, 0.5)
    assert np.max(np.abs(phi - reference_potential(0.5, x))) < 1e-3
    assert abs(currents["collector"] / J_CL - 0.5) < 0.01
    assert fractions["emitter"] == 0.0


def test_diode_above_child_langmuir_limits_transmitted_current(tmp_path):
    _, _, fractions, currents, *_ = _run(tmp_path, 2.0)
    assert abs(currents["collector"] / J_CL - 1.0) < 0.1
    assert fractions["emitter"] > 0.3
