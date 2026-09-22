"""Steady-state Vlasov–Poisson iteration: converges below, oscillates above the Child–Langmuir limit."""

import numpy as np
from steady_state_iteration import diode, diode_iteration

from struphy.pic.ray_tracing import StepControl


def test_iteration_converges_to_steady_diode_below_limit(tmp_path):
    iteration, x, phi = diode_iteration(tmp_path, 0.5, n_rays=200, dt=0.01)
    assert iteration.converged and len(iteration.history) < 20
    assert np.max(np.abs(phi - diode.reference_potential(0.5, x))) < 1e-4
    assert abs(iteration.history[-1].exit_current - 1.0) < 1e-12


def test_iteration_does_not_converge_above_limit(tmp_path):
    iteration, *_ = diode_iteration(tmp_path, 2.0, n_rays=200, dt=0.01, max_rounds=12)
    assert not iteration.converged
    passing = [r.exit_current for r in iteration.history]
    # alternates between a transmitted and a reflected beam
    assert max(passing) > 0.99 and min(passing) < 0.01


def test_adaptive_steps_match_the_fine_fixed_step_in_the_diode(tmp_path):
    """Per-ray steps reach the accuracy of the smallest fixed step with about half the steps, against the analytic ODE."""
    fine, x, phi_fine = diode_iteration(tmp_path, 0.5, n_rays=200, dt=0.005)
    coarse, _, phi_coarse = diode_iteration(tmp_path, 0.5, n_rays=200, dt=0.02)
    adaptive, _, phi = diode_iteration(tmp_path, 0.5, n_rays=200, dt=0.02, step_control=StepControl())
    reference = diode.reference_potential(0.5, x)
    error = lambda p: np.max(np.abs(p - reference))
    assert error(phi) < 1e-5 and error(phi) < error(phi_coarse)  # coarse fixed step: 2e-5
    assert error(phi) < 1.5 * error(phi_fine)
    steps = lambda it: np.mean([r.steps for r in it.history])
    assert steps(adaptive) < 0.6 * steps(fine)
