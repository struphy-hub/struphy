"""Poisson–Boltzmann extraction building block: the planar Bohm sheath (coarse, fast version)."""

import numpy as np
from bohm_sheath import reference, sheath


def test_bohm_sheath_matches_steady_state_ode(tmp_path):
    iteration, x, phi = sheath(tmp_path, v0=1.3, num_elements=32, n_rays=30, dt=0.06, tol=3e-4)
    assert iteration.converged
    assert np.max(np.abs(phi - reference(1.3, x))) < 1e-2  # 32 elements, 30 rays; 2e-4 at the example settings
    assert iteration.history[-1].lost_current["plasma"] == 0.0  # no ion is reflected
