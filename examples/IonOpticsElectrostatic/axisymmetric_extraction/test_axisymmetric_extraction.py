"""Axisymmetric extraction (coarse, fast): round apertures, plasma behind the aperture, beam extracted."""

import os

import numpy as np
import pytest
from axisymmetric_extraction import DEFAULT, run


def test_axisymmetric_extraction_forms_plasma_and_extracts_beam(tmp_path):
    iteration = run(tmp_path, num_elements=(48, 18), n_rays=256, dt=0.1, max_rounds=4, n_tracked=0)
    model = iteration.model
    record = iteration.history[-1]
    assert abs(sum(record.lost_current.values()) - 1.0) < 1e-9
    # rays reflect at the axis and wedge faces: nothing leaves there
    assert record.lost_current["other"] == 0.0
    # in some round, current gets past the plasma electrode through the round aperture (a single round of
    # this coarse, unconverged iteration can extract nothing, so look at all rounds)
    assert max(r.exit_current + r.lost_current["puller"] for r in iteration.history) > 0.004
    phi = np.asarray(model.em_fields.phi.spline(np.array([0.02, 0.6]), np.array([0.3]), np.array([0.5]))).ravel()
    n_e = -model.plasma.charge_density(phi) / model.plasma.density
    assert n_e[0] > 0.3 and n_e[1] < 1e-3
    assert DEFAULT.extraction_voltage > 0


@pytest.mark.skipif(
    os.environ.get("STRUPHY_RUN_DENSE_ION_OPTICS") != "1",
    reason="set STRUPHY_RUN_DENSE_ION_OPTICS=1 for the two-minute 1024-ray extraction check",
)
def test_axisymmetric_extraction_dense_ray_noise_floor(tmp_path):
    """Production ray count: stable observables, but the charge residual remains noise limited."""
    iteration = run(
        tmp_path,
        num_elements=(48, 18),
        n_rays=1024,
        dt=0.1,
        max_rounds=70,
        n_tracked=0,
        tol=1e-3,
        verbose=False,
    )
    average = iteration.averaged(10)
    exit_mean, exit_std = average["exit_current"]
    assert 0.08 < exit_mean < 0.12
    assert exit_std < 0.01
    assert iteration.history[-1].potential_change < 1e-4
    assert abs(sum(iteration.history[-1].lost_current.values()) - 1.0) < 1e-12
    # 1024 rays stabilize integrated current, but edge-ray fate changes keep
    # the alpha-independent charge residual above its convergence tolerance.
    assert not iteration.converged
    assert iteration.history[-1].residual > iteration.tol
