"""Cylindrical space charge against the uniform round-beam envelope."""

import numpy as np

from axisymmetric_beam_expansion import measured_envelope, reference_envelope, run


def test_axisymmetric_space_charge_matches_round_beam_envelope(tmp_path):
    iteration = run(
        tmp_path,
        num_elements=(48, 12),
        n_rays=256,
        dt=0.1,
        max_rounds=10,
        planes=np.linspace(0.2, 0.8, 9),
    )
    z, measured = measured_envelope(iteration)
    expected = reference_envelope(z)
    np.testing.assert_allclose(measured, expected, rtol=0.01, atol=0.01)
    assert iteration.history[-1].lost_current["wall"] == 0.0
    assert iteration.history[-1].exit_current > 0.99
