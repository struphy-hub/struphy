"""End-to-end checks of projection, force normalization and particle transport."""

import h5py
import numpy as np
import pytest

from planar_accelerator import build_simulation


@pytest.mark.parametrize("epsilon", [0.5, 2.0])
def test_constant_acceleration(tmp_path, epsilon):
    sim = build_simulation(tmp_path, dt=0.01, end_time=0.1, epsilon=epsilon)
    sim.run()
    with h5py.File(tmp_path / "planar_accelerator/data/data_proc0.hdf5") as f:
        t = f["time/value"][:]
        markers = f["kinetic/ions/markers"][:]
    initial = markers[0]
    expected = np.broadcast_to(initial[None, :, :6], markers[:, :, :6].shape).copy()
    expected[:, :, :3] += t[:, None, None] * initial[None, :, 3:6]
    expected[:, :, 0] += 0.5 * (0.4 / epsilon) * t[:, None] ** 2
    expected[:, :, 3] += (0.4 / epsilon) * t[:, None]
    np.testing.assert_allclose(markers[:, :, :6], expected, atol=2e-11, rtol=0)
