"""Axisymmetric (r, z) wedge: Laplace solve and ray focusing of the two-tube lens (coarse)."""

import numpy as np
from axisymmetric_lens import incoming_axis_crossing, potential_error, reference, trace


def test_wedge_potential_converges(tmp_path):
    coarse, fine = potential_error(tmp_path, (60, 10)), potential_error(tmp_path, (120, 20))
    assert fine < 0.1 and coarse / fine > 3.0  # about second order (kinked linear-gap trace)


def test_wedge_rays_focus_like_exact_field(tmp_path):
    radii = np.array([1.0, 2.0, 3.0])
    z, r, _ = trace(tmp_path, radii, num_elements=(60, 10), dt=0.08)
    crossings = np.array([incoming_axis_crossing(z[:, j], r[:, j]) for j in range(len(radii))])
    z_ref, y_ref = reference(radii)
    from axisymmetric_lens import axis_crossing

    np.testing.assert_allclose(crossings, axis_crossing(z_ref, y_ref), atol=0.6)  # 60x10 elements
