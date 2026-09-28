"""Surface-emitted rays from the aperture lips are traced with the plasma rays and part of them is extracted."""

import numpy as np
from surface_emission import outlet_currents, run


def test_surface_rays_are_extracted_with_the_plasma_beam(tmp_path):
    iteration = run(tmp_path, n_volume=200, n_surface=100, max_rounds=4, n_tracked=0)
    record = iteration.history[-1]
    assert abs(sum(record.lost_current.values()) - 1.0) < 1e-9
    volume_out, surface_out = outlet_currents(iteration)
    assert surface_out > 0.0 and volume_out >= 0.0
    # surface ions on the plasma-facing slope cannot climb the sheath: part of the surface current returns to the wall
    exits = record.exit_records
    assert np.all(exits[:, 8] >= 0)
