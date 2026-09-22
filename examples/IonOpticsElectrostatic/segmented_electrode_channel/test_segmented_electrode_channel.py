"""End-to-end checks of segmented voltages, curved geometry and trajectories."""

import numpy as np

from double_aperture_accelerator import UNITS
from segmented_verification import physical_paths, scipy_reference, trace, transverse_at_planes


def test_segmented_geometry_field_and_rays_end_to_end(tmp_path):
    iteration = trace(tmp_path, num_elements=(48, 12), dt=0.05)
    domain, model = iteration.sim.domain, iteration.model

    # The mapped walls contain both 5 mm chambers and 2.25 mm apertures.
    probe = np.array([[10 / 80, 0.0, 0.5], [22.5 / 80, 0.0, 0.5], [22.5 / 80, 1.0, 0.5]])
    xyz = np.asarray(domain(probe, change_out_order=True, remove_outside=False))
    # The profile kink is represented by the spline mapping; this coarse CI
    # mesh is within 0.08 mm of the requested lip position.
    np.testing.assert_allclose(xyz[:, 1], [-5.0, -2.25, 2.25], atol=8e-2)

    # Segment interiors have their prescribed Dirichlet values on both walls.
    checks = ((10.0, 0.0), (22.5, -5.0), (40.0, -5.0), (60.0, -10.0), (76.0, -10.0))
    for x, kilovolts in checks:
        for side in (0.0, 1.0):
            value = float(model.em_fields.phi.spline(np.array([[x / 80, side, 0.5]]))[0])
            # At the one-element-wide aperture plateau the trace projection is
            # within 3.2 V; long segment interiors are exact to roundoff.
            assert abs(value - kilovolts) < 3.2e-3

    record = iteration.history[-1]
    assert record.exit_current == 1.0
    assert record.lost_current["wall"] == 0.0
    # The central proton gains 10 keV between the 0 and -10 kV regions.
    central = record.exit_records[np.argmin(np.abs(record.exit_records[:, 1]))]
    assert abs(float(UNITS.kinetic_energy_eV(np.linalg.norm(central[3:6]))) - 12e3) < 80.0

    # Independent high-order integration of the FE field checks the mapped pusher.
    paths = physical_paths(iteration)
    planes = np.linspace(5.0, 75.0, 15)
    for ray in range(len(iteration.rays)):
        reference, _ = scipy_reference(iteration, ray)
        actual_y = transverse_at_planes(paths[:, ray], planes)
        reference_y = transverse_at_planes(reference, planes)
        np.testing.assert_allclose(actual_y, reference_y, atol=1e-3)
