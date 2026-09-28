"""Slit plasma extraction (coarse, fast): a plasma forms behind the aperture and a beam is extracted."""

import numpy as np
from plasma_extraction import DEFAULT, run


def test_extraction_forms_plasma_and_extracts_beam(tmp_path):
    iteration = run(tmp_path, num_elements=(48, 18), n_rays=200, dt=0.1, max_rounds=4, n_tracked=0)
    model = iteration.model
    record = iteration.history[-1]
    # every ray ends somewhere, and part of the beam passes both apertures
    assert abs(sum(record.lost_current.values()) - 1.0) < 1e-9
    # in some round, a beam is extracted through the slit (single rounds of a coarse unconverged iteration fluctuate)
    assert max(r.exit_current + r.lost_current["puller"] for r in iteration.history) > 0.02
    # quasi-neutral plasma deep in the chamber, no electrons beyond the plasma electrode
    y_mid = np.array([0.5])
    phi = np.asarray(model.em_fields.phi.spline(np.array([0.02, 0.6]), y_mid, np.array([0.5]))).ravel()
    n_e = -model.plasma.charge_density(phi) / model.plasma.density
    assert n_e[0] > 0.3 and n_e[1] < 1e-3
    assert DEFAULT.extraction_voltage > 0


def test_the_domain_walls_are_the_electrode_profile_exactly():
    """A cubic mapping rounds the lips (the plasma slit came out 15 % narrow on 48 elements); degree 1
    on elements whose boundaries include every wall corner reproduces the profile."""
    from plasma_extraction import aligned, build_domain

    domain = build_domain(DEFAULT, (48, 18))
    assert domain.num_elements[0] == aligned(DEFAULT, (48, 18))[0] == 64
    assert domain.wall_error < 1e-12
    x, y = DEFAULT.profile()
    eta1 = np.linspace(0, 1, 2001)
    xs, top, _ = (np.ravel(v) for v in domain(eta1, np.array([1.0]), np.array([0.5])))
    lip = (xs >= DEFAULT.plasma_lip[0]) & (xs <= DEFAULT.plasma_lip[1])
    assert np.allclose(top[lip], DEFAULT.plasma_aperture) and np.isclose(top.max(), DEFAULT.chamber_half_height)
