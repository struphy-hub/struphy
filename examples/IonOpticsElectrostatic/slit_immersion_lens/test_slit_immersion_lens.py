"""Zero-current lens baseline: Laplace solve, pusher and SI units against the exact-field reference."""

import numpy as np

from analytic import SlitImmersionLens
from slit_immersion_lens import (
    DESIGN,
    UNITS,
    analytic_lens,
    axis_crossings,
    build_simulation,
    compare_with_reference,
    load_trajectories,
)


def test_analytic_lens_matches_plate_potential_and_is_harmonic():
    lens = SlitImmersionLens(h=1.0, g=0.4, xc=0.0, v1=0.0, v2=1.0)
    x = np.linspace(-3.0, 3.0, 13)
    np.testing.assert_allclose(lens.phi(x, 1.0), lens.plate_potential(x), atol=1e-12)
    np.testing.assert_allclose(lens.phi(x, -1.0), lens.plate_potential(x), atol=1e-12)

    x0, y0, d = 0.13, 0.37, 1e-3
    laplacian = (
        lens.phi(x0 + d, y0) + lens.phi(x0 - d, y0) + lens.phi(x0, y0 + d) + lens.phi(x0, y0 - d) - 4 * lens.phi(x0, y0)
    ) / d**2
    assert abs(laplacian) < 1e-5
    ex, ey = lens.efield(x0, y0)
    assert abs(ex + (lens.phi(x0 + d, y0) - lens.phi(x0 - d, y0)) / (2 * d)) < 1e-6
    assert abs(ey + (lens.phi(x0, y0 + d) - lens.phi(x0, y0 - d)) / (2 * d)) < 1e-6


def test_zero_current_beam_follows_exact_field_rays(tmp_path):
    sim = build_simulation(tmp_path, num_elements=(160, 20), dt=0.04)
    sim.run()
    time, states = load_trajectories(sim)
    error, exact = compare_with_reference(time, states)

    # All rays leave through the downstream end, none hit an electrode.
    assert np.all(np.isnan(states[-1, :, 0]))
    assert np.all(np.nanmax(states[..., 0], axis=0) > DESIGN.length / UNITS.length - 0.5)
    # Coarse mesh: tens of µm, converging with refinement (see README).
    assert error < 1e-2
    np.testing.assert_allclose(axis_crossings(states), axis_crossings(exact), atol=0.05, equal_nan=True)

    # Energy: v²/2 + phi is conserved; the beam gains Z e |V2 - V1|.
    lens = analytic_lens()
    speed = np.hypot(states[..., 2], states[..., 3])
    total = 0.5 * speed**2 + lens.phi(states[..., 0], states[..., 1])
    assert np.nanmax(np.abs(total - total[0])) < 1e-3 * abs(lens.v2 - lens.v1)
