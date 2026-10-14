"""Tests for the guiding-center post-processing of saved full orbits."""

import os

import numpy as np

from struphy import domains, equils
from struphy.post_processing.orbits import orbits_tools

N_MARKERS = 6


def test_guiding_center_and_classification(tmp_path):
    """Orbits as written by ``_post_process_markers`` (Particles6D quantities, marker ID last)."""
    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0)
    equil = equils.AdhocTorus(a=1.0, R0=3.0)
    equil.domain = domain

    rng = np.random.default_rng(0)
    etas = np.column_stack((rng.uniform(0.3, 0.9, N_MARKERS), rng.random(N_MARKERS), rng.random(N_MARKERS)))
    x = domain(etas, change_out_order=True)
    b = equil.b_cart(etas)[0].T
    absB = np.linalg.norm(b, axis=1)
    b /= absB[:, None]
    v = rng.normal(size=(N_MARKERS, 3))
    vpar = np.sum(v * b, axis=1)
    v_perp = v - vpar[:, None] * b

    # second time step: parallel velocity of the first marker reversed, last marker lost
    path_orbits = tmp_path / "orbits"
    path_orbits.mkdir()
    for n in range(2):
        v_n = v.copy()
        if n == 1:
            v_n[0] -= 2 * vpar[0] * b[0]
        orbits = np.column_stack((x, v_n, np.full(N_MARKERS, 0.5), np.arange(N_MARKERS)))
        if n == 1:
            orbits[-1, :-1] = 0.0
        np.save(path_orbits / f"ions_{n}.npy", orbits)

    orbits_tools.post_process_orbit_guiding_center(domain, equil, str(tmp_path), "ions")

    gc = np.load(tmp_path / "guiding_center" / "ions_0.npy")
    assert gc.shape == (N_MARKERS, 8)
    assert np.array_equal(gc[:, 0], np.arange(N_MARKERS))
    assert np.allclose(gc[:, 1:4], x - np.cross(b, v_perp) / absB[:, None])
    assert np.allclose(gc[:, 4], vpar)
    assert np.allclose(gc[:, 5], np.linalg.norm(v_perp, axis=1))
    assert np.allclose(gc[:, 6], np.sum(v_perp**2, axis=1) / (2 * absB))
    assert np.allclose(gc[:, 7], 0.5)
    assert os.path.exists(tmp_path / "guiding_center" / "ions_0.txt")

    orbits_tools.post_process_orbit_classification(str(tmp_path), "ions")

    classes = np.load(tmp_path / "guiding_center" / "ions_1.npy")[:, -1]
    assert np.array_equal(classes, [1, 0, 0, 0, 0, -1])
