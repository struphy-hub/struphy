import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("gvec") is None, reason="gvec is not installed")

GVEC_PARAMS = {"dat_file": "run_01/CIRCTOK_State_0000_00000000.dat", "param_file": "run_01/parameter.ini"}


def _markers(n=7, seed=3):
    """``n`` markers at random logical positions, away from the hole at the axis."""
    rng = np.random.default_rng(seed)
    markers = rng.uniform(0.0, 1.0, (n, 3))
    markers[:, 0] = rng.uniform(0.1, 0.9, n)
    return markers


@pytest.mark.parametrize("name", ["bv", "jv", "p0", "n0", "absB0"])
def test_gvec_marker_evaluation_matches_grid(name):
    """Marker (flat) evaluation gives one value per marker, equal to the grid evaluation at each marker.

    Before, gvec evaluated the markers' rho, theta and zeta as a tensor grid, so ``bv`` and ``jv`` at N markers
    returned N x N x N arrays, and ``absB0`` failed.
    """
    from struphy.fields_background.equils import GVECequilibrium

    equil = GVECequilibrium(**GVEC_PARAMS)
    markers = _markers()
    flat = getattr(equil, name)(markers)
    flat = flat if isinstance(flat, tuple) else (flat,)

    for ip, (e1, e2, e3) in enumerate(markers):
        grid = getattr(equil, name)(np.array([e1]), np.array([e2]), np.array([e3]))
        grid = grid if isinstance(grid, tuple) else (grid,)
        for f, g in zip(flat, grid):
            assert f.shape == (len(markers),)
            assert np.isclose(f[ip], g.ravel()[0], rtol=1e-12, atol=1e-14)


def test_gvec_marker_evaluation_boozer_raises():
    """gvec computes the Boozer transform per flux surface, so marker evaluation with ``use_boozer=True`` raises."""
    from struphy.fields_background.equils import GVECequilibrium

    equil = GVECequilibrium(**GVEC_PARAMS, use_boozer=True)
    with pytest.raises(NotImplementedError, match="use_boozer"):
        equil.bv(_markers())
