"""Tests for the dispersion analysis on labeled field data."""

import numpy as np
import pytest

from struphy.diagnostics.diagn_tools import power_spectrum_2d
from struphy.post_processing.arrays import data_array

LENGTH, SPEED = 20.0, 1.0


def standing_waves(dt=0.05, tend=2 * LENGTH, nx=128):
    """Standing waves of all resolved wavenumbers with phase speed SPEED along eta3, as a field of a Output.

    The time window holds whole periods of every wave, so the spectrum has no leakage.
    """
    t = np.arange(0.0, tend, dt)
    eta = np.linspace(0.0, 1.0, nx, endpoint=False)
    z = eta * LENGTH
    rng = np.random.default_rng(0)
    values = np.zeros((t.size, eta.size))
    for n in range(1, nx // 2):
        k = 2 * np.pi * n / LENGTH
        values += np.cos(k * z[None, :] + rng.uniform(0, 2 * np.pi)) * np.cos(k * SPEED * t[:, None])
    field = np.zeros((t.size, 2, 2, 1, nx))
    field[:, 1, :, 0, :] = values[:, None, :]
    mesh = np.meshgrid(np.zeros(2), np.zeros(1), z, indexing="ij")
    coords = {"t": t, "component": [0, 1], "e1": [0.0, 0.5], "e2": [0.0], "e3": eta}
    coords.update({name: (("e1", "e2", "e3"), grid) for name, grid in zip(("X", "Y", "Z"), mesh)})
    return data_array(field, ("t", "component", "e1", "e2", "e3"), coords, name="e_field_log")


@pytest.mark.parametrize("physical", [True, False])
def test_fitted_phase_speed(physical):
    field = standing_waves()
    omega, kvec, dispersion, coeffs = power_spectrum_2d(
        field, component=1, slice_at=(0, 0, None), physical=physical, fit_branches=1, noise_level=0.5
    )
    assert dispersion.shape == (omega.size, kvec.size)
    # on the logical grid, wavenumbers are scaled by the domain length
    expected = SPEED if physical else SPEED / LENGTH
    assert coeffs[0][0] == pytest.approx(expected, rel=0.02)


def test_needs_exactly_one_fft_direction():
    with pytest.raises(AssertionError, match="slice_at"):
        power_spectrum_2d(standing_waves(tend=1.0), slice_at=(None, None, 0))
