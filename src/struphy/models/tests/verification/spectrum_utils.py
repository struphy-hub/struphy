"""Space-time spectra of 1D wave runs and fits of their dispersion branches, shared by the verification tests."""

import numpy as np
from scipy.fft import fft2, fftfreq
from scipy.signal import argrelextrema


def power_spectrum(field, component=0):
    """Space-time power spectrum |F(omega, k)| of a field along z, at the first x and y grid point."""
    if "component" in field.dims:
        field = field.isel(component=component)
    data = field.isel(eta1=0, eta2=0).transpose("t", "eta3")
    time, z = data.t.values, data.Z.values
    nt, nz = data.shape
    power = (2.0 / nt) * (2.0 / nz) * np.abs(fft2(data.values))[: nt // 2, : nz // 2]
    omega = 2 * np.pi * fftfreq(nt, time[1] - time[0])[: nt // 2]
    k = 2 * np.pi * fftfreq(nz, z[1] - z[0])[: nz // 2]
    return omega, k, power


def fit_branches(omega, k, power, n_branches, noise_level, order=10):
    """Fit omega = v * k + b to each of the n_branches spectral peaks; returns [(v, b), ...]."""
    k_fit, omega_fit = [], [[] for _ in range(n_branches)]
    for i in range(k.size // 8, k.size // 2):
        column = power[:, i]
        maxima = argrelextrema(column, np.greater, order=order)[0]
        peaks = sorted(j for j in maxima if column[j] > noise_level * column.max())
        if not peaks:
            continue
        assert len(peaks) == n_branches, (
            f"Found {len(peaks)} branches at k={k[i]:.3f}, expected {n_branches}. Try another noise_level or order."
        )
        k_fit.append(k[i])
        for branch, j in zip(omega_fit, peaks):
            branch.append(omega[j])
    return [np.polyfit(k_fit, branch, deg=1) for branch in omega_fit]
