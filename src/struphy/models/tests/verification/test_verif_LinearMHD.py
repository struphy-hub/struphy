import logging
import os
import shutil

import cunumpy as xp
import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from scipy.fft import fft2, fftfreq
from scipy.signal import argrelextrema

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.models import LinearMHD

set_logging_level()
logger = logging.getLogger("struphy")


def _power_spectrum(field, component=0):
    """Space-time power spectrum |F(omega, k)| of a field along z, at the first x and y grid point."""
    if "component" in field.dims:
        field = field.isel(component=component)
    data = field.isel(e1=0, e2=0).transpose("t", "e3")
    time, z = data.t.values, data.Z.values
    nt, nz = data.shape
    power = (2.0 / nt) * (2.0 / nz) * np.abs(fft2(data.values))[: nt // 2, : nz // 2]
    omega = 2 * np.pi * fftfreq(nt, time[1] - time[0])[: nt // 2]
    k = 2 * np.pi * fftfreq(nz, z[1] - z[0])[: nz // 2]
    return omega, k, power


def _fit_branches(omega, k, power, n_branches, noise_level, order=10):
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


@pytest.mark.parametrize("algo", ["implicit", "explicit"])
def test_slab_waves_1d(algo: str, do_plot: bool = False):
    # light-weight model instance
    model = LinearMHD()

    # environment options
    test_folder = os.path.join(os.getcwd(), "verification_tests")
    out_folders = os.path.join(test_folder, "LinearMHD")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="slab_waves_1d")

    # time stepping
    time_opts = Time(dt=0.15, Tend=180.0)

    # geometry
    domain = domains.Cuboid(r3=60.0)

    # fluid equilibrium (can be used as part of initial conditions)
    B0x = 0.0
    B0y = 1.0
    B0z = 1.0
    beta = 3.0
    n0 = 0.7
    equil = equils.HomogenSlab(B0x=B0x, B0y=B0y, B0z=B0z, beta=beta, n0=n0)

    # grid
    grid = grids.TensorProductGrid(num_elements=(1, 1, 64))

    # derham options
    derham_opts = DerhamOptions(degree=(1, 1, 3))

    # propagator options
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo=algo)

    # initial conditions (background + perturbation)
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=0, seed=123))
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=1, seed=123))
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=2, seed=123))

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
        equil=equil,
    )

    # run
    run = sim.run().with_time_units("normalized")

    # post processing
    run.pproc()

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        # first fft
        Bsquare = B0x**2 + B0y**2 + B0z**2
        p0 = beta * Bsquare / 2

        omega, k, power = _power_spectrum(run.fields.mhd.velocity, component=0)
        fits = _fit_branches(omega, k, power, n_branches=1, noise_level=0.5)

        # assert
        vA = xp.sqrt(Bsquare / n0)
        v_alfven = vA * B0z / xp.sqrt(Bsquare)
        logger.info(f"{v_alfven =}")
        assert xp.abs(fits[0][0] - v_alfven) < 0.07

        # second fft
        omega, k, power = _power_spectrum(run.fields.mhd.pressure)
        fits = _fit_branches(omega, k, power, n_branches=2, noise_level=0.4)

        # assert
        gamma = 5 / 3
        cS = xp.sqrt(gamma * p0 / n0)

        delta = (4 * B0z**2 * cS**2 * vA**2) / ((cS**2 + vA**2) ** 2 * Bsquare)
        v_slow = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 - xp.sqrt(1 - delta)))
        v_fast = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 + xp.sqrt(1 - delta)))
        logger.info(f"{v_slow =}")
        logger.info(f"{v_fast =}")
        assert xp.abs(fits[0][0] - v_slow) < 0.05
        assert xp.abs(fits[1][0] - v_fast) < 0.19

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_slab_waves_1d(algo="implicit", do_plot=True)
