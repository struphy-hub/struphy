import logging
import os
import shutil

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

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
from struphy.models.tests.verification.spectrum_utils import fit_branches, power_spectrum

set_logging_level()
logger = logging.getLogger("struphy")


@pytest.mark.parametrize(
    "b_space, projected, algo",
    [
        ("Hdiv", False, "implicit"),
        ("Hdiv", False, "explicit"),
        ("Hcurl", False, "implicit"),
        ("Hcurl", True, "implicit"),
    ],
)
def test_slab_waves_1d(b_space: str, projected: bool, algo: str, do_plot: bool = False):
    # light-weight model instance
    model = LinearMHD(b_space=b_space, projected=projected)

    # environment options
    test_folder = os.path.join(os.getcwd(), "verification_tests")
    out_folders = os.path.join(test_folder, f"LinearMHD_{b_space}{'_projected' if projected else ''}")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="slab_waves_1d")

    # time stepping
    time_opts = Time(dt=0.15, Tend=90.0)

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

        omega, k, power = power_spectrum(run.fields.mhd.velocity, component=0)
        fits = fit_branches(omega, k, power, n_branches=1, noise_level=0.5)

        # assert
        vA = xp.sqrt(Bsquare / n0)
        v_alfven = vA * B0z / xp.sqrt(Bsquare)
        logger.info(f"{v_alfven =}, measured {fits[0][0]}")
        assert xp.abs(fits[0][0] - v_alfven) < 0.07

        # second fft
        omega, k, power = power_spectrum(run.fields.mhd.pressure)
        fits = fit_branches(omega, k, power, n_branches=2, noise_level=0.4)

        # assert
        gamma = 5 / 3
        cS = xp.sqrt(gamma * p0 / n0)

        delta = (4 * B0z**2 * cS**2 * vA**2) / ((cS**2 + vA**2) ** 2 * Bsquare)
        v_slow = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 - xp.sqrt(1 - delta)))
        v_fast = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 + xp.sqrt(1 - delta)))
        logger.info(f"{v_slow =}, measured {fits[0][0]}")
        logger.info(f"{v_fast =}, measured {fits[1][0]}")
        # the fits underestimate the speeds by 2-6% (numerical dispersion at the larger k of the fit; explicit
        # LinearMHD: slow 1.008, fast 2.81)
        assert xp.abs(fits[0][0] - v_slow) < 0.075
        assert xp.abs(fits[1][0] - v_fast) < 0.19

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_slab_waves_1d("Hdiv", False, algo="implicit", do_plot=True)
