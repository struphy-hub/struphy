import os
import shutil

import cunumpy as xp
import pytest
from feectools.ddm.mpi import mpi as MPI

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
)
from struphy.diagnostics.diagn_tools import power_spectrum_2d
from struphy.models import LinearMHD


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
    grid = grids.TensorProductGrid(Nel=(1, 1, 64))

    # derham options
    derham_opts = DerhamOptions(p=(1, 1, 3))

    # species parameters
    model.mhd.set_species_properties()

    # propagator options
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo=algo)
    model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options(b_field=model.em_fields.b_field)

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
        verbose=True,
    )

    # run
    sim.run(verbose=True)

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc(verbose=True)

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.load_plotting_data(verbose=True)

        # first fft
        u_of_t = sim.spline_values.mhd.velocity_log.data

        Bsquare = B0x**2 + B0y**2 + B0z**2
        p0 = beta * Bsquare / 2

        disp_params = {"B0x": B0x, "B0y": B0y, "B0z": B0z, "p0": p0, "n0": n0, "gamma": 5 / 3}

        _1, _2, _3, coeffs = power_spectrum_2d(
            u_of_t,
            "velocity_log",
            grids=sim.grids_log,
            grids_mapped=sim.grids_phy,
            component=0,
            slice_at=[0, 0, None],
            do_plot=do_plot,
            disp_name="MHDhomogenSlab",
            disp_params=disp_params,
            fit_branches=1,
            noise_level=0.5,
            extr_order=10,
            fit_degree=(1,),
        )

        # assert
        vA = xp.sqrt(Bsquare / n0)
        v_alfven = vA * B0z / xp.sqrt(Bsquare)
        print(f"{v_alfven =}")
        assert xp.abs(coeffs[0][0] - v_alfven) < 0.07

        # second fft
        p_of_t = sim.spline_values.mhd.pressure_log.data

        _1, _2, _3, coeffs = power_spectrum_2d(
            p_of_t,
            "pressure_log",
            grids=sim.grids_log,
            grids_mapped=sim.grids_phy,
            component=0,
            slice_at=[0, 0, None],
            do_plot=do_plot,
            disp_name="MHDhomogenSlab",
            disp_params=disp_params,
            fit_branches=2,
            noise_level=0.4,
            extr_order=10,
            fit_degree=(1, 1),
        )

        # assert
        gamma = 5 / 3
        cS = xp.sqrt(gamma * p0 / n0)

        delta = (4 * B0z**2 * cS**2 * vA**2) / ((cS**2 + vA**2) ** 2 * Bsquare)
        v_slow = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 - xp.sqrt(1 - delta)))
        v_fast = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 + xp.sqrt(1 - delta)))
        print(f"{v_slow =}")
        print(f"{v_fast =}")
        assert xp.abs(coeffs[0][0] - v_slow) < 0.05
        assert xp.abs(coeffs[1][0] - v_fast) < 0.19

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_slab_waves_1d(algo="implicit", do_plot=True)
