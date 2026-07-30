import logging
import os
import shutil

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
    perturbations,
)
from struphy.models import Poisson

logger = logging.getLogger("struphy")


def test_poisson_1d(do_plot=False):
    # light-weight model instance
    model = Poisson(with_t_dep_source=True)

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "Poisson")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="time_source_1d")

    # time stepping
    time_opts = Time(dt=0.1, Tend=2.0)

    # geometry
    l1 = -5.0
    r1 = 5.0
    domain = domains.Cuboid(
        l1=l1,
        r1=r1,
    )

    # fluid equilibrium (can be used as part of initial conditions)
    equil = None

    # grid
    grid = grids.TensorProductGrid(num_elements=(48, 1, 1))

    # propagator options
    omega = 2 * xp.pi
    if model.with_t_dep_source:
        model.propagators.source.options = model.propagators.source.Options(omega=omega)

    # background, perturbations and initial conditions
    l = 2
    amp = 1e-1
    pert = perturbations.ModesCos(ls=(l,), amps=(amp,))
    model.em_fields.source.add_perturbation(pert)

    # analytical solution
    Lx = r1 - l1
    rhs_exact = lambda e1, e2, e3, t: amp * xp.cos(l * 2 * xp.pi / Lx * e1) * xp.cos(omega * t)
    phi_exact = lambda e1, e2, e3, t: (
        amp / (l * 2 * xp.pi / Lx) ** 2 * xp.cos(l * 2 * xp.pi / Lx * e1) * xp.cos(omega * t)
    )

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
    )

    # run
    sim.run()

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.load_plotting_data()

        phi = sim.spline_values.em_fields.phi_log.data
        source = sim.spline_values.em_fields.source_log.data
        x = sim.grids_phy[0][:, 0, 0]
        y = sim.grids_phy[1][0, :, 0]
        z = sim.grids_phy[2][0, 0, :]
        time = sim.t_grid

        interval = 2
        c = 0
        if do_plot:
            fig = plt.figure(figsize=(12, 40))

        err = 0.0
        for i, t in enumerate(phi):
            phi_h = phi[t][0][:, 0, 0]
            phi_e = phi_exact(x, 0, 0, t)
            new_err = xp.abs(xp.max(phi_h - phi_e)) / (amp / (l * 2 * xp.pi / Lx) ** 2)
            err = max(err, new_err)

            if do_plot and i % interval == 0:
                plt.subplot(5, 2, 2 * c + 1)
                plt.plot(x, phi_h, label="phi")
                plt.plot(x, phi_e, "r--", label="exact")
                plt.title(f"phi at {t =}")
                plt.ylim(-amp / (l * 2 * xp.pi / Lx) ** 2, amp / (l * 2 * xp.pi / Lx) ** 2)
                plt.legend()

                plt.subplot(5, 2, 2 * c + 2)
                plt.plot(x, source[t][0][:, 0, 0], label="rhs")
                plt.plot(x, rhs_exact(x, 0, 0, t), "r--", label="exact")
                plt.title(f"source at {t =}")
                plt.ylim(-amp, amp)
                plt.legend()

                c += 1
                if c > 4:
                    break

        plt.show()
        logger.info(f"{err =}")
        assert err < 0.0057

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_poisson_1d(do_plot=False)
