import os

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mpi4py import MPI

from struphy import main
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.io.options import BaseUnits, DerhamOptions, EnvironmentOptions, FieldsBackground, Time
from struphy.kinetic_background import maxwellians
from struphy.models.toy import Poisson
from struphy.pic.utilities import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    WeightsParameters,
)
from struphy.topology import grids

test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")


def test_poisson_1d(do_plot=False):
    # environment options
    out_folders = os.path.join(test_folder, "Poisson")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="time_source_1d")

    # units
    base_units = BaseUnits()

    # time stepping
    time_opts = Time(dt=0.1, Tend=2.0)

    # geometry
    l1 = -5.0
    r1 = 5.0
    l2 = -5.0
    r2 = 5.0
    l3 = -6.0
    r3 = 6.0
    domain = domains.Cuboid(
        l1=l1,
        r1=r1,
    )  # l2=l2, r2=r2, l3=l3, r3=r3)

    # fluid equilibrium (can be used as part of initial conditions)
    equil = None

    # grid
    grid = grids.TensorProductGrid(Nel=(48, 1, 1))

    # derham options
    derham_opts = DerhamOptions()

    # light-weight model instance
    model = Poisson()

    # propagator options
    omega = 2 * np.pi
    model.propagators.source.options = model.propagators.source.Options(omega=omega)
    model.propagators.poisson.options = model.propagators.poisson.Options(rho=model.em_fields.source)

    # background, perturbations and initial conditions
    l = 2
    amp = 1e-1
    pert = perturbations.ModesCos(ls=(l,), amps=(amp,))
    model.em_fields.source.add_perturbation(pert)

    # analytical solution
    Lx = r1 - l1
    rhs_exact = lambda e1, e2, e3, t: amp * np.cos(l * 2 * np.pi / Lx * e1) * np.cos(omega * t)
    phi_exact = (
        lambda e1, e2, e3, t: amp / (l * 2 * np.pi / Lx) ** 2 * np.cos(l * 2 * np.pi / Lx * e1) * np.cos(omega * t)
    )

    # start run
    verbose = True

    main.run(
        model,
        params_path=None,
        env=env,
        base_units=base_units,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        verbose=verbose,
    )

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        main.pproc(env.path_out)

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        simdata = main.load_data(env.path_out)

        phi = simdata.spline_values["em_fields"]["phi_log"]
        source = simdata.spline_values["em_fields"]["source_log"]
        x = simdata.grids_phy[0][:, 0, 0]
        y = simdata.grids_phy[1][0, :, 0]
        z = simdata.grids_phy[2][0, 0, :]
        time = simdata.t_grid

        interval = 2
        c = 0
        if do_plot:
            fig = plt.figure(figsize=(12, 40))

        err = 0.0
        for i, t in enumerate(phi):
            phi_h = phi[t][0][:, 0, 0]
            phi_e = phi_exact(x, 0, 0, t)
            new_err = np.abs(np.max(phi_h - phi_e)) / (amp / (l * 2 * np.pi / Lx) ** 2)
            if new_err > err:
                err = new_err

            if do_plot and i % interval == 0:
                plt.subplot(5, 2, 2 * c + 1)
                plt.plot(x, phi_h, label="phi")
                plt.plot(x, phi_e, "r--", label="exact")
                plt.title(f"phi at {t = }")
                plt.ylim(-amp / (l * 2 * np.pi / Lx) ** 2, amp / (l * 2 * np.pi / Lx) ** 2)
                plt.legend()

                plt.subplot(5, 2, 2 * c + 2)
                plt.plot(x, source[t][0][:, 0, 0], label="rhs")
                plt.plot(x, rhs_exact(x, 0, 0, t), "r--", label="exact")
                plt.title(f"source at {t = }")
                plt.ylim(-amp, amp)
                plt.legend()

                c += 1
                if c > 4:
                    break

        plt.show()
        print(f"{err = }")
        assert err < 0.0057


if __name__ == "__main__":
    # test_light_wave_1d(algo="explicit", do_plot=True)
    test_poisson_1d(do_plot=False)
