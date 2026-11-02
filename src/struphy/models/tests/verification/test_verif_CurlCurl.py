import logging
import os
import shutil

import cunumpy as xp
import pytest
from matplotlib import pyplot as plt
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
    perturbations,
)
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import CurlCurl

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("hfun", ["cos", "sin"])
def test_curl_curl_1d(hfun, do_plot=False):
    # light-weight model instance
    model = CurlCurl(with_t_dep_source=True)

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "CurlCurl")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder=f"time_source_1d_{hfun}")

    # time stepping
    time_opts = Time(dt=0.1, Tend=1.0)

    # geometry: unit cube, the 1-form components are the Cartesian ones
    domain = domains.Cuboid()

    # grid
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    # propagator options
    omega = 2 * xp.pi
    sigma = 2.0
    model.propagators.source.options = model.propagators.source.Options(omega=omega, hfun=hfun)
    model.propagators.curl_curl.options = model.propagators.curl_curl.Options(
        sigma=sigma,
        precond="StiffnessPreconditioner",
        solver_params=SolverParameters(tol=1e-12, maxiter=1000),
    )

    # source J = (0, 0, amp * cos(k x)), solution E = J / (k^2 + sigma)
    l = 2
    amp = 1e-1
    k = l * 2 * xp.pi
    model.em_fields.source.add_perturbation(
        perturbations.ModesCos(ls=(l,), amps=(amp,), given_in_basis="1", comp=2),
    )
    h = xp.cos if hfun == "cos" else xp.sin
    e_exact = lambda x, t: amp / (k**2 + sigma) * xp.cos(k * x) * h(omega * t)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    # run
    run = sim.run().with_time_units("normalized")

    # post processing
    run.pproc(physical=True)

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        ez = run.fields.em_fields.e_field_xyz.isel(component=2, eta2=0, eta3=0)
        x = ez.X.values

        err = 0.0
        for i, t in enumerate(ez.t.values):
            ez_h = ez.isel(t=i).values
            ez_e = e_exact(x, t)
            err = max(err, xp.max(xp.abs(ez_h - ez_e)) / (amp / (k**2 + sigma)))

            if do_plot and i % 2 == 0:
                plt.plot(x, ez_h, label=f"t={t:.1f}")
                plt.plot(x, ez_e, "k--")

        if do_plot:
            plt.legend()
            plt.show()

        logger.info(f"{err =}")
        assert err < 1e-4

        shutil.rmtree(os.path.join(out_folders, env.sim_folder))


if __name__ == "__main__":
    test_curl_curl_1d("sin", do_plot=True)
