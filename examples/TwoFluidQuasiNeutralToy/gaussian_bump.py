from cunumpy import zeros_like
from struphy.io.options import EnvironmentOptions, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy import Simulation
from struphy.linear_algebra.solver import SolverParameters

import os
import glob
import cunumpy as xp
import matplotlib.pyplot as plt
from mpi4py import MPI

from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy

name = "runs/sim_2D_gaussian"
env = EnvironmentOptions(sim_folder=name)

B0 = 1
nu = 1.0
nu_e = 1.0
Nel = (8, 8, 1)
p = (2, 2, 1)
epsilon = 1.0
dt = 10
Tend = 10
sigma = 0
tol = 1e-6

time_opts = Time(dt=dt, Tend=Tend)
domain = domains.Cuboid()
equil = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
grid = grids.TensorProductGrid(num_elements=Nel)
derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

def source_function_u(x, y, z):
    bump = xp.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.1**2))
    return bump, zeros_like(x), zeros_like(x)

def source_function_ue(x, y, z):
    return zeros_like(x), zeros_like(x), zeros_like(x)

model = TwoFluidQuasiNeutralToy()

model.propagators.qn_full.options = model.propagators.qn_full.Options(
    nu=nu,
    nu_e=nu_e,
    eps_norm=epsilon,
    stab_sigma=sigma,
    source_u=source_function_u,
    source_ue=source_function_ue,
    solver="gmres",
    solver_params=SolverParameters(info=True, tol=tol),
)

sim = Simulation(
    model=model,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        simdata = sim.plotting_data

        n1_vals = simdata.grids_log[0]
        n2_vals = simdata.grids_log[1]
        X, Y = xp.meshgrid(n1_vals, n2_vals, indexing="ij")

        os.makedirs(f"{name}/plots", exist_ok=True)
        for f in glob.glob(f"{name}/plots/*.png"):
            os.remove(f)

        def save_plot(data, title, fname, t):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.contourf(X, Y, data, levels=50)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title} at t={t:.3f}")
            plt.savefig(f"{name}/plots/{fname}_{t:.3f}.png", dpi=300)
            plt.close(fig)

        for t in simdata.spline_values.ions.u_log.data.keys():
            u_ions      = simdata.spline_values.ions.u_log.data[t]
            u_electrons = simdata.spline_values.electrons.u_log.data[t]
            phi         = simdata.spline_values.em_fields.phi_log.data[t]

            save_plot(u_ions[0][:, :, 0],      "u_ix", "plot_uix", t)
            save_plot(u_ions[1][:, :, 0],      "u_iy", "plot_uiy", t)
            save_plot(u_electrons[0][:, :, 0], "u_ex", "plot_uex", t)
            save_plot(u_electrons[1][:, :, 0], "u_ey", "plot_uey", t)
            save_plot(phi[0][:, :, 0],         "phi",  "plot_phi", t)