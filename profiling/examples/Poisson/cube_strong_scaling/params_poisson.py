# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Poisson strong scaling on 3D cube"
description = """
Strong scaling test for Poisson equation on a 3D cube.
The manufactured solution is a simple product of sines and cosines.
Homogeneous Dirichlet boundary conditions are set in direction x.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.WARNING)

import argparse

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import Poisson

# Units
base_units = BaseUnits()

# Model instance
model = Poisson(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.em_fields.source.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
args, _ = parser.parse_known_args()

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    restart=False,
)

# Time stepping
time_opts = Time()

# Geometry
Lx = 2.0
Ly = 3.0
Lz = 4.0
domain = domains.Cuboid(r1=Lx, l2=-Ly/2, r2=Ly/2, r3=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(256, 256, 256), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Simulation object
sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

# ------------------
# Propagator options
# ------------------

from struphy.linear_algebra.solver import SolverParameters
solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=True)
model.propagators.poisson.options = model.propagators.poisson.Options(stab_eps=0.0,
                                                                      solver="pcg", 
                                                                      precond=None,
                                                                      solver_params=solver_params,
                                                                      )

# ------------------
# Initial conditions
# ------------------
import numpy as np
from struphy.initial.base import GenericPerturbation

def exact_solution(x, y, z):
    return np.sin(np.pi / Lx * x) * np.cos(12 * np.pi / Ly * y + 4 * np.pi / Lz * z)

def rhs_fun(x, y, z):
    return exact_solution(x, y, z) * ((np.pi / Lx) ** 2 + (12 * np.pi / Ly) ** 2 + (4 * np.pi / Lz) ** 2)

rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="physical")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    sim.run(profiling_activated=True, one_time_step=True)
    sim.pproc(parallel_pproc=True)
    
    def plot_slices(num, exact, name, slice_pt_x=0, slice_pt_y=0, slice_pt_z=0):
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(16, 12))
        
        plt.subplot(3, 3, 1)
        plt.pcolor(y[slice_pt_x, :, :], z[slice_pt_x, :, :], num[slice_pt_x, :, :])
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("{} from struphy, slice at x = {:.2f}".format(name, x[slice_pt_x, 0, 0]))
        
        plt.subplot(3, 3, 4)
        plt.pcolor(y[slice_pt_x, :, :], z[slice_pt_x, :, :], exact(x[slice_pt_x, :, :], y[slice_pt_x, :, :], z[slice_pt_x, :, :]))
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("{} exact, slice at x = {:.2f}".format(name, x[slice_pt_x, 0, 0]))
        
        plt.subplot(3, 3, 7)
        plt.pcolor(y[slice_pt_x, :, :], z[slice_pt_x, :, :], np.abs(num[slice_pt_x, :, :] - exact(x[slice_pt_x, :, :], y[slice_pt_x, :, :], z[slice_pt_x, :, :])))
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("{} error, slice at x = {:.2f}".format(name, x[slice_pt_x, 0, 0]))
        
        plt.subplot(3, 3, 2)
        plt.pcolor(x[:, slice_pt_y, :], z[:, slice_pt_y, :], num[:, slice_pt_y, :])
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("{} from struphy, slice at y = {:.2f}".format(name, y[0, slice_pt_y, 0]))
        
        plt.subplot(3, 3, 5)
        plt.pcolor(x[:, slice_pt_y, :], z[:, slice_pt_y, :], exact(x[:, slice_pt_y, :], y[:, slice_pt_y, :], z[:, slice_pt_y, :]))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("{} exact, slice at y = {:.2f}".format(name, y[0, slice_pt_y, 0]))
        
        plt.subplot(3, 3, 8)
        plt.pcolor(x[:, slice_pt_y, :], z[:, slice_pt_y, :], np.abs(num[:, slice_pt_y, :] - exact(x[:, slice_pt_y, :], y[:, slice_pt_y, :], z[:, slice_pt_y, :])))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("{} error, slice at y = {:.2f}".format(name, y[0, slice_pt_y, 0]))
        
        plt.subplot(3, 3, 3)
        plt.pcolor(x[:, :, slice_pt_z], y[:, :, slice_pt_z], num[:, :, slice_pt_z])
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("{} from struphy, slice at z = {:.2f}".format(name, z[0, 0, slice_pt_z]))
        
        plt.subplot(3, 3, 6)
        plt.pcolor(x[:, :, slice_pt_z], y[:, :, slice_pt_z], exact(x[:, :, slice_pt_z], y[:, :, slice_pt_z], z[:, :, slice_pt_z]))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("{} exact, slice at z = {:.2f}".format(name, z[0, 0, slice_pt_z]))
        
        plt.subplot(3, 3, 9)
        plt.pcolor(x[:, :, slice_pt_z], y[:, :, slice_pt_z], np.abs(num[:, :, slice_pt_z] - exact(x[:, :, slice_pt_z], y[:, :, slice_pt_z], z[:, :, slice_pt_z])))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("{} error, slice at z = {:.2f}".format(name, z[0, 0, slice_pt_z]))

        return fig

    if sim.comm.rank == 0:
        sim.load_plotting_data()
        
        Tstart = sim.t_grid[0]
        rhs_data = sim.spline_values.em_fields.source_log
        print(rhs_data)
        rhs = rhs_data.data[Tstart][0]
        
        Tend = sim.t_grid[-1]
        phi_data = sim.spline_values.em_fields.phi_log
        print(phi_data)
        phi = phi_data.data[Tend][0]
        x = sim.grids_phy[0]
        y = sim.grids_phy[1]
        z = sim.grids_phy[2]
        
        slice_pt_x = x.shape[0] // 2
        slice_pt_y = y.shape[1] // 2
        slice_pt_z = 0
        
        fig_rhs = plot_slices(rhs, rhs_fun, "RHS", slice_pt_x=slice_pt_x, slice_pt_y=slice_pt_y, slice_pt_z=slice_pt_z)
        fig_phi = plot_slices(phi, exact_solution, "Phi", slice_pt_x=slice_pt_x, slice_pt_y=slice_pt_y, slice_pt_z=slice_pt_z)

        rel_err_rhs = np.max(np.abs(rhs - rhs_fun(x, y, z))) / np.max(np.abs(rhs_fun(x, y, z)))
        rel_err_phi = np.max(np.abs(phi - exact_solution(x, y, z))) / np.max(np.abs(exact_solution(x, y, z)))

        print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
        print(f"Max relative error in Phi: {rel_err_phi:.2e}")

        assert rel_err_rhs < 1e-3, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_phi < 1e-2, f"The computed solution does not match the exact solution, max rel error = {rel_err_phi}."

        import os
        # `path_out` is the run's output folder; `sim_folder` alone is a bare name
        # resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-run<id>`.
        results_dir = os.path.join(sim.env.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
        np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
        np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

        fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
        fig_phi.savefig(os.path.join(results_dir, "phi_slices.png"))
