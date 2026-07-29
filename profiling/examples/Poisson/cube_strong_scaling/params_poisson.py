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
    profiling_activated=True,
    profiling_trace=True,
    restart=False,
)

# Time stepping
time_opts = Time()

# Geometry
Lx = 2.0
Ly = 3.0
Lz = 4.0
domain = domains.Cuboid(r1=Lx, l2=-Ly/2.0, r2=Ly/2.0, r3=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(32, 128, 128), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 2), bcs=(("dirichlet", "dirichlet"), None, None))

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

model.propagators.poisson.options = model.propagators.poisson.Options(solver="pcg", precond=None)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
model.em_fields.source.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
from struphy.initial.base import GenericPerturbation
import numpy as np

def exact_solution(x, y, z):
    return np.sin(np.pi/Lx * x) * np.cos(8*np.pi/Ly * y) * np.sin(4*np.pi/Lz * z)

def rhs(x, y, z):
    return exact_solution(x, y, z) * ((np.pi/Lx)**2 + (8*np.pi/Ly)**2 + (4*np.pi/Lz)**2)

rhs_perturbation = GenericPerturbation(rhs, given_in_basis="physical")

model.em_fields.source.add_perturbation(rhs_perturbation)

if __name__ == "__main__":
    sim.run(one_time_step=True) 