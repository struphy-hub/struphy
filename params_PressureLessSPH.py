# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default PressureLessSPH"
description = """
This is the default simulation for the model PressureLessSPH. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

import argparse
import os

parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "--backend",
    choices=("numpy", "cupy"),
    default="numpy",
    help="Array backend to run the simulation with (default: numpy).",
)
args = parser.parse_args()

# Must be set before struphy (and therefore cunumpy) is imported.
os.environ["ARRAY_BACKEND"] = args.backend


import logging
from struphy import set_logging_level
set_logging_level(logging.WARNING)

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

# For particles:
from struphy import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    WeightsParameters,
    SortingParameters,
    SavingParameters,
    maxwellians,
)

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import PressureLessSPH

# Units
base_units = BaseUnits()

# Model instance
model = PressureLessSPH(base_units=base_units)

# List all variables and decide whether to save their data
model.cold_fluid.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.backend}",
    profiling_activated=True,
    save_restart=False,
)


# Time stepping
# 10 steps: long enough to average out start-up effects when comparing the
# NumPy and CuPy backends, short enough to iterate on.
time_opts = Time(dt=0.01, Tend=0.1)

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid(num_elements=(32, 32, 16))

# Derham options
derham_opts = DerhamOptions()

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

# -------------------
# Particle parameters
# -------------------

# Np and the grid above are sized for backend comparisons: big enough that the
# particle push dominates the run, small enough to fit comfortably on one GPU.
# The seed is fixed because marker loading is otherwise unseeded, and two runs
# of the *same* backend then differ enough to swamp any backend comparison.
loading_params = LoadingParameters(Np=1_000_000, seed=1234)
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
sorting_params = SortingParameters()
saving_params = SavingParameters()
model.cold_fluid.set_markers(loading_params=loading_params,
                             weights_params=weights_params,
                             boundary_params=boundary_params,
                             sorting_params=sorting_params,
                             saving_params=saving_params,
                             )

# ------------------
# Propagator options
# ------------------

model.propagators.push_eta.options = model.propagators.push_eta.Options()
phi = equil.p0
model.propagators.push_v.phi = phi
model.propagators.push_v.options = model.propagators.push_v.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) sph variables
background = equils.ConstantVelocity()
model.cold_fluid.var.add_background(background)

# Perturbations for (some) sph variables
perturbation = perturbations.TorusModesCos()
model.cold_fluid.var.add_perturbation(del_n=perturbation)

if __name__ == "__main__":
    sim.run()
