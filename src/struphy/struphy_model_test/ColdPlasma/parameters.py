# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default ColdPlasma"
description = """
This is the default simulation for the model ColdPlasma. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

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
    ProfilingOptions,
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
from struphy.models import ColdPlasma

# Units
base_units = BaseUnits()

# Model instance
model = ColdPlasma(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.electrons.current.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions()

# Time stepping
time_opts = Time()

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid()

# Derham options
derham_opts = DerhamOptions()

# Profiling options
profiling_opts = ProfilingOptions()

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
    profiling_opts=profiling_opts,
)

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.ohm.options = model.propagators.ohm.Options()
model.propagators.jxb.options = model.propagators.jxb.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
model.electrons.current.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
model.electrons.current.add_perturbation(perturbations.TorusModesCos(given_in_basis="v", comp=0))
model.electrons.current.add_perturbation(perturbations.TorusModesCos(given_in_basis="v", comp=1))
model.electrons.current.add_perturbation(perturbations.TorusModesCos(given_in_basis="v", comp=2))

if __name__ == "__main__":
    sim.run()
