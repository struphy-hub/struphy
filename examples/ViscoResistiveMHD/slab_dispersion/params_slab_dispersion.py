# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default ViscoResistiveMHD"
description = """
This is the default simulation for the model ViscoResistiveMHD. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

import logging
from struphy import set_logging_level
set_logging_level(logging.INFO)


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

from struphy.models import ViscoResistiveMHD

# Units
base_units = BaseUnits()

# Model instance
model = ViscoResistiveMHD(base_units=base_units, with_viscosity=False, with_resistivity=False)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.entropy.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions()

# Time stepping
time_opts = Time(dt=0.15, Tend=180.0)

# Geometry
domain = domains.Cuboid(r3=60.0)

# Fluid equilibrium (can be used as part of initial conditions)
B0x = 0.0
B0y = 1.0
B0z = 1.0
beta = 3.0
n0 = 0.7
gamma = 5.0 / 3.0
equil = equils.HomogenSlab(
    B0x=B0x,
    B0y=B0y,
    B0z=B0z,
    beta=beta,
    n0=n0,
)

N_el = 64
grid = grids.TensorProductGrid(num_elements=(1, 1, N_el))

derham_opts = DerhamOptions(degree=(1, 1, 3))

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

model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='full')
model.propagators.variat_mom.options = model.propagators.variat_mom.Options()
model.propagators.variat_ent.options = model.propagators.variat_ent.Options()
model.propagators.variat_mag.options = model.propagators.variat_mag.Options()


# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.
for component in range(3):
    model.mhd.velocity.add_perturbation(
        perturbations.Noise(
            amp=0.001,
            comp=component,
            seed=123,
        )
    )


# Background for (some) FEEC variables
model.mhd.density.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="n3",
    )
)

model.mhd.entropy.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="s3_monoatomic",
    )
)

model.em_fields.b_field.add_background(
    FieldsBackground(
        type="FluidEquilibrium",
        variable="b2",
    )
)


if __name__ == "__main__":
    sim.run(profiling_activated=True)