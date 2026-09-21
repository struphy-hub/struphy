# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default LinearMHD"
description = """
This is the default simulation for the model LinearMHD. 
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
import numpy as np

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import LinearMHD

# Units
base_units = BaseUnits()

# Model instance
model = LinearMHD(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    save_step=1,
    out_folders="/u/shrusi/git_repos/struphy/examples/LinearMHD/itpa_tae_benchmark/",
    sim_folder="sim4_higherResolution",
    max_runtime=60
    )

# Time stepping
time_opts = Time(dt=0.1, Tend=10.)

# Geometry
domain = domains.HollowTorus(
    a1=0.1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=6
)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=1.0,
    R0=10.0,
    B0=3.0,
    q_kind=0,
    p_kind=1,
    q0=1.71,
    q1=1.87,
    p1=0.95,
    p2=0.05,
    beta=0.0018 
    )
# Grid
grid = grids.TensorProductGrid(num_elements=(24,96,16))

# Derham options
derham_opts = DerhamOptions(
    degree=(3, 3, 3),
    bcs=(("dirichlet", "dirichlet"), None, None)
)

# Profiling options
profiling_opts = ProfilingOptions(
    use_line_profiler=True
)

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
    profiling_opts=profiling_opts
)

# ------------------
# Propagator options
# ------------------

model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).

# Background for (some) FEEC variables

# Perturbations for (some) FEEC variables
ms_radial_1, ms_radial_2 = 10, 11
perturbation_radial = perturbations.TorusModesSin(
    ms              = (ms_radial_1, ms_radial_2),
    ns              = (-1, -1),
    amps            = (1e-3, 1e-3),
    pfuns           = ("exp", "exp"),
    pfun_params     = ([0.5,0.1], [0.5,0.1]),
    comp            = 0,
    given_in_basis="2"
)
perturbation_poloidal = perturbations.TorusModesCos(
    ms              = (10, 11),
    ns              = (-1, -1),
    amps            = (1e-3 * 1 / (2*np.pi * ms_radial_1), 1e-3 * 1 / (2*np.pi * ms_radial_2)),
    pfuns           = ("d_exp", "d_exp"),
    pfun_params     = ([0.5,0.1], [0.5,0.1]),
    comp            = 1,
    given_in_basis="2"
)
model.mhd.velocity.add_perturbation(perturbation=perturbation_radial)
model.mhd.velocity.add_perturbation(perturbation=perturbation_poloidal)



if __name__ == "__main__":
    sim.run(profiling_activated=True)