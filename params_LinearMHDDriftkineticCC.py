# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default LinearMHDDriftkineticCC"
description = """
This is the default simulation for the model LinearMHDDriftkineticCC. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

import logging
import numpy as np
from struphy import set_logging_level
from struphy.initial.base import Perturbation
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

from struphy.models import LinearMHDDriftkineticCC

# Units
base_units = BaseUnits()

# Model instance
model = LinearMHDDriftkineticCC(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.pressure.save_data = True
model.mhd.velocity.save_data = True
model.energetic_ions.var.save_data = True

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

# -------------------
# Particle parameters
# -------------------

loading_params = LoadingParameters()
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
sorting_params = SortingParameters()
saving_params = SavingParameters()
model.energetic_ions.set_markers(loading_params=loading_params,
                                 weights_params=weights_params,
                                 boundary_params=boundary_params,
                                 sorting_params=sorting_params,
                                 saving_params=saving_params,
                                 )

# ------------------
# Propagator options
# ------------------

model.propagators.push_bxe.options = model.propagators.push_bxe.Options()
model.propagators.push_parallel.options = model.propagators.push_parallel.Options()
model.propagators.shearalfen_cc5d.options = model.propagators.shearalfen_cc5d.Options()
model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options()
model.propagators.cc5d_density.options = model.propagators.cc5d_density.Options()
model.propagators.cc5d_gradb.options = model.propagators.cc5d_gradb.Options()
model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.


class RadialVelocityPerturbation(Perturbation):
    """User-defined vector-component perturbation with a radial envelope.

    This class is saved inline in ``run_metadata.json`` and can be restored with
    ``Simulation.from_output(..., trust_initial_condition_source=True)``.
    """

    def __init__(self, amplitude=0.02, comp=0):
        self.params = {"amplitude": amplitude, "comp": comp}
        self.given_in_basis = "v"
        self.comp = comp

    def __call__(self, eta1, eta2, eta3, flat_eval=False):
        return self.params["amplitude"] * np.sin(np.pi * eta1) * np.cos(2.0 * np.pi * eta2)


# Background for (some) FEEC variables
model.mhd.velocity.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=0))
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=1))
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=2))
# A custom perturbation class can be combined with built-in perturbations.
model.mhd.velocity.add_perturbation(RadialVelocityPerturbation(amplitude=0.02, comp=0))

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# User-defined profiles can be used anywhere a kinetic Maxwellian accepts a
# callable. They are written inline to run_metadata.json and restored with
# Simulation.from_output(..., trust_initial_condition_source=True).
# The restoration namespace supplies np/numpy and xp/cp/cupy.
def energetic_ion_density(*etas):
    """A weak periodic density modulation."""
    eta1, eta2, eta3 = etas[0].T if len(etas) == 1 else etas
    return 1.0 + 0.05 * np.cos(2.0 * np.pi * eta2)


def energetic_ion_parallel_flow(*etas):
    """A small parallel flow with a radial envelope."""
    eta1, eta2, eta3 = etas[0].T if len(etas) == 1 else etas
    return 0.1 * np.sin(np.pi * eta1) * np.sin(2.0 * np.pi * eta3)


def energetic_ion_parallel_thermal_speed(*etas):
    """A positive, smoothly varying parallel thermal speed."""
    eta1, eta2, eta3 = etas[0].T if len(etas) == 1 else etas
    return 1.0 + 0.05 * np.cos(2.0 * np.pi * eta1)


# Background for kinetic species
maxwellian_1 = maxwellians.GyroMaxwellian2D(
    n=(energetic_ion_density, None),
    u_para=(energetic_ion_parallel_flow, None),
    vth_para=(energetic_ion_parallel_thermal_speed, None),
)
maxwellian_2 = maxwellians.GyroMaxwellian2D(n=(0.1, None))
background = maxwellian_1 + maxwellian_2
model.energetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.TorusModesCos()
maxwellian_1pt = maxwellians.GyroMaxwellian2D(
    n=(energetic_ion_density, perturbation),
    u_para=(energetic_ion_parallel_flow, None),
    vth_para=(energetic_ion_parallel_thermal_speed, None),
)
init = maxwellian_1pt + maxwellian_2
model.energetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
