# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = f"""
This is the default simulation for the model VlasovMaxwellOneSpecies. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

# ------------------
# Import Struphy API
# ------------------

# For particles:
from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    KernelDensityPlot,
    LoadingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import VlasovMaxwellOneSpecies

# ---------------------
# Instance of the model
# ---------------------

model = VlasovMaxwellOneSpecies()

# List all species and set their physical properties (charge and mass number, etc.)
model.em_fields.set_species_properties()
model.kinetic_ions.set_species_properties()

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True
model.measure_gauss_error(measure = True)

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="cvT")

# Units
base_units = BaseUnits()

# Time stepping
time_opts = Time(Tend = 0.01)

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid(Nel = (24, 1, 1))

# Derham options
derham_opts = DerhamOptions()

# Simulation object
sim = Simulation(
    model=model,
    params_path=__file__,
    env=env,
    base_units=base_units,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

# -------------------
# Particle parameters
# -------------------

loading_params = LoadingParameters(ppc = 1_000, seed = 1234)
weights_params = WeightsParameters(control_variate = True)
boundary_params = BoundaryParameters()
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               )
model.kinetic_ions.set_sorting_boxes()

binplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))
model.kinetic_ions.set_save_data(binning_plots=(binplot,))

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
model.em_fields.phi.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
model.em_fields.phi.add_perturbation(perturbations.TorusModesCos())

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species
maxwellian_1 = maxwellians.Maxwellian3D(n=(1.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(0.1, None))
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.TorusModesCos()
maxwellian_1pt = maxwellians.Maxwellian3D(n=(1.0, perturbation))
init = maxwellian_1pt + maxwellian_2
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    print("Executing simulation")
    sim.run(verbose=True)

    # ------------------
    # Sister simulation object with control_variate = False
    # ------------------
    weights_params = WeightsParameters(control_variate = False)
    env = EnvironmentOptions(sim_folder="cvF")

    sister_sim = sim.spawn_sister(env=env)

    print("Executing sister simulation")
    sister_sim.model.kinetic_ions.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        )
    sister_sim.run(verbose=True)