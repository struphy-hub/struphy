import os
# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = """
Weak Landau damping: A linear test case for the VlasovAmpereOneSpecies model.
This test involves a small amplitude electrostatic perturbation in a uniform, collisionless plasma.
The perturbation is damped due to phase mixing effects (Landau damping) as particles interact with 
the self-consistent electric field. This benchmark validates the numerical discretization of the 
Vlasov-Ampère system and the accuracy of particle-in-cell methods.
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
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import VlasovAmpereOneSpecies

# Units
base_units = BaseUnits()

# Model instance
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0 = False)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="sim_data_petsc", out_folders=os.environ.get("STRUPHY_PROFILING_OUT_FOLDERS", os.getcwd()))

# Time stepping
time_opts = Time(dt = 0.05, Tend = 20.0, split_algo = "LieTrotter")

# Geometry
domain = domains.Cuboid(r1 = 12.56) # r1 -> pi * 4 -> k = 0.5

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(16, 16, 16))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

# Simulation object
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

# -------------------
# Particle parameters
# -------------------

loading_params = LoadingParameters(ppc=20, seed=42)
weights_params = WeightsParameters(control_variate= True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(4, 4, 4), do_sort=True)

binplot = BinningPlot(slice='e1_v1', n_bins= (128, 128), ranges= ((0.,1.), (-5.,5.)))
saving_params = SavingParameters(binning_plots=(binplot,))

model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               sorting_params=sorting_params,
                               saving_params=saving_params,
                               bufsize = 0.4,
                               )

# ------------------
# Propagator options
# ------------------

model.propagators.push_eta.options = model.propagators.push_eta.Options() 
if model.with_B0:
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0", solver="petsc")

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps = (0.001,), ls = (1,))
init = maxwellians.Maxwellian3D(n = (1.0,perturbation))
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()