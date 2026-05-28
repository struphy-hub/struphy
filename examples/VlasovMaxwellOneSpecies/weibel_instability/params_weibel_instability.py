# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = """
Weibel instability: A linear test case for the VlasovMaxwellOneSpecies model.
This test considers a plasma with an anisotropic velocity distribution, where 
temperature differs between directions. Small magnetic perturbations grow due to the anisotropy, 
leading to the generation of transverse magnetic fields. 

Note that for this simulaiton, the control_variate is set to False as it violates the Gauss law, 
and create growth in electric field.
"""

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

from struphy.models import VlasovMaxwellOneSpecies

# Units
base_units = BaseUnits()

# Model instance
model = VlasovMaxwellOneSpecies(base_units=base_units,
                                alpha=1.0, 
                                epsilon=-1.0,
                                measure_gauss_law=True)

# ---------------------
# Parameters setup
# ---------------------

import cunumpy as xp
k = 1.25
B_pert_amp = -1e-4

vth1_background_val = 0.02/xp.sqrt(2)
vth2_background_val = vth1_background_val * xp.sqrt(12)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="sim_data")

# Time stepping
time_opts = Time(dt = 0.05, Tend = 400, split_algo = "LieTrotter")

# Geometry
domain = domains.Cuboid(r1 = 2*xp.pi/k)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements = (32,1,1))

# Derham options
derham_opts = DerhamOptions(degree = (3,1,1))

# Siumlation object
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

loading_params = LoadingParameters(Np = 100000, 
                                   set_zero_velocity = (False, False, True), 
                                   moments = (0.0,0.0,0.0,vth1_background_val,vth2_background_val,1.0),
                                   seed=1234,
                                   )
weights_params = WeightsParameters(control_variate = False)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim = (16,1,1), do_sort = True)

binplot_dens = BinningPlot(slice="e1_v1", n_bins= (128, 128), ranges= ((0.,1.), (-0.1,0.1))) 
binplot_velocity = BinningPlot(slice="v1_v2", n_bins= (128, 128), ranges= ((-0.1,0.1), (-0.1,0.1))) 
binplot_current = tuple(
    [BinningPlot(slice=f"e{i}", n_bins= 32, ranges= (0.,1.), output_quantity=f"current_{j}") for j in range(1,4) for i in range(1,4)] 
    )
saving_params = SavingParameters(binning_plots=(binplot_dens, binplot_velocity, *binplot_current))

model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               sorting_params=sorting_params,
                               saving_params=saving_params,
                               bufsize = 2.0,
                               )

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

maxwellian = maxwellians.Maxwellian3D(
                    vth1=(vth1_background_val, None) , vth2=(vth2_background_val, None)
                )
model.kinetic_ions.var.add_background(maxwellian)

# Perturbations of initial magnetic field
model.em_fields.b_field.add_perturbation(perturbation = perturbations.ModesCos(amps=(B_pert_amp,), ls = (1,), comp = 2)) # Initial Bz depending on x-axis

if __name__ == "__main__":
    sim.run(verbose=True)