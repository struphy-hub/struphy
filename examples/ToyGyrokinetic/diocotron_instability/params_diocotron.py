# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Diocotron instability"
description = """
The Diocotron instability is a shear-driven instability that occurs in non-neutral plasmas confined by a magnetic field. 
It typically appears when there is velocity shear in the E×B drift of a plasma column.

The parameter of this simulation file is based on a paper called:

'A new fully two-dimensional conservative semi-Lagrangian
method: applications on polar grids, from diocotron instability
to ITG turbulence'

DOI: 10.1140/epjd/e2014-50180-9
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
    maxwellians,
)

import cunumpy as xp

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import ToyDrift
model = ToyDrift()

# List all species and set their physical properties (charge and mass number, etc.)
model.em_fields.set_species_properties()
model.kinetic_ions.set_species_properties(alpha=1.0, epsilon=1.0)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="simdata")

# Units
base_units = BaseUnits(kBT=1.0)

# Time stepping
time_opts = Time(dt=0.05, Tend=20.0, split_algo="LieTrotter")

# Geometry
domain = domains.HollowCylinder(a1=1.0, a2=10.0, Lz=10.0)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid(Nel=(32,64,1), mpi_dims_mask=(False,True,False))

# Derham options
derham_opts = DerhamOptions(
    p=(3,3,1), 

    # impose dirichlet boundary conditions at r_min and r_max
    spl_kind=(False,True,True), 
    dirichlet_bc=(
        (True, True),
        (False, False),
        (False, False),
    ))

# Simulation object
sim = Simulation(
    model=model,
    name=name,
    description=description,
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

loading_params = LoadingParameters(ppc = 500, seed=1234)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               bufsize=2.0,
                               )
model.kinetic_ions.set_sorting_boxes(boxes_per_dim=(16,16,1), do_sort=True)

# density binning
eta_bin = BinningPlot(slice='e1_e2', n_bins= (128,128), ranges= ((0.0, 1.0), (0.0,1.0)))
model.kinetic_ions.set_save_data(binning_plots=(eta_bin, ))

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options()
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(phi=model.em_fields.phi, evaluate_e_field=True)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# piecewise function for initial condition of density
r_minus, r_plus = 4.0, 5.0
ms = 4
def n_init(etas,r_minus=r_minus,r_plus=r_plus):

    # transform logical coordinate to polar
    a1, a2 = domain.params["a1"], domain.params["a2"]
    radial = (a1 + (a2 - a1) * etas[:,0])

    return 1.0 * ( (r_minus <= radial) & (radial < r_plus) )

# Background for kinetic species
background = maxwellians.GyroMaxwellian2D(n=(0.0, None), equil=equil)
model.kinetic_ions.var.add_background(background)


eta_minus = (r_minus - domain.params["a1"])/(domain.params["a2"] - domain.params["a1"])
eta_plus = (r_plus - domain.params["a1"])/(domain.params["a2"] - domain.params["a1"])

# Perturbations for (some) kinetic species

# for linear case amps = (1e-6,)
perturbation = perturbations.ModesCos(amps=(0.5,), ms=(ms,), perb_domain=((eta_minus,eta_plus), None, None))
init = maxwellians.GyroMaxwellian2D(n=(n_init, perturbation), equil=equil)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run(verbose=True)
