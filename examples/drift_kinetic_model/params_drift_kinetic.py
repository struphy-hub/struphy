# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Default DriftKineticElectrostaticAdiabatic"
description = """
This is the default simulation for the model DriftKineticElectrostaticAdiabatic. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.
"""

import logging
from struphy import set_logging_level
set_logging_level(logging.INFO)

import cunumpy as xp

# ------------------
# Import Struphy API
# ------------------

from struphy.propagators import propagators_fields

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

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import DriftKineticElectrostaticAdiabatic

# Units
base_units = BaseUnits(kBT=1.0)

# Model instance
model = DriftKineticElectrostaticAdiabatic(base_units=base_units, epsilon=1.0, alpha=1.0)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="sim_2")

# Time stepping
time_opts = Time(dt=0.01, Tend=0.01, split_algo="Strang")

# Geometry
a1, a2, Lz = 1.0, 14.5, 1506.759067
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)

# Grid
Nx, Ny, Nz = 16, 32, 16
grid = grids.TensorProductGrid(num_elements=(Nx, Ny, Nz),mpi_dims_mask=(True,True,True))

# Derham options
derham_opts = DerhamOptions(degree=(1,1,1), bcs=(("dirichlet", "dirichlet"), None, None))

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

ppc = 200
loading_params = LoadingParameters(ppc=ppc, loading="pseudo_random", seed=1234)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=('remove','periodic','periodic'))
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               )
model.kinetic_ions.set_sorting_boxes(do_sort=True, boxes_per_dim=(12,24,12), sorting_frequency=0)

binplot = BinningPlot(slice='e1_e2', n_bins=(64,128), ranges=((0.0, 1.0), (0.0, 1.0)))
model.kinetic_ions.set_save_data(binning_plots=(binplot,))

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options(solver_params=propagators_fields.SolverParameters(maxiter=5000, info=True))
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(phi=model.em_fields.phi, algo="explicit", evaluate_e_field=True)
model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(phi=model.em_fields.phi, algo="explicit", evaluate_e_field=True)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
#model.em_fields.phi.add_background(FieldsBackground(values=(0.0,)))

# Perturbations for (some) FEEC variables
#model.em_fields.phi.add_perturbation(perturbations.TorusModesCos())

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species


rp = (a2+a1)/2
amps=1e-6
ms, ns = 5, 1
kappa_n0 = 0.055
kappa_Ti = kappa_Te = 0.27586
delta_r_Ti = delta_r_Te = 1.45
delta_r_n0 = delta_r_Ti/2
delta_r = 4 * delta_r_n0 / delta_r_Ti
C_Ti = C_Te = 1.0
N_integrate = 1000000
C_n0 = (a2-a1) / xp.sum(xp.exp(-kappa_n0*delta_r_n0*xp.tanh((xp.linspace(a1,a2,N_integrate)-rp)/delta_r_n0))*(a2-a1)/N_integrate)

def n0(r):
    return C_n0 * xp.exp(-kappa_n0*delta_r_n0*xp.tanh((r-rp)/delta_r_n0))
from struphy.initial.base import GenericPerturbation

def n_init(*etas):
    if len(etas)==1:
        eta1=etas[0][:,0]
    else:
        eta1=etas[0]
    r = (a1 + (a2 - a1) * eta1)
    return n0(r)

def pert_func(*etas):
    if len(etas)==1:
        e1,e2,e3 = etas[0][:,0], etas[0][:,1], etas[0][:,1]
    else:
        e1, e2, e3 = etas[0], etas[1], etas[2]
    r = (a1 + (a2 - a1) * e1)
    teta = 2*xp.pi * e2
    z = Lz * e3
    return n0(r)*amps*xp.exp(-(r-rp)**2/delta_r**2)*xp.cos(2*xp.pi*ns*z/Lz + ms*teta)

def vth_i(*etas):
    if len(etas)==1:
        eta1=etas[0][:,0]
    else:
        eta1=etas[0]
    r = (a1 + (a2 - a1) * eta1)
    return 1.0#xp.sqrt(C_Ti * xp.exp(-kappa_Ti*delta_r_Ti*xp.tanh((r-rp)/delta_r_Ti)))

def vth_e(*etas):
    if len(etas)==1:
        eta1=etas[0][:,0]
    else:
        eta1=etas[0]
    r = (a1 + (a2 - a1) * eta1)
    return xp.sqrt(C_Te * xp.exp(-kappa_Te*delta_r_Te*xp.tanh((r-rp)/delta_r_Te)))

#import matplotlib.pyplot as plt
#plt.plot(xp.linspace(0,1,100),n_init(xp.array([[xp.linspace(0,1,100), xp.linspace(0,1,100), xp.linspace(0,1,100)]]))[0])
#plt.show()

# Perturbations for (some) kinetic species

perturbation = GenericPerturbation(pert_func)
background = maxwellians.GyroMaxwellian2D(n=(n_init, None), equil=equil)
#background.plot_density_profile(dim_1="e1", dim_2="e2", in_physical=True, domain=domain, resol=100, integrate_resol=10, logical_coord=(0.0,0.0,0.0), plot_3D=False, use_mu=True)
model.kinetic_ions.var.add_background(background)
init = maxwellians.GyroMaxwellian2D(n=(n_init, perturbation), equil=equil, vth_para=(vth_i,None), vth_perp=(vth_i,None))
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run(verbose=True, one_time_step=False)