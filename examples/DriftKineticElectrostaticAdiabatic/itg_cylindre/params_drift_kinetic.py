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
from struphy.models import DriftKineticElectrostaticAdiabatic

# ------------------
# Import Struphy API
# ------------------
from struphy.propagators import implicit_diffusion

restart = False

# Units
base_units = BaseUnits(kBT=1.0)

# Model instance
model = DriftKineticElectrostaticAdiabatic(
    base_units=base_units,
    epsilon=1.0,
    use_diagnostic_poisson=True,
)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="sim_1", restart=False)

# Time stepping
time_opts = Time(dt=5.0, Tend=500.0, split_algo="LieTrotter")

# Geometry
a1, a2, Lz = 0.1, 14.5, 1506.759067
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)

# Grid
Nx, Ny, Nz = (32, 5 * 10, 10)
num_element = (Nx, Ny, Nz)
grid = grids.TensorProductGrid(num_elements=num_element, mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Simulation object
from feectools.ddm.mpi import mpi as MPI

rank = MPI.COMM_WORLD.Get_rank()
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

ppc = 50  # run with 200 minimum
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0.0, 0.0, 2.0, 2.0))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(
    do_sort=True,
    boxes_per_dim=(12, 12, 6),
    sorting_frequency=0,
)

binplot = BinningPlot(slice="e1_e2", n_bins=(64, 128), ranges=((0.0, 1.0), (0.0, 1.0)))
saving_params = SavingParameters(binning_plots=(binplot,))

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options.solver_params = implicit_diffusion.SolverParameters(maxiter=3000, tol=1e-14)
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(algo="explicit", evaluate_e_field=True)
model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(algo="explicit", evaluate_e_field=True)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
# model.em_fields.phi.add_background(FieldsBackground(values=(0.0,)))

# Perturbations for (some) FEEC variables
# model.em_fields.phi.add_perturbation(perturbations.TorusModesCos())

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species


rp = (a2 + a1) / 2
amps = 1e-6
ms, ns = 5, 1
kappa_n0 = 0.055
kappa_Ti = kappa_Te = 0.27586
delta_r_Ti = delta_r_Te = 1.45
delta_r_n0 = delta_r_Ti / 2
delta_r = 4 * delta_r_n0 / delta_r_Ti
C_Ti = C_Te = 1.0
N_integrate = 1000000
C_n0 = (a2 - a1) / xp.sum(
    xp.exp(-kappa_n0 * delta_r_n0 * xp.tanh((xp.linspace(a1, a2, N_integrate) - rp) / delta_r_n0))
    * (a2 - a1)
    / N_integrate
)


def n0(r):
    return C_n0 * xp.exp(-kappa_n0 * delta_r_n0 * xp.tanh((r - rp) / delta_r_n0))


from struphy.initial.base import GenericPerturbation


def n_init(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = a1 + (a2 - a1) * eta1
    return n0(r)


def pert_func(*etas):
    if len(etas) == 1:
        e1, e2, e3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 2]
    else:
        e1, e2, e3 = etas[0], etas[1], etas[2]
    r = a1 + (a2 - a1) * e1
    teta = 2 * xp.pi * e2
    z = Lz * e3
    return n0(r) * amps * xp.exp(-((r - rp) ** 2) / delta_r**2) * xp.cos(2 * xp.pi * ns * z / Lz + ms * teta)


def Ti(r):
    return C_Ti * xp.exp(-kappa_Ti * delta_r_Ti * xp.tanh((r - rp) / delta_r_Ti))


def vth_i(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = a1 + (a2 - a1) * eta1
    return xp.sqrt(Ti(r))


def vth_e(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = a1 + (a2 - a1) * eta1
    return xp.sqrt(Ti(r))


def n0_xyz(x, y, z):
    r = xp.sqrt(x**2 + y**2)
    return n0(r)


def p_xyz(x, y, z):
    r = xp.sqrt(x**2 + y**2)
    return n0(r) * Ti(r)


equil.p_xyz = p_xyz
equil.n_xyz = n0_xyz


perturbation = GenericPerturbation(pert_func)
background = maxwellians.GyroMaxwellian2D(
    n=(n_init, None),
    vth_para=(vth_i, None),
    vth_perp=(vth_i, None),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_background(background)
init = maxwellians.GyroMaxwellian2D(
    n=(n_init, perturbation),
    vth_para=(vth_i, None),
    vth_perp=(vth_i, None),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run(profiling_activated=True)
