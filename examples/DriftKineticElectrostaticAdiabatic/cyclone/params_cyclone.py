# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Cyclone instability"
description = """
The Cyclone instability is a shear-driven instability that occurs in non-neutral plasmas confined by a magnetic field. 
It typically appears when there is velocity shear in the E×B drift of a plasma column.

The parameter of this simulation file is based on a paper called:

'Nonlinear quasisteady state benchmark of global gyrokinetic codes',
and is described in the part III of the article.

DOI: 10.1140/epjd/e2014-50180-9
"""

# ------------------
# Import Struphy API
# ------------------


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
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import DriftKineticElectrostaticAdiabatic
from struphy.pic.accumulation.filter import FilterParameters

base_units = BaseUnits(
    kBT=0.1916
)  # provides the correct value for epsilon = 1.4142e-3 = 0.36/(180*sqrt(2)) from the paper
model = DriftKineticElectrostaticAdiabatic(
    base_units=base_units,
    use_diagnostic_poisson=True,
)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(sim_folder="sim_1", profiling_activated=True, profiling_trace=True, restart=False)

# Time stepping
time_opts = Time(dt=0.001, Tend=0.01, split_algo="LieTrotter")

a, r_min, R0 = 0.36, 0.01, 1.0
num_elements = (32, 5 * 27, 5)
degree = (3, 3, 3)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(a=a, R0=R0, B0=1.0, q_kind=2, q0=0.86, q1=2.52 + 0.86, l=-0.16, psi_k=5, psi_nel=200)

# Geometry
# domain = domains.Tokamak(equil, num_elements=num_elements[:2], degree=degree[:2], r_min=r_min, num_elements_pre=(128, 512), p_pre=(4, 4), xi_param="sfl", tor_period=19)
domain = domains.HollowTorus(
    a1=r_min, a2=a, R0=R0, sfl=True, pol_period=1, tor_period=19
)  # use a hollowtorus to avoid premaping

# Grid
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, False))

# Derham options
derham_opts = DerhamOptions(
    degree=degree,
    bcs=(("dirichlet", "dirichlet"), None, None),
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
)

# -------------------
# Particle parameters
# -------------------

ppc = 50  # run with 200 minimum
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0, 0, 4, 4))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(boxes_per_dim=(12, 12, 6), do_sort=True, sorting_frequency=5)

# density binning
eta_bin = BinningPlot(slice="e1_e2", n_bins=(64, 64), ranges=((0.01, 0.99), (0.0, 1.0)))
eta_bin2 = BinningPlot(slice="e2_e3", n_bins=(64, 64), ranges=((0.0, 1.0), (0.0, 1.0)))
saving_params = SavingParameters(n_markers=100, binning_plots=(eta_bin,))

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=1.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options(
    which_geometry="toroidal",
    solver_params=SolverParameters(tol=1e-12, maxiter=3000, recycle=False),
    filter_params={model.kinetic_ions.var: FilterParameters("fourier_in_tor", (1,), repeat=1)},
)
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="explicit",
    evaluate_e_field=True,
    maxiter=100,
)
model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(
    algo="explicit",
    evaluate_e_field=True,
    maxiter=100,
)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# piecewise function for initial condition of density, perturbation modes ns=19 and ms=27 in a whole torus geometry
ns = 1
ms = 27
amps = 1.0e-6
kappa_n = 2.23
kappa_Ti = 6.96
Delta_n = Delta_Ti = 0.3
delta_r = 0.02
r0 = 0.5 * a
n0 = 1.0
Ti0 = 1.0


def n_r(r):
    return n0 * xp.exp(-kappa_n * a * Delta_n * xp.tanh((r - r0) / (Delta_n * a)))


def n_init(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = r_min + (a - r_min) * eta1
    return n_r(r)


def Ti_r(r):
    return Ti0 * xp.exp(-kappa_Ti * a * Delta_Ti * xp.tanh((r - r0) / (Delta_Ti * a)))


def vth_init(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = r_min + (a - r_min) * eta1
    return xp.sqrt(Ti_r(r))


def n_xyz(x, y, z):
    r = xp.sqrt((xp.sqrt(x**2 + y**2) - R0) ** 2 + z**2)
    return n_r(r)


def p_xyz(x, y, z):
    r = xp.sqrt((xp.sqrt(x**2 + y**2) - R0) ** 2 + z**2)
    return n_r(r) * Ti_r(r)


equil.p_xyz = p_xyz
equil.n_xyz = n_xyz


def pert_func(*etas):
    if len(etas) == 1:
        e1, e2, e3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 2]
    else:
        e1, e2, e3 = etas[0], etas[1], etas[2]
    r = (a - r_min) * e1 + r_min
    teta = 2 * xp.arctan(xp.sqrt((R0 + r) / (R0 - r)) * xp.tan(xp.pi * e2))
    phi = 2 * xp.pi * e3
    return n_r(r) * amps * xp.exp(-((r - r0) ** 2) / delta_r**2) * xp.cos(ms * teta - ns * phi)


# Background for kinetic species
background = maxwellians.GyroMaxwellian2D(
    n=(n_init, None), vth_para=(vth_init, None), vth_perp=(vth_init, None), equil=equil
)
model.kinetic_ions.var.add_background(background)
# background.plot_density_profile("e1", "e2", domain=domain, plot_3D=True, in_physical=True)
# background.plot_density_profile("e1", "v1", domain=domain)
# background.plot_density_profile("e1", "v2", domain=domain, use_mu=True, equil=equil)

from struphy.initial.base import GenericPerturbation

perturbation = GenericPerturbation(pert_func, given_in_basis="0")
init = maxwellians.GyroMaxwellian2D(
    n=(n_init, perturbation), vth_para=(vth_init, None), vth_perp=(vth_init, None), equil=equil
)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
