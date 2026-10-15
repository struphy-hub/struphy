"""Cyclone base case with the DriftKineticElectrostaticAdiabatic model: toroidal ion temperature gradient (ITG)
instability of an n = 19 mode in an ad hoc tokamak with circular flux surfaces, with the profiles of the global
gyrokinetic benchmark of Lapillonne et al. (2010), based on the Cyclone base case of Dimits et al. (2000)."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Cyclone base case (DriftKineticElectrostaticAdiabatic)"
description = """
Cyclone base case: ion temperature gradient (ITG) driven instability in a tokamak with circular flux surfaces,
with drift-kinetic ions and adiabatic electrons (DriftKineticElectrostaticAdiabatic). The radial gradients of the
ion temperature and density drive the instability, which grows exponentially from a small perturbation and
saturates into ITG turbulence.

Physics. The ion guiding-center distribution f(X, v_par, mu, t) is advected along the characteristics
    dX/dt     = v_par B*/B*_par + E* x b0 / B*_par,
    dv_par/dt = (1/epsilon) B*/B*_par . E*,
with E* = -grad phi - epsilon mu grad|B0|, B* = B0 + epsilon v_par curl b0 and B*_par = B* . b0, i.e. parallel
streaming, mirror force and the E x B, grad-B and curvature drifts (epsilon = normalized cyclotron period, mu the
magnetic moment). The electrons respond adiabatically, n_e = n0 (1 + e phi / T_e), and the electrostatic
potential follows from the quasi-neutrality (gyrokinetic Poisson) equation
    -div_perp( n0/|B0|^2 grad_perp phi ) + n0 phi / (Z epsilon^2 T0) = (1/epsilon) int (f - f0) B*_par dv_par dmu,
where the first term is the ion polarization density and the second the adiabatic electron response (control
variate: only the deviation f - f0 from the background enters). In a torus, the grad-B and curvature drifts of
the ions on the outboard side (bad curvature) together with a steep ion temperature gradient release the free
energy of the profiles: a temperature perturbation causes a vertical charge separation through the
temperature-dependent drifts, whose E x B flow carries hot ions outwards and amplifies the perturbation once the
normalized gradient R/L_T exceeds a threshold. The unstable modes balloon at the outboard side and are nearly
aligned with the field lines, m approximately n q(r0).

Setup (global Cyclone base case of Lapillonne et al.). Ad hoc tokamak (AdhocTorus) with major radius R0 = 1,
minor radius a = 0.36, B0 = 1 and safety factor q(r) = 0.86 - 0.16 r/a + 2.52 r^2/a^2 (q = 1.41 and magnetic shear
(r/q) dq/dr = 0.84 at r0 = a/2). Density and temperature profiles
    n(r) = exp(-kappa_n a Delta_n tanh((r - r0)/(Delta_n a))),
    T(r) = exp(-kappa_T a Delta_T tanh((r - r0)/(Delta_T a))),
i.e. R0/L_n = kappa_n = 2.23 and R0/L_T = kappa_T = 6.96 at r0, with widths Delta_n = Delta_T = 0.3, and
T_e = T_i. The base units (kBT = 0.1916) give epsilon = 1.4142e-3, i.e. rho* = rho_s/a = 1/180. The domain is a
HollowTorus (inner radius 0.01, straight field line angle) with a 1/19 toroidal period: the logical mode n = 1
is the toroidal mode n = 19. The initial density perturbation of amplitude 1e-6 is the (m, n) = (27, 19) mode in
the straight field line angle, localized around r0 (Gaussian width 0.02), with k_theta rho_s = n q(r0)/r0 rho_s
of about 0.3, near the maximum of the ITG spectrum. Ions are loaded as a gyro-Maxwellian with the above
profiles (delta-f with control variate); markers leaving at the radial boundary are removed. The charge density
is filtered to the logical toroidal Fourier mode 1 (n = 19) in the Poisson solve. Time step dt = 0.001, end time
Tend = 0.01 (a short demonstration run; increase Tend for the full linear growth phase).

Expected result. The pproc fits an exponential to phi_integral, the poloidally and toroidally averaged phi^2 at
mid radius, which grows at twice the linear growth rate of the ITG mode in the linear phase; it also shows the
potential, the charge density and the perturbed distribution in the poloidal plane at the first, middle and last
saved time. In the benchmark the linear phase is followed by the saturation into ITG turbulence.

References:
- X. Lapillonne, B. F. McMillan, T. Goerler, S. Brunner, T. Dannert, F. Jenko, F. Merz and L. Villard,
  "Nonlinear quasisteady state benchmark of global gyrokinetic codes", Phys. Plasmas 17, 112321 (2010).
- A. M. Dimits et al., "Comparisons and physics basis of tokamak transport models and turbulence simulations",
  Phys. Plasmas 7, 969 (2000).
- A. J. Brizard and T. S. Hahm, "Foundations of nonlinear gyrokinetic theory", Rev. Mod. Phys. 79, 421 (2007).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid.
# `--test-run` selects a small grid, few markers and a short run (three saved time steps) for the tests
# (test_examples.py). Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[32, 135, 5], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 27, 4]), few markers and three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 27, 4]

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
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import DriftKineticElectrostaticAdiabatic
from struphy.pic.accumulation.filter import FilterParameters

# ---------------------
# Instance of the model
# ---------------------


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
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping
dt = 0.001
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 0.01, split_algo="LieTrotter")

a, r_min, R0 = 0.36, 0.01, 1.0
num_elements = tuple(args.num_elements)
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

ppc = 5 if args.test_run else 50  # run with 200 minimum
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0, 0, 4, 4))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(
    boxes_per_dim=(4, 4, 2) if args.test_run else (12, 12, 6), do_sort=True, sorting_frequency=5
)

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
    n=(n_init, None),
    vth_para=(vth_init, None),
    vth_perp=(vth_init, None),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_background(background)
# background.plot_density_profile("e1", "e2", domain=domain, plot_3D=True, in_physical=True)
# background.plot_density_profile("e1", "v1", domain=domain)
# background.plot_density_profile("e1", "v2", domain=domain, use_mu=True, equil=equil)

from struphy.initial.base import GenericPerturbation

perturbation = GenericPerturbation(pert_func, given_in_basis="0")
init = maxwellians.GyroMaxwellian2D(
    n=(n_init, perturbation),
    vth_para=(vth_init, None),
    vth_perp=(vth_init, None),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run(profiling_activated=True)
