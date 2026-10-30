"""Ion temperature gradient (ITG) instability with the DriftKineticElectrostaticAdiabatic model in a hollow
cylinder with a uniform axial magnetic field (4D drift-kinetic test case of the semi-Lagrangian literature,
e.g. Crouseilles et al. (2014)); also a template for setting up this model."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "ITG instability in a cylinder (DriftKineticElectrostaticAdiabatic)"
description = """
Ion temperature gradient (ITG) instability in a periodic cylinder with a uniform axial magnetic field, with
drift-kinetic ions and adiabatic electrons (DriftKineticElectrostaticAdiabatic). This is the classical
drift-kinetic test case in cylindrical geometry (a straightened torus); the file also serves as a template for
setting up simulations with this model.

Physics. The ion guiding-center distribution f(X, v_par, mu, t) is advected along the characteristics
    dX/dt     = v_par B*/B*_par + E* x b0 / B*_par,
    dv_par/dt = (1/epsilon) B*/B*_par . E*,
with E* = -grad phi - epsilon mu grad|B0|, B* = B0 + epsilon v_par curl b0 and B*_par = B* . b0. In the uniform
field B0 = e_z used here, this reduces to parallel streaming v_par along z, the E x B drift E x e_z and the
parallel acceleration by E_z. The electrons respond adiabatically, n_e = n0 (1 + e phi / T_e), and the
electrostatic potential follows from the quasi-neutrality (gyrokinetic Poisson) equation
    -div_perp( n0/|B0|^2 grad_perp phi ) + n0 phi / (Z epsilon^2 T0) = (1/epsilon) int (f - f0) B*_par dv_par dmu,
with the ion polarization density on the left and the control variate f0 (only f - f0 enters). Without
curvature, the drive is the slab ITG mechanism: a mode with finite parallel wave number k_par = 2 pi n / Lz couples
the parallel ion dynamics (ion sound) with the E x B advection of the radial density and temperature gradients;
when the temperature gradient is large compared to the density gradient (eta_i = L_n/L_T above a threshold of
order one, here kappa_T/kappa_n = 5) the mode is unstable.

Setup. HollowCylinder with r in [0.1, 14.5] and length Lz = 1506.759067 (= 2 pi x 239.8), uniform field B0 = 1
along z (HomogenSlab), epsilon = 1, T = 1 in the base units. Radial profiles centred at
rp = (a1 + a2)/2 = 7.3,
    T_i(r) = T_e(r) = exp(-kappa_T Delta_T tanh((r - rp)/Delta_T)),   kappa_T = 0.27586, Delta_T = 1.45,
    n0(r) = C_n exp(-kappa_n Delta_n tanh((r - rp)/Delta_n)),         kappa_n = 0.055, Delta_n = Delta_T/2,
with C_n normalizing the mean density to one. The initial density perturbation of amplitude 1e-6 is the mode
cos(m theta + 2 pi n z/Lz) with (m, n) = (5, 1), localized around rp. Ions are loaded as a gyro-Maxwellian with
the above profiles (delta-f with control variate); markers leaving at the radial boundary are removed and phi
vanishes at r = 0.1 and r = 14.5. Grid of 32 x 50 x 10 elements (default), time step dt = 5, end time Tend = 500.

Expected result. The pproc fits an exponential to phi_integral, the theta- and z-averaged phi^2 at mid radius,
which grows at twice the linear growth rate of the (5, 1) mode; it also shows the potential, the charge density
and the (perturbed) distribution in the poloidal plane at the first, middle and last saved time. In the
literature, the linear phase is followed by the nonlinear saturation of the ITG modes.

References:
- N. Crouseilles, P. Glanc, S. A. Hirstoaga, E. Madaule, M. Mehrenberger and J. Petri, "A new fully
  two-dimensional conservative semi-Lagrangian method: applications on polar grids, from diocotron instability to
  ITG turbulence", Eur. Phys. J. D 68, 252 (2014).
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[32, 50, 10], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 10, 4]), few markers and three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 10, 4]

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
from struphy.models import DriftKineticElectrostaticAdiabatic
from struphy.propagators import implicit_diffusion

# ------------------
# Import Struphy API
# ------------------


# ---------------------
# Instance of the model
# ---------------------


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
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping
dt = 5.0
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 500.0, split_algo="LieTrotter")

# Geometry
a1, a2, Lz = 0.1, 14.5, 1506.759067
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)

# Grid
num_element = tuple(args.num_elements)
grid = grids.TensorProductGrid(num_elements=num_element, mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Simulation object
from maybempi import MPI

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

ppc = 5 if args.test_run else 50  # run with 200 minimum
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0.0, 0.0, 2.0, 2.0))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(
    do_sort=True,
    boxes_per_dim=(4, 4, 2) if args.test_run else (12, 12, 6),
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
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo="explicit", evaluate_e_field=True)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
    algo="explicit", evaluate_e_field=True
)

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
        eta1 = etas[0][0]
    else:
        eta1 = etas[0]
    r = a1 + (a2 - a1) * eta1
    return n0(r)


def pert_func(*etas):
    if len(etas) == 1:
        e1, e2, e3 = etas[0][0], etas[0][1], etas[0][2]
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
        eta1 = etas[0][0]
    else:
        eta1 = etas[0]
    r = a1 + (a2 - a1) * eta1
    return xp.sqrt(Ti(r))


def vth_e(*etas):
    if len(etas) == 1:
        eta1 = etas[0][0]
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
