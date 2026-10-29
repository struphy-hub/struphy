"""Strong (nonlinear) Landau damping with the VlasovAmpereOneSpecies model: a large density perturbation in a
1D periodic box traps particles in the potential wells, forming phase-space vortices."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov-Ampere strong Landau damping"
description = """
Strong (nonlinear) Landau damping with the VlasovAmpereOneSpecies model: electrostatic 1D-3V kinetic electrons
(PIC) in a periodic box with a neutralizing ion background, with a large (50 %) density perturbation of a
Maxwellian. The resulting electron plasma wave is first Landau damped, but the electrons trapped in its potential
wells stop the damping: the field amplitude grows again and oscillates, and phase-space vortices form.

Physics. With alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency 1/omega_p;
with vth = 1 lengths are in Debye lengths) and no background magnetic field (equil = None) the model solves
    df/dt + v . grad_x f - E . grad_v f = 0,
    dE/dt = int v f d^3v       (Ampere's law; the current density of the electrons is -int v f d^3v),
where the initial field follows from Poisson's equation div E = n0 - n with n = int f d^3v and the ion density
n0 = int f0 d^3v of the background f0; Gauss's law is then kept by the Ampere equation. For small amplitudes the
wave is Landau damped (see the weak Landau damping example): the resonant electrons with velocities close to the
phase velocity omega/k exchange energy with the wave, and the perturbation of f mixes in phase (filamentation in
velocity space). At large amplitude the electrons near the phase velocity are trapped in the potential wells and
bounce with the frequency omega_b = sqrt(k |E|); when 1/omega_b is shorter than the damping time, linear theory
breaks down. The trapped electrons stop the damping, the electric energy grows again after an initial decay and
then oscillates with the bounce period, and the distribution function forms a vortex (a hole) around the phase
velocity, a BGK-like nonlinear state.

Setup. Periodic box of length L = 12.56 (about 4 pi, so k = 2 pi / L = 0.5), 32 cells and linear splines (the
default degree 1) in x. Maxwellian electrons with vth = 1 and the density n = 1 + 0.5 cos(k x). Control-variate
(delta-f) weights with 1000 markers per cell; time step dt = 0.05 up to t = 75 (Lie-Trotter splitting).

Expected result. The pproc shows the electric energy on a log scale (a strong initial decay with oscillations at
twice the wave frequency, omega about 1.4, then a regrowth and nonlinear oscillations), the distribution function
in the (x, v1) plane over time (filamentation and trapped-particle vortices) and the fields along x at the first,
middle and last saved time. There is no analytical curve to compare with; this is a standard benchmark of Vlasov codes.

References:
- L. D. Landau, "On the vibrations of the electronic plasma", J. Phys. USSR 10, 25 (1946).
- T. O'Neil, "Collisionless damping of nonlinear plasma oscillations", Phys. Fluids 8, 2255 (1965).
- E. Sonnendruecker, J. Roche, P. Bertrand and A. Ghizzo, "The semi-Lagrangian method for the numerical
  resolution of the Vlasov equation", J. Comput. Phys. 149, 201 (1999).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder. `--test-run` selects few markers and a short run (three
# saved time steps) for the tests (test_examples.py). Unknown flags are ignored so that a driver can forward other
# parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_data", help="Name of the output folder.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers (ppc = 50) and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()

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
from struphy.models import VlasovAmpereOneSpecies

# ---------------------
# Instance of the model
# ---------------------


# Units
base_units = BaseUnits()

# Model instance
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder)

# Time stepping
dt = 0.05
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 75.0, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid(r1=12.56)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))

# Derham options
derham_opts = DerhamOptions()

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

loading_params = LoadingParameters(ppc=50 if args.test_run else 1000)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)

binplot = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-5.0, 5.0)))
saving_params = SavingParameters(binning_plots=(binplot,))

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=0.4,
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
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

# Background for kinetic species
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps=(0.5,), ls=(1,))
init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
