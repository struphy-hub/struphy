"""Bump-on-tail instability with the VlasovAmpereOneSpecies model: a bump in the high-velocity tail of the
electron distribution drives Langmuir waves that grow and saturate by particle trapping."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov-Ampere bump-on-tail instability"
description = """
Bump-on-tail instability with the VlasovAmpereOneSpecies model: electrostatic 1D-3V kinetic electrons (PIC) in a
periodic box with a neutralizing ion background. A small, fast electron population (the "bump") on the tail of the
bulk distribution drives Langmuir (electron plasma) waves unstable; the waves grow, trap the resonant electrons
and saturate.

Physics. With alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency 1/omega_p) and
no background magnetic field (with_B0=False) the model solves
    df/dt + v . grad_x f - E . grad_v f = 0,
    dE/dt = int v f d^3v       (Ampere's law; the current density of the electrons is -int v f d^3v),
where the initial field follows from Poisson's equation div E = n0 - n with n = int f d^3v and the ion density
n0 = int f0 d^3v of the background f0; Gauss's law is then kept by the Ampere equation. Small perturbations
exp(i (k x - omega t)) obey the kinetic dispersion relation
    1 + sum_s n_s / (k vth_s)^2 [1 + xi_s Z(xi_s)] = 0,   xi_s = (omega - k u_s) / (sqrt(2) k vth_s),
with the plasma dispersion function Z and the density n_s, drift u_s and thermal speed vth_s of each Maxwellian.
For a single Maxwellian the waves are Landau damped, because there are more slow than fast resonant electrons.
Between the bulk and the bump the slope of f has the opposite sign: waves with their phase velocity omega/k in
this region take more energy from the resonant electrons than they give (inverse Landau damping) and grow. Once
the bounce frequency of the electrons trapped in the wave potential is comparable to the growth rate, trapping
flattens the bump (phase-space vortices, plateau) and the wave saturates and oscillates.

Setup. Periodic box of length L = 62.83 (about 20 pi, so k = 2 pi / L = 0.1), 32 cells and cubic splines in x.
The electrons are the sum of two Maxwellians: a bulk with n = 0.9, drift u1 = 3 and vth = 1, and a bump with
n = 0.1, drift u1 = -4.5 (7.5 bulk thermal speeds away from the bulk) and vth1 = 0.5. Only the bump density is
perturbed, by 0.05 cos(k x). Control-variate (delta-f) weights with 1000 markers per cell, drawn from a Maxwellian
with vth1 = 3 that covers both populations; time step dt = 0.1 up to t = 60 (Lie-Trotter splitting).

Expected result. The numerical solution of the dispersion relation above for this distribution gives growth rates
of about 0.20 for k = 0.1 and 0.22 for k = 0.2 (larger k are stable); the pproc does not compare with them. It
shows the initial velocity distribution with the bump, the electric energy on a log scale (exponential growth,
then saturation with trapping oscillations), the distribution function in the (x, v1) plane over time (vortices
of trapped electrons at the bump) and the fields along x at the first, middle and last saved time.

References:
- L. D. Landau, "On the vibrations of the electronic plasma", J. Phys. USSR 10, 25 (1946).
- T. O'Neil, "Collisionless damping of nonlinear plasma oscillations", Phys. Fluids 8, 2255 (1965).
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
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

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
dt = 0.1
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 60.0, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid(r1=62.83)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

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

loading_params = LoadingParameters(ppc=50 if args.test_run else 1000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0), seed=1234)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)

binplot_1 = BinningPlot(
    slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-10.0, 10.0))
)  # for initial velocity distribution
binplot_2 = BinningPlot(
    slice="v1", n_bins=128, ranges=(-10.0, 10.0)
)  # for progression of velocity and space distribution
saving_params = SavingParameters(binning_plots=(binplot_1, binplot_2))

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
if model.with_B0:
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
maxwellian_1 = maxwellians.Maxwellian3D(n=(9 / 10, None), u1=(3.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(1 / 10, None), u1=(-4.5, None), vth1=(0.5, None))
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps=(0.05,), ls=(1,))
init1 = maxwellians.Maxwellian3D(n=(9 / 10, None), u1=(3.0, None))
init2 = maxwellians.Maxwellian3D(n=(1 / 10, perturbation), u1=(-4.5, None), vth1=(0.5, None))
init = init1 + init2
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
