"""Two-stream instability with the VlasovAmpereOneSpecies model: two counter-streaming electron beams in a 1D
periodic box become unstable; the nonlinear phase shows particle trapping and phase-space vortices."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov-Ampere two-stream instability"
description = """
Two-stream instability with the VlasovAmpereOneSpecies model: electrostatic 1D-3V kinetic electrons (PIC) in a
periodic box with a neutralizing ion background. Two counter-streaming electron beams of equal density are
unstable: a small density perturbation grows exponentially, until the electrons are trapped in the potential wells
of the wave and form a phase-space vortex.

Physics. With alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency 1/omega_p of
the total density) and no background magnetic field (with_B0=False) the model solves
    df/dt + v . grad_x f - E . grad_v f = 0,
    dE/dt = int v f d^3v       (Ampere's law; the current density of the electrons is -int v f d^3v),
where the initial field follows from Poisson's equation div E = n0 - n with n = int f d^3v and the ion density
n0 = int f0 d^3v of the background f0; Gauss's law is then kept by the Ampere equation. Small perturbations
exp(i (k x - omega t)) of two Maxwellian beams with densities n_s, drifts u_s and thermal speed vth obey
    1 + sum_s n_s / (k vth)^2 [1 + xi_s Z(xi_s)] = 0,   xi_s = (omega - k u_s) / (sqrt(2) k vth),
with the plasma dispersion function Z. Each beam alone carries two space-charge (Langmuir) waves with the
Doppler-shifted frequencies omega = k u_s +- sqrt(n_s); where waves of the two beams have nearly equal frequencies
they couple and grow at the expense of the kinetic energy of the beams. For cold beams (vth = 0) with n_s = 1/2 and
u_s = +-u0 the symmetric mode is purely growing, omega^2 = k^2 u0^2 + 1/2 - sqrt(2 k^2 u0^2 + 1/4) < 0, for
k u0 < 1; the thermal spread reduces the growth rate. The growth ends when the bounce frequency of the trapped
electrons reaches the growth rate: the beams roll up into a vortex in the (x, v1) plane, and the electric energy
saturates and oscillates.

Setup. Periodic box of length L = 31.42 (about 10 pi, so k = 2 pi / L = 0.2), 32 cells and cubic splines in x.
The electrons are two Maxwellians with n = 0.5 each, drifts u1 = +3 and -3 and vth = 1 (k u0 = 0.6); the density
of each beam is perturbed by 0.001 cos(k x). Control-variate (delta-f) weights with 1000 markers per cell, drawn
from a Maxwellian with vth1 = 3 that covers both beams; time step dt = 0.1 up to t = 50 (Lie-Trotter splitting).

Expected result. The kinetic dispersion relation above gives, for k = 0.2, a purely growing mode (omega = 0 + i
gamma) with gamma = 0.2845 (cold beams: 0.353); the electric field grows as exp(gamma t) and the electric energy
as exp(2 gamma t), followed by the nonlinear saturation. The pproc plots the electric energy on a log scale
together with an analytical reference curve built from 0.2845, the distribution function in the (x, v1) plane
over time (the roll-up of the beams into a vortex) and the fields along x at the first, middle and last saved
time.

References:
- E. Sonnendruecker, J. Roche, P. Bertrand and A. Ghizzo, "The semi-Lagrangian method for the numerical
  resolution of the Vlasov equation", J. Comput. Phys. 149, 201 (1999).
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
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 50.0, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid(r1=31.42)

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

loading_params = LoadingParameters(ppc=50 if args.test_run else 1000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)

binplot = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-10.0, 10.0)))
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
maxwellian_1 = maxwellians.Maxwellian3D(n=(0.5, None), u1=(3.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(0.5, None), u1=(-3.0, None))
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps=(0.001,), ls=(1,))
init1 = maxwellians.Maxwellian3D(n=(0.5, perturbation), u1=(3.0, None))
init2 = maxwellians.Maxwellian3D(n=(0.5, perturbation), u1=(-3.0, None))
init = init1 + init2
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
