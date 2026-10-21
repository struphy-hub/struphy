"""Localized pulse in a homogeneous 2D slab with the LinearMHD models: the pulse splits into fast, slow and shear
Alfven waves, whose fronts are known analytically (Friedrichs diagram). A cheap 2D comparison of LinearMHD,
LinearMHDCurlCurl and LinearMHDCurlCurlwithProj."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "2D slab pulse (LinearMHD)"
description = """
Localized pulse in a homogeneous magnetized plasma, in 2D (x, y), with the LinearMHD models: linear ideal MHD around
a static, uniform equilibrium. A Gaussian pressure pulse launches the compressible fast and slow magnetosonic waves,
a Gaussian pulse of the out-of-plane velocity u_z the shear Alfven wave. Run the same file with --model LinearMHD
(magnetic field in H(div)), --model LinearMHDCurlCurl (magnetic field in H(curl), curl-curl solve of the shear
Alfven step preconditioned by Hiptmair-Xu, numerically damped) and --model LinearMHDCurlCurlwithProj (H(curl),
energy conserving shear Alfven step), each with its own --sim-folder, and compare the runs with the pproc script
(several output folders as arguments). On 2 MPI ranks, the 140 time steps take about 75 s with LinearMHD and about
40 s with each of the curl-curl models.

Physics. In a uniform plasma with magnetic field B0, number density n0 and pressure p0, a plane wave with wave
vector k at the angle phi to B0 has one of the phase velocities
    v_A(phi) = v_A |cos phi|,
    v_f,s(phi)^2 = (v_A^2 + c_s^2) / 2 * (1 +- sqrt(1 - 4 v_A^2 c_s^2 cos^2 phi / (v_A^2 + c_s^2)^2)),
with the Alfven velocity v_A = |B0| / sqrt(n0) and the sound speed c_s = sqrt(gamma p0 / n0), gamma = 5/3. A
localized pulse spreads with the group velocities: its fronts are the Friedrichs diagrams of the fast and slow
waves, while the shear Alfven wave is not dispersive along B0 (group velocity +- v_A b0 for every k), so the u_z pulse
splits into two copies of itself, which move along B0 with the velocity v_A:
    u_z(x, t) = (u_z0(x - v_A t b0) + u_z0(x + v_A t b0)) / 2.

Setup. Periodic square [-2, 2]^2 (Cuboid, one element along z), uniform B0 = (cos 30 deg, sin 30 deg, 0) (tilted
with respect to the grid), n0 = 1 and beta = 0.432, such that c_s = 0.6 v_A. Gaussian pulses of width 0.12 at the
origin, in the pressure and in u_z (physical components, amplitude 1e-3).

Expected result. At t = 1.4, before the fronts reach the periodic boundary: the fast front perpendicular to B0 at
the distance sqrt(v_A^2 + c_s^2) t = 1.63, the slow cusps and the fast front along B0 at c_s t = 0.84 and v_A t = 1.4,
and the two u_z pulses at the distance v_A t = 1.4 along +- B0. The pproc script overlays the analytic fronts on the
pressure and u_z snapshots and plots the energies. In the periodic, homogeneous slab the solution is known exactly
(each Fourier mode evolves with the matrix exponential of its linear system): the pproc script measures the relative
L2 errors of the pressure and of u_z with respect to it at t = 1.4 (default grid: about 1.3% and 3.7% with
LinearMHD, 0.5% and 1% with the curl-curl models) and asserts them below 3% and 6%, and asserts the conservation of the
perturbed energy en_tot = en_U + en_B + en_thermal within 1e-4 for LinearMHD and LinearMHDCurlCurlwithProj
(LinearMHDCurlCurl changes it by about 2e-5 here). In general, linear MHD conserves the kinetic energy plus the
potential energy delta W, which contains terms with the equilibrium current and pressure gradient and differs from
en_B + en_thermal, so en_tot is not conserved. In this uniform equilibrium (J0 = 0, grad p0 = 0) with periodic
boundaries, delta W = en_B + en_thermal, so en_tot is conserved, and both splitting steps conserve it up to the
solver tolerances.

References:
- J. P. Goedbloed and S. Poedts, "Principles of Magnetohydrodynamics", Cambridge University Press (2004),
  chapter 5 (waves in homogeneous plasmas, group velocity diagrams).
"""

import argparse
import logging
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid.
# `--test-run` selects a small grid and a short run (three saved time steps) for the tests (test_examples.py).
# Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by another script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_data", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 64, 1], help="Number of elements per direction.")
parser.add_argument(
    "--model",
    default="LinearMHD",
    choices=["LinearMHD", "LinearMHDCurlCurl", "LinearMHDCurlCurlwithProj"],
    help="LinearMHD (default, magnetic field in H(div), implicit solve for the velocity), LinearMHDCurlCurl "
    "(magnetic field in H(curl), implicit curl-curl solve for the magnetic field, numerically damped) or "
    "LinearMHDCurlCurlwithProj (as LinearMHDCurlCurl, energy conserving shear Alfven step).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [16, 16, 1]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.model != "LinearMHD":
    name = f"2D slab pulse ({args.model})"
if args.test_run:
    args.num_elements = [16, 16, 1]

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------

import numpy as np

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    ProfilingOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import LinearMHD, LinearMHDCurlCurl, LinearMHDCurlCurlwithProj

# Units
base_units = BaseUnits()

# Model instance
model_class = {
    "LinearMHD": LinearMHD,
    "LinearMHDCurlCurl": LinearMHDCurlCurl,
    "LinearMHDCurlCurlwithProj": LinearMHDCurlCurlwithProj,
}[args.model]
model = model_class(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=10, sim_folder=args.sim_folder)

# Time stepping
dt = 0.01
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 1.4)

# Geometry
domain = domains.Cuboid(l1=-2.0, r1=2.0, l2=-2.0, r2=2.0, l3=0.0, r3=1.0)

# Fluid equilibrium (can be used as part of initial conditions): |B0| = 1, B0 tilted by 30 degrees to the x-axis,
# c_s^2 = gamma beta |B0|^2 / 2 / n0 = 0.36
theta = np.pi / 6
equil = equils.HomogenSlab(B0x=np.cos(theta), B0y=np.sin(theta), B0z=0.0, beta=0.432, n0=1.0)

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options (periodic)
derham_opts = DerhamOptions(degree=(3, 3, 1))

# Profiling options
profiling_opts = ProfilingOptions(use_line_profiler=False)

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
    profiling_opts=profiling_opts,
)

# ------------------
# Propagator options
# ------------------

# the default preconditioners: StiffnessPreconditioner (LinearMHD), HiptmairXu (LinearMHDCurlCurl) and the mass
# matrix preconditioner (LinearMHDCurlCurlwithProj)
model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Gaussian pulses at the origin (physical coordinates and components): the pressure launches the fast and slow
# waves, the out-of-plane velocity u_z the shear Alfven wave
pulse = dict(amp=1e-3, center=(0.0, 0.0, 0.0), sigma=(0.12, 0.12, None), given_in_basis="physical")
model.mhd.pressure.add_perturbation(perturbation=perturbations.Gaussian(**pulse))
model.mhd.velocity.add_perturbation(perturbation=perturbations.Gaussian(**pulse, comp=2))

if __name__ == "__main__":
    sim.run(profiling_activated=True)
