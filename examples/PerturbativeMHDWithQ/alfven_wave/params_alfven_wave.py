"""Large-amplitude circularly polarized Alfvén wave in 1D with the PerturbativeMHDWithQ model: an exact nonlinear
solution of ideal MHD that propagates undistorted, used to check that the nonlinear steps neither damp nor distort it."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Circularly polarized Alfven wave (PerturbativeMHDWithQ)"
description = """
Circularly polarized Alfven wave of large amplitude in 1D, with the PerturbativeMHDWithQ model: the ideal MHD equations
for the perturbations of a uniform, static equilibrium, with the square root of the pressure, q = sqrt(p), as
thermodynamic variable.

Physics. With B0 = B0 e_x, the magnetic field perturbation B = A (0, cos 2 pi (x - v_A t), sin 2 pi (x - v_A t)) and
the velocity u = -B / sqrt(rho0) are an exact solution of the nonlinear ideal MHD equations for any amplitude A: the
magnitude |B0 + B| is constant, so there is no magnetic pressure gradient, the magnetic tension is balanced by the
inertia, and the density and pressure stay uniform. The wave travels with v_A = B0 / sqrt(rho0) without change of
shape. The nonlinear terms ((curl B) x B, curl (u x B), u . grad u) of the continuous equations vanish for this
solution; the run checks that their discrete versions do not distort or damp it.

Setup. Periodic unit interval (Cuboid, one element along y and z), uniform equilibrium B0 = e_x, rho0 = 1, beta = 0.5
(HomogenSlab), so v_A = 1. A = 0.5 B0. Splines of degree 3 on 32 cells, dt = 0.02 up to t = 4 (four periods), Strang
splitting, default options (weights of the nonlinear steps at t^n).

Expected result. The pproc script plots B_y and B_z at the saved times next to the exact wave and the energies. The
maximal error of the magnetic field relative to A at t = 4 is about 1e-2 (asserted below 3e-2), the amplitude is kept
to about 2e-6 (asserted below 1e-3), and the energy of the perturbation en_tot is conserved to about 1e-14 (asserted
below 1e-8; in this uniform equilibrium every step conserves it).

References:
- J. P. Goedbloed and S. Poedts, "Principles of Magnetohydrodynamics", Cambridge University Press (2004),
  section 5.4 (nonlinear Alfven waves).
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[32, 1, 1], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [16, 1, 1]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [16, 1, 1]

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
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
from struphy.models import PerturbativeMHDWithQ

# Units
base_units = BaseUnits()

# Model instance
model = PerturbativeMHDWithQ(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.sqrt_p.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=25, sim_folder=args.sim_folder)

# Time stepping
dt = 0.02
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 4.0, split_algo="Strang")

# Geometry: periodic unit interval
domain = domains.Cuboid()

# Equilibrium: B0 = e_x, rho0 = 1 (v_A = 1), p0 = beta |B0|^2 / 2
equil = equils.HomogenSlab(B0x=1.0, B0y=0.0, B0z=0.0, beta=0.5, n0=1.0)

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options (periodic)
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

# ------------------
# Propagator options
# ------------------

model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
model.propagators.nonlin_alf.options = model.propagators.nonlin_alf.Options()
model.propagators.bg_current.options = model.propagators.bg_current.Options()
model.propagators.bg_q.options = model.propagators.bg_q.Options()
model.propagators.sound.options = model.propagators.sound.Options()
model.propagators.nonlin_sound.options = model.propagators.nonlin_sound.Options()
model.propagators.mom_adv.options = model.propagators.mom_adv.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

AMP = 0.5
model.em_fields.b_field.add_perturbation(
    perturbation=perturbations.ModesCos(ls=(1,), amps=(AMP,), comp=1, given_in_basis="physical")
)
model.em_fields.b_field.add_perturbation(
    perturbation=perturbations.ModesSin(ls=(1,), amps=(AMP,), comp=2, given_in_basis="physical")
)
model.mhd.velocity.add_perturbation(
    perturbation=perturbations.ModesCos(ls=(1,), amps=(-AMP,), comp=1, given_in_basis="physical")
)
model.mhd.velocity.add_perturbation(
    perturbation=perturbations.ModesSin(ls=(1,), amps=(-AMP,), comp=2, given_in_basis="physical")
)

if __name__ == "__main__":
    sim.run()
