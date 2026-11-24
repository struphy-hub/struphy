"""Nonlinear simple sound wave in 1D with the PerturbativeEulerWithQ model: the wave steepens, and the solution is
compared with the exact solution (method of characteristics) before the shock."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Nonlinear simple sound wave (PerturbativeEulerWithQ)"
description = """
Right-going simple sound wave of finite amplitude in 1D, with the PerturbativeEulerWithQ model: the compressible Euler
equations for the perturbations of a uniform fluid at rest, with the square root of the pressure, q = sqrt(p), as
thermodynamic variable.

Physics. In a simple wave the Riemann invariant u - 2 c / (gamma - 1) is constant (c the sound speed), and the flow is
isentropic, rho = rho0 (c / c0)^(2 / (gamma - 1)), p = p0 (rho / rho0)^gamma. Every value of u travels with the speed
c0 + (gamma + 1) / 2 u, so the wave steepens: for u0(x) = A sin(2 pi x),

    u(x, t) = A sin(2 pi xi),   x = xi + (c0 + (gamma + 1) / 2 A sin(2 pi xi)) t,

until the shock at t_s = 1 / ((gamma + 1) / 2 * 2 pi A). The linear model would propagate the initial profile
unchanged with c0.

Setup. Periodic unit interval (Cuboid, one element along y and z), uniform equilibrium (HomogenSlab, the magnetic
field is not used) with rho0 = 1, p0 = 0.6, so c0 = 1. A = 0.05, so t_s = 2.39. The initial density and q are those
of the simple wave (given as functions). Splines of degree 3 on 128 cells, dt = 0.01 up to t = 1.5, with Strang
splitting and midpoint weights of the nonlinear steps (second order in time).

Expected result. The pproc script plots u at the saved times next to the exact solution and the energies. The maximal
error of u relative to A at t = 1.5 is about 1e-2 (asserted below 2e-2; it grows towards the shock, where the
profile is no longer resolved); the linear model, without the steepening, is off by about 0.4 there. en_tot, the
energy of the perturbation, changes by the splitting error only (about 2e-5; each step conserves its own energy
exactly).

References:
- L. D. Landau and E. M. Lifshitz, "Fluid Mechanics", Pergamon (1987), chapter X (simple waves).
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[128, 1, 1], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [32, 1, 1]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [32, 1, 1]

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
    Simulation,
    Time,
    domains,
    equils,
    grids,
)
from struphy.initial.base import GenericPerturbation

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import PerturbativeEulerWithQ

# Units (the thermal velocity scale needs the unit of the temperature)
base_units = BaseUnits(kBT=1.0)

# Model instance
model = PerturbativeEulerWithQ(base_units=base_units)

# List all variables and decide whether to save their data
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.sqrt_p.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=25, sim_folder=args.sim_folder)

# Time stepping (Strang splitting: second order with the midpoint weights below)
dt = 0.01
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 1.5, split_algo="Strang")

# Geometry: periodic unit interval
domain = domains.Cuboid()

# Fluid equilibrium (the magnetic field is not used): rho0 = 1, p0 = beta |B0|^2 / 2 = 0.6, c0^2 = gamma p0 / rho0 = 1
equil = equils.HomogenSlab(B0z=1.0, beta=1.2, n0=1.0)

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

model.propagators.bg_q.options = model.propagators.bg_q.Options()
model.propagators.sound.options = model.propagators.sound.Options()
model.propagators.nonlin_sound.options = model.propagators.nonlin_sound.Options(weights="midpoint")
model.propagators.mom_adv.options = model.propagators.mom_adv.Options(weights="midpoint")

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

GAMMA = 5 / 3
AMP = 0.05
RHO0, P0 = 1.0, 0.6
C0 = np.sqrt(GAMMA * P0 / RHO0)


def simple_wave(x):
    """Velocity, density and q of the right-going simple wave with u = AMP sin(2 pi x)."""
    u = AMP * np.sin(2 * np.pi * x)
    c = C0 + (GAMMA - 1) / 2 * u
    rho = RHO0 * (c / C0) ** (2 / (GAMMA - 1))
    return u, rho, np.sqrt(P0 * (rho / RHO0) ** GAMMA)


def u_init(x, y, z):
    return simple_wave(x)[0]


def rho_init(x, y, z):
    return simple_wave(x)[1] - RHO0


def q_init(x, y, z):
    return simple_wave(x)[2] - np.sqrt(P0)


model.mhd.velocity.add_perturbation(perturbation=GenericPerturbation(u_init, given_in_basis="physical", comp=0))
model.mhd.density.add_perturbation(perturbation=GenericPerturbation(rho_init, given_in_basis="physical"))
model.mhd.sqrt_p.add_perturbation(perturbation=GenericPerturbation(q_init, given_in_basis="physical"))

if __name__ == "__main__":
    sim.run()
