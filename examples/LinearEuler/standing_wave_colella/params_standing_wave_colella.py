"""Standing sound wave in the curvilinear, periodic Colella domain with the LinearEuler model, compared with the exact
solution (given in physical coordinates, so the mapping only enters the discretization)."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Standing sound wave in the Colella domain (LinearEuler)"
description = """
Standing sound wave in a uniform fluid at rest, in 2D (x, y), with the LinearEuler model: the linearized
compressible Euler equations around a static equilibrium.

Physics. With rho0, p0 constant, the pressure perturbation satisfies the wave equation d^2 p / dt^2 = c_s^2 Lap p with
the sound speed c_s = sqrt(gamma p0 / rho0). In the periodic unit square, the standing wave

    p(x, y, t) = A cos(2 pi x) cos(2 pi y) cos(omega t),
    u(x, y, t) = A / (rho0 omega) sin(omega t) 2 pi (sin(2 pi x) cos(2 pi y), cos(2 pi x) sin(2 pi y), 0),

with omega = 2 sqrt(2) pi c_s, is an exact solution: the energy oscillates between the kinetic energy
en_U = en_tot sin^2(omega t) and the thermal energy en_thermal = en_tot cos^2(omega t), and en_tot is conserved.

Setup. Colella domain of unit size, x = eta1 + alpha sin(2 pi eta1) sin(2 pi eta2), y = eta2 + alpha sin(2 pi eta2)
sin(2 pi eta1) with alpha = 0.1: a periodic, curvilinear distortion of the unit square, so the metric of the mapping
enters all operators while the exact solution is the one of the square. Uniform equilibrium (HomogenSlab, the magnetic
field is not used) with rho0 = 1 and p0 = 0.6, so c_s = 1 and the period is 1/sqrt(2). Initial pressure perturbation
A cos(2 pi x) cos(2 pi y) in physical coordinates (A = 0.01), zero velocity. Splines of degree 3 on 16 x 16 cells,
dt = 0.01 up to t = 1.5 (about two periods).

Expected result. The pproc script compares the pressure with the exact solution at the first, middle and last saved
time; the relative L2 error at the last time is about 3e-3 (asserted below 1%). The energies follow the sin^2 and
cos^2 laws and en_tot is conserved to the solver tolerances (about 4e-7, asserted below 1e-6).

References:
- L. D. Landau and E. M. Lifshitz, "Fluid Mechanics", Pergamon (1987), chapter VIII (sound).
- P. Colella, M. R. Dorr, J. A. F. Hittinger and D. F. Martin, "High-order, finite-volume methods in mapped
  coordinates", J. Comput. Phys. 230 (2011) 2952 (the Colella mapping).
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[16, 16, 1], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 8, 1]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 8, 1]

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
from struphy.models import LinearEuler

# Units (the thermal velocity scale needs the unit of the temperature)
base_units = BaseUnits(kBT=1.0)

# Model instance
model = LinearEuler(base_units=base_units)

# List all variables and decide whether to save their data
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=2, sim_folder=args.sim_folder)

# Time stepping
dt = 0.01
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 1.5)

# Geometry: periodic, curvilinear distortion of the unit square
domain = domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0)

# Fluid equilibrium (the magnetic field is not used): rho0 = 1, p0 = beta |B0|^2 / 2 = 0.6, c_s^2 = gamma p0 / rho0 = 1
equil = equils.HomogenSlab(B0z=1.0, beta=1.2, n0=1.0)

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options (periodic)
derham_opts = DerhamOptions(degree=(3, 3, 1))

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

model.propagators.sound.options = model.propagators.sound.Options()
model.propagators.bg_pressure.options = model.propagators.bg_pressure.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# standing wave: pressure A cos(2 pi x) cos(2 pi y) in physical coordinates, zero velocity
model.mhd.pressure.add_perturbation(
    perturbation=perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(1e-2,), Lx=1.0, Ly=1.0, given_in_basis="physical")
)

if __name__ == "__main__":
    sim.run()
