"""MHD waves in a homogeneous slab with the nonlinear variational ideal MHD model (ViscoResistiveMHD without
viscosity and resistivity): Alfvén, slow and fast magnetosonic speeds from the space-time spectrum."""

# -----------------------------
# Description of the simulation
# -----------------------------

name = "Slab MHD dispersion (ViscoResistiveMHD)"
description = """
Small-amplitude MHD waves in a homogeneous slab, computed with the nonlinear variational MHD model ViscoResistiveMHD
(density, entropy, velocity and magnetic field, here without viscosity and resistivity, i.e. ideal MHD).

Physics. Linearized around a homogeneous state (n0, p0, B0), ideal MHD supports three waves with wave vector k.
For k along z and B0 = (0, B0y, B0z), their phase speeds are
    Alfvén:     v_A |cos theta|,
    slow/fast:  v^2 = (c_s^2 + v_A^2) / 2 (1 -/+ sqrt(1 - delta)),   delta = 4 c_s^2 v_A^2 cos^2 theta / (c_s^2 + v_A^2)^2,
with v_A^2 = |B0|^2 / n0, c_s^2 = gamma p0 / n0 and cos theta = B0z / |B0|. The Alfvén wave is polarized
along x (perpendicular to k and B0), the slow and fast waves compress the plasma and show up in the pressure.

Setup. Cuboid of length Lz = 60 in z (one element in x and y), B0 = (0, 1, 1), plasma beta = 3, n0 = 0.7,
gamma = 5/3 (HomogenSlab), 64 cubic elements in z, periodic. All velocity components start from noise of amplitude
1e-3, which excites all three waves at all resolved wave numbers. Time step dt = 0.15 up to Tend = 180
(1200 steps), with the default (Newton/Picard) options of the variational propagators.

Expected result. The space-time Fourier transform of u_x (Alfvén) and of the pressure perturbation (slow and fast)
shows the three branches omega = v k. The pproc fits the speeds and prints them with the exact values
v_A |cos theta| = 1.195, v_slow = 1.074, v_fast = 2.974; a run gave 1.187, 1.112 and 2.916 (within a few percent,
about one frequency bin 2 pi / Tend; the fast branch bends down at large k by numerical dispersion).
The run takes about 10 minutes on one process.
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the number of elements in z.
# `--test-run` selects a small grid and a short run (three saved time steps) for the tests (test_examples.py).
# Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, default=64, help="Number of elements in z.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (16 elements) and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = 16

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.models import ViscoResistiveMHD

# ---------------------
# Physical parameters (also used by the pproc)
# ---------------------

B0x, B0y, B0z = 0.0, 1.0, 1.0
beta = 3.0
n0 = 0.7
gamma = 5.0 / 3.0
Lz = 60.0

# ---------------------
# Instance of the model
# ---------------------

base_units = BaseUnits()
model = ViscoResistiveMHD(base_units=base_units, with_viscosity=False, with_resistivity=False)

model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.entropy.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# --test-run: the initial, a middle and a final saved time step
dt = 0.15
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 180.0)

domain = domains.Cuboid(r3=Lz)

equil = equils.HomogenSlab(B0x=B0x, B0y=B0y, B0z=B0z, beta=beta, n0=n0)

grid = grids.TensorProductGrid(num_elements=(1, 1, args.num_elements))

derham_opts = DerhamOptions(degree=(1, 1, 3))

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

model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")
model.propagators.variat_mom.options = model.propagators.variat_mom.Options()
model.propagators.variat_ent.options = model.propagators.variat_ent.Options()
model.propagators.variat_mag.options = model.propagators.variat_mag.Options()

# ------------------
# Initial conditions
# ------------------

# full-f model: the backgrounds are the total initial fields, plus noise in the velocity
model.mhd.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
model.mhd.entropy.add_background(FieldsBackground(type="FluidEquilibrium", variable="s3_monoatomic"))
model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

for component in range(3):
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=1e-3, comp=component, seed=123))

if __name__ == "__main__":
    sim.run()
