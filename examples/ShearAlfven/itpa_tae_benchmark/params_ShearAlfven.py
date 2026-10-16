"""Toroidal Alfven eigenmode (TAE) of the ITPA benchmark with the ShearAlfven model: an n = 6 shear Alfven wave
in the gap of the Alfven continuum at the crossing of the m = 10 and m = 11 branches (q = 1.75, r = 0.5) in a
large-aspect-ratio tokamak."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "ITPA TAE benchmark (ShearAlfven)"
description = """
Toroidal Alfven eigenmode (TAE) with the ShearAlfven model: the shear Alfven part of linear ideal MHD around a
static (zero-flow) equilibrium, without pressure and density perturbations (no coupling to the magnetosonic waves;
compare with the LinearMHD version of this benchmark). No energetic particles: the TAE is excited by the initial
condition, not driven. With --model ShearAlfvenCurlCurl, the magnetic field is in H(curl) and the implicit step
solves a curl-curl system for it, preconditioned by Hiptmair-Xu (same physics, same expected frequency).

Physics. Shear Alfven waves propagate along the magnetic field with the local frequency
    omega_A(r) = |k_par(r)| v_A(r),   v_A(r) = B(r) / sqrt(mu0 rho(r)),
where, for a mode with poloidal mode number m and toroidal mode number n in a torus of major radius R with safety
factor q(r), the parallel wave number is
    k_par(r) = (n q(r) - m) / (q(r) R).
The frequencies of two counter-propagating modes with the same n and poloidal mode numbers m and m + 1 cross where
k_par,m = -k_par,m+1, i.e. at the radius r_c with
    q(r_c) = (2m + 1) / (2n).
Toroidicity couples the two harmonics and opens a gap in the shear Alfven continuum at this crossing, in which the
toroidal Alfven eigenmode (TAE) exists with the frequency
    omega_TAE = n / (2m + 1) * v_A(r_c) / R.
In Struphy units, v_A = v' v_unit and R = R' x_unit, so omega_TAE = n / (2m + 1) * v'(r_c) / R' * v_unit / x_unit,
with the velocity unit v_unit (the Alfven velocity of the base units, set by the model) and the length unit x_unit
(base units).

Setup (ITPA TAE benchmark). Large-aspect-ratio tokamak with circular flux surfaces (AdhocTorus): major radius
R0 = 10, minor radius a = 1, toroidal field B0 = 3, safety factor q(r) = 1.71 + 0.16 r^2/a^2, number density
n(r) = 0.2 + 0.8 (1 - r^2/a^2) and on-axis beta 0.0018. Only the toroidal mode number n = 6 is computed, on a sixth
of the torus (HollowTorus with tor_period = 6 and a hole of radius 0.1): the logical mode number -1 along eta3 is
the physical n = 6. The velocity is initialized with the m = 10 and m = 11 harmonics, localized around r = 0.5
(given as 2-form components).

Expected result. With n = 6 and m = 10 the gap is at q(r_c) = 21/12 = 1.75, i.e. r_c = 0.5, where n = 0.8 and
|B| = 3.0 (flux-surface average), so v_A(r_c) = 3.36 v_unit and
    omega_TAE = 6/21 * 3.36/10 = 0.096 v_unit / x_unit,
a period of about 65 time units (about 33 kHz with the default base units x_unit = 1 m, B = 1 T, n = 1e20 m^-3 and
hydrogen, where v_unit = 2.2e6 m/s). The run up to t = 500 covers about 7.6 periods; the pproc script shows the
energies, the components of the fields and the radial profiles of the (m, n) harmonics.

References:
- W. W. Heidbrink, "Basic physics of Alfven instabilities in toroidal plasmas", Phys. Plasmas 15, 055501 (2008).
- C. Z. Cheng, L. Chen and M. S. Chance, "High-n ideal and resistive shear Alfven waves in tokamaks",
  Ann. Phys. 161, 21 (1985).
- A. Koenies et al., "Benchmark of gyrokinetic, kinetic MHD and gyrofluid codes for the linear calculation of fast
  particle driven TAE dynamics", Nucl. Fusion 58, 126027 (2018).
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[24, 96, 16], help="Number of elements per direction.")
parser.add_argument(
    "--model",
    default="ShearAlfven",
    choices=["ShearAlfven", "ShearAlfvenCurlCurl"],
    help="ShearAlfven (default, magnetic field in H(div), implicit solve for the velocity) or ShearAlfvenCurlCurl "
    "(magnetic field in H(curl), implicit curl-curl solve for the magnetic field).",
)
parser.add_argument(
    "--precond",
    default=None,
    choices=["MassMatrixPreconditioner", "MultiGrid", "StiffnessPreconditioner", "HiptmairXu"],
    help="Preconditioner of the shear Alfven solve: of the mass matrix (default for ShearAlfven), a geometric "
    "multigrid V-cycle of the whole Schur operator (ShearAlfven only; experimental, currently slower here), the "
    "Kronecker (fast diagonalization) preconditioner of the whole Schur operator with the E x B auxiliary space "
    "correction (ShearAlfven only; 8x32x4, dt = 0.1: about 11 instead of 31 PCG iterations per step, 2x faster; "
    "larger gains for larger time steps), or Hiptmair-Xu for the curl-curl operator (ShearAlfvenCurlCurl only, its "
    "default).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 32, 4]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.precond is None:
    args.precond = "HiptmairXu" if args.model == "ShearAlfvenCurlCurl" else "MassMatrixPreconditioner"
if args.precond == ("HiptmairXu" if args.model == "ShearAlfven" else "MultiGrid"):
    parser.error(f"--precond {args.precond} is not available for --model {args.model}.")
if args.model == "ShearAlfvenCurlCurl":
    name = "ITPA TAE benchmark (ShearAlfvenCurlCurl)"
if args.test_run:
    args.num_elements = [8, 32, 4]

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
    FieldsBackground,
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
from struphy.models import ShearAlfven, ShearAlfvenCurlCurl

# Units
base_units = BaseUnits()

# Model instance
model_class = ShearAlfvenCurlCurl if args.model == "ShearAlfvenCurlCurl" else ShearAlfven
model = model_class(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.velocity.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=10, sim_folder=args.sim_folder, max_runtime=470)

# Time stepping
dt = 0.1
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 500.0)

# Geometry
domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=6)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(a=1.0, R0=10.0, B0=3.0, q_kind=0, p_kind=1, q0=1.71, q1=1.87, p1=0.95, p2=0.05, beta=0.0018)
# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profiling options
profiling_opts = ProfilingOptions(use_line_profiler=True)

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

model.propagators.shear_alf.options = model.propagators.shear_alf.Options(precond=args.precond)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
# model.mhd.velocity.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
ms_radial_1, ms_radial_2 = 10, 11
perturbation_radial = perturbations.TorusModesSin(
    ms=(ms_radial_1, ms_radial_2),
    ns=(-1, -1),
    amps=(1e-3, 1e-3),
    pfuns=("exp", "exp"),
    pfun_params=([0.5, 0.1], [0.5, 0.1]),
    comp=0,
    given_in_basis="2",
)
perturbation_poloidal = perturbations.TorusModesCos(
    ms=(10, 11),
    ns=(-1, -1),
    amps=(1e-3 * 1 / (2 * np.pi * ms_radial_1), 1e-3 * 1 / (2 * np.pi * ms_radial_2)),
    pfuns=("d_exp", "d_exp"),
    pfun_params=([0.5, 0.1], [0.5, 0.1]),
    comp=1,
    given_in_basis="2",
)
model.mhd.velocity.add_perturbation(perturbation=perturbation_radial)
model.mhd.velocity.add_perturbation(perturbation=perturbation_poloidal)

if __name__ == "__main__":
    sim.run()
