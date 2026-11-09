"""Toroidal Alfven eigenmode (TAE) of the ITPA benchmark with the LinearMHD model: an n = 6 shear Alfven wave
in the gap of the Alfven continuum at the crossing of the m = 10 and m = 11 branches (q = 1.75, r = 0.5) in a
large-aspect-ratio tokamak."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "ITPA TAE benchmark (LinearMHD)"
description = """
Toroidal Alfven eigenmode (TAE) with the LinearMHD model: linear ideal MHD around a static (zero-flow)
equilibrium, i.e. shear Alfven waves together with the coupling to the magnetosonic waves through the pressure
and density perturbations. No energetic particles: the TAE is excited by the initial condition, not driven.
The defaults follow the ShearAlfven version of this benchmark (PR #887). By default (--b-space Hcurl), the magnetic
field is in H(curl) and the implicit shear Alfven step solves a curl-curl system for it, preconditioned by
Hiptmair-Xu. This step is not exactly energy conserving and damps the wave numerically, the more the larger the
time step (ShearAlfven: at t = 300, about 74%, 56% and 28% of the initial energy are left with dt = 0.1, 0.2 (the
default) and 0.5), while the frequency hardly depends on the time step. --b-space Hcurl --projected is the version
with the exactly energy conserving shear Alfven step (mass matrix preconditioner, much slower).

With --b-space Hdiv (magnetic field in H(div), implicit solve for the velocity, exactly energy conserving), the
induction equation uses the commuting projection Pi^1[B0 x u] of the Lorentz drift, whose interpolation error in
the poloidal direction shifts the parallel wave number k_par ~ m b^2 + n b^3 (contravariant components). Near the
gap, n q - m nearly cancels, so that a small error in the poloidal part gives a large error in the frequency: on
the default grid (4.4 elements per wavelength of the m = 10 harmonic) the effective poloidal mode number is about
2-3% too small, the continuum is shifted and there is no TAE (ShearAlfven: omega about 0.13 at r = 0.5). Hdiv needs
about 8 elements per poloidal wavelength, e.g. --num-elements 24 88 8 (ShearAlfven: omega = 0.0835, i.e. 4.07e5
rad/s). The H(curl) version has no such error (weak products with mass matrices).

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
R0 = 10, minor radius a = 1, toroidal field B0 = 3, safety factor q(r) = 1.71 + 0.16 r^2/a^2, hydrogen
bulk with the flat number density 2e19 m^-3 (n = 1 in the density unit, base units n = 0.2) and on-axis beta 0.0018,
as in the ITPA benchmark (Koenies et al. 2018). Only the toroidal mode number n = 6 is computed, on a sixth
of the torus (HollowTorus with tor_period = 6 and a hole of radius 0.1): the logical mode number -1 along eta3 is
the physical n = 6. The velocity is initialized with the m = 10 and m = 11 harmonics, localized around r = 0.5
(given as 2-form components).

Expected result. With n = 6 and m = 10 the gap is at q(r_c) = 21/12 = 1.75, i.e. r_c = 0.5, where n = 1 and
|B| = 3.0 (flux-surface average), so v_A(r_c) = 3.0 v_unit and
    omega_TAE = 6/21 * 3.0/10 = 0.086 v_unit / x_unit,
a period of about 73 time units. With the base units x_unit = 1 m, B = 1 T, n = 2e19 m^-3 and the hydrogen bulk,
v_unit = 4.88e6 m/s, so omega_TAE = 4.2e5 rad/s (67 kHz). The ITPA benchmark reports the ideal MHD frequencies
4.01e5 rad/s (CAS3D) and 4.13e5 rad/s (GYGLES). The run up to t = 500 covers about 6.8 periods; the pproc script
asserts the measured frequency within 10% of omega_TAE and of the benchmark value 4.07e5 rad/s, and shows the
energies, the components of the fields and the radial profiles of the (m, n) harmonics. With the defaults, the
2500 time steps take about 70 min on 4 MPI ranks (1.7 s per step) and give omega = 0.0837 (4.08e5 rad/s); about
38% of the initial energy is left at t = 500 (numerical damping of the H(curl) step).

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
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid; `--dt` and `--tend` the time
# stepping; `--a1` the hole around the axis; `--degree` the spline degrees.
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[24, 44, 8], help="Number of elements per direction.")
parser.add_argument(
    "--b-space",
    default="Hcurl",
    choices=["Hdiv", "Hcurl"],
    help="Space of the magnetic field (model flag b_space): Hcurl (default, implicit curl-curl solve for the magnetic "
    "field, numerically damped unless --projected) or Hdiv (implicit solve for the velocity, exactly energy "
    "conserving; needs --num-elements 24 88 8 for the right frequency, see the description).",
)
parser.add_argument(
    "--projected",
    action="store_true",
    help="With --b-space Hcurl: projection of the Lorentz force in the curl-curl term, energy conserving shear Alfven "
    "step (model flag projected).",
)
parser.add_argument(
    "--precond",
    default=None,
    choices=["MassMatrixPreconditioner", "MultiGrid", "StiffnessPreconditioner", "HiptmairXu"],
    help="Preconditioner of the shear Alfven solve: the Kronecker (fast diagonalization) preconditioner of the whole "
    "Schur operator with the E x B auxiliary space correction (--b-space Hdiv only, its default; 24x44x8, dt = 0.5, "
    "4 MPI ranks: 2.5 s per time step instead of 17 s with the mass matrix preconditioner), of the mass matrix "
    "(default with --projected), a geometric multigrid V-cycle of the whole Schur operator (--b-space Hdiv "
    "only; experimental, currently slower here), or Hiptmair-Xu for the curl-curl operator (--b-space Hcurl only; "
    "its default without --projected).",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.2,
    help="Time step (default 0.2); data is saved about every time unit. With --b-space Hcurl, a smaller time step "
    "damps less (see the description).",
)
parser.add_argument("--tend", type=float, default=500.0, help="End time (default 500, about 6.8 TAE periods).")
parser.add_argument(
    "--a1", type=float, default=0.1, help="Radius of the hole around the magnetic axis (default 0.1, in units of a)."
)
parser.add_argument("--degree", type=int, nargs=3, default=[3, 3, 3], help="Spline degrees per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 32, 4]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.projected and args.b_space != "Hcurl":
    parser.error("--projected requires --b-space Hcurl.")
if args.precond is None:
    if args.b_space == "Hdiv":
        args.precond = "StiffnessPreconditioner"
    else:
        args.precond = "MassMatrixPreconditioner" if args.projected else "HiptmairXu"
if args.precond in ("MultiGrid", "StiffnessPreconditioner") and args.b_space != "Hdiv":
    parser.error(f"--precond {args.precond} requires --b-space Hdiv.")
if args.precond == "HiptmairXu" and args.b_space != "Hcurl":
    parser.error("--precond HiptmairXu requires --b-space Hcurl.")
name = f"ITPA TAE benchmark (LinearMHD, b_space={args.b_space}{', projected' if args.projected else ''})"
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
from struphy.models import LinearMHD

# Units of the ITPA benchmark: x_unit = 1 m, B_unit = 1 T and the bulk density 2e19 m^-3 (hydrogen)
base_units = BaseUnits(x=1.0, B=1.0, n=0.2)

# Model instance
model = LinearMHD(base_units=base_units, b_space=args.b_space, projected=args.projected)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    out_folders=args.out_folder, save_step=max(1, round(1.0 / args.dt)), sim_folder=args.sim_folder, max_runtime=470
)

# Time stepping
dt = args.dt
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else args.tend)

# Geometry
domain = domains.HollowTorus(a1=args.a1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=6)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=1.0, R0=10.0, B0=3.0, q_kind=0, p_kind=1, q0=1.71, q1=1.87, n1=0.0, n2=0.0, na=1.0, p1=0.95, p2=0.05, beta=0.0018
)
# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options
# Boundary conditions at the walls (flux surfaces, B0.n = 0): with --b-space Hdiv, b.n = 0 and u.n = 0 are imposed
# strongly (Dirichlet). With --b-space Hcurl, free boundary conditions are used: Dirichlet would force the tangential
# magnetic field to zero (unphysical, it damps the TAE). With free boundary conditions, u.n = 0 holds weakly in the
# shear Alfven step and p = 0 weakly in the sound wave step.
bcs_eta1 = ("dirichlet", "dirichlet") if args.b_space == "Hdiv" else ("free", "free")
derham_opts = DerhamOptions(degree=tuple(args.degree), bcs=(bcs_eta1, None, None))

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
model.propagators.bg_current.options = model.propagators.bg_current.Options()
model.propagators.sound.options = model.propagators.sound.Options()
model.propagators.bg_pressure.options = model.propagators.bg_pressure.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).

# Background for (some) FEEC variables

# Perturbations for (some) FEEC variables
# The m = 10 and m = 11 harmonics of the TAE (n = 6: logical mode number -1 along eta3), Gaussian around eta1_c (the
# radius r_c = 0.5 of the gap), as 2-form components: divergence free for the poloidal amplitude amp / (2 pi m)
# ("d_exp" is the derivative of "exp")
eta1_c = (0.5 - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
sigma, amp = 0.1, 1e-3
ms = (10, 11)
perturbation_radial = perturbations.TorusModesSin(
    ms=ms,
    ns=(-1, -1),
    amps=(amp, amp),
    pfuns=("exp", "exp"),
    pfun_params=([eta1_c, sigma], [eta1_c, sigma]),
    comp=0,
    given_in_basis="2",
)
perturbation_poloidal = perturbations.TorusModesCos(
    ms=ms,
    ns=(-1, -1),
    amps=tuple(amp / (2 * np.pi * m) for m in ms),
    pfuns=("d_exp", "d_exp"),
    pfun_params=([eta1_c, sigma], [eta1_c, sigma]),
    comp=1,
    given_in_basis="2",
)
model.mhd.velocity.add_perturbation(perturbation=perturbation_radial)
model.mhd.velocity.add_perturbation(perturbation=perturbation_poloidal)


if __name__ == "__main__":
    sim.run(profiling_activated=True)
