"""Toroidal Alfven eigenmode (TAE) with low mode numbers for the LinearMHD models: an n = 1 shear Alfven wave in
the gap of the Alfven continuum at the crossing of the m = 1 and m = 2 branches (q = 1.5, r = 0.5) in a tokamak of
aspect ratio 5. A cheap verification of the TAE frequency on a coarse grid (compare with the ITPA TAE benchmark
and with the ShearAlfven version of this example)."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Low-mode TAE (LinearMHD)"
description = """
Toroidal Alfven eigenmode (TAE) with low mode numbers with the LinearMHD model: linear ideal MHD around a static
(zero-flow) equilibrium, i.e. shear Alfven waves together with the coupling to the magnetosonic waves through the
pressure and density perturbations. No energetic particles: the TAE is excited by the initial condition, not driven.
The low mode numbers (n = 1, m = 1 and 2) need much less resolution than the ITPA TAE benchmark (n = 6, m = 10 and
11), so that this case verifies the frequency of the three LinearMHD models on a laptop. With --model
LinearMHDCurlCurl, the magnetic field is in H(curl) and the shear Alfven step solves a curl-curl system,
preconditioned by Hiptmair-Xu (not energy conserving: the wave is damped numerically); --model
LinearMHDCurlCurlwithProj has the exactly energy conserving shear Alfven step. On 2 MPI ranks, the 1200 time steps
take about 20 min with LinearMHD and with LinearMHDCurlCurl (1 s per step each) and about 2 h with
LinearMHDCurlCurlwithProj (5.6 s per step).

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

Setup. Tokamak with circular flux surfaces (AdhocTorus): major radius R0 = 5, minor radius a = 1, toroidal field
B0 = 1, safety factor q(r) = 1.25 + r^2/a^2 (q = 1.5 at r = 0.5), flat number density n = 1 (as in the ITPA
benchmark of Koenies et al. 2018) and on-axis beta 0.0018. The full torus is computed
(HollowTorus with tor_period = 1 and a hole of radius 0.1): the logical mode number -1 along eta3 is the physical
n = 1. The velocity is initialized with divergence-free m = 1 and m = 2 harmonics, localized around r = 0.5 (given as
2-form components).

Expected result. With n = 1 and m = 1 the gap is at q(r_c) = 3/2, i.e. r_c = 0.5, where n = 1 and |B| = 1.0
(flux-surface average), so v_A(r_c) = 1.0 v_unit and
    omega_TAE = 1/3 * 1.0/5 = 0.067 v_unit / x_unit,
a period of about 94 time units. The run up to t = 600 covers about 6.4 periods; the pproc script asserts the
measured frequency within 10% of omega_TAE (the formula gives the center of the gap, whose relative width is about
the inverse aspect ratio r_c/R0 = 0.1) and shows the energies, the components of the fields and the radial profiles
of the (m, n) harmonics.

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
parser.add_argument("--num-elements", type=int, nargs=3, default=[16, 16, 8], help="Number of elements per direction.")
parser.add_argument(
    "--model",
    default="LinearMHD",
    choices=["LinearMHD", "LinearMHDCurlCurl", "LinearMHDCurlCurlwithProj"],
    help="LinearMHD (default, magnetic field in H(div), implicit solve for the velocity), LinearMHDCurlCurl "
    "(magnetic field in H(curl), implicit curl-curl solve for the magnetic field, numerically damped) or "
    "LinearMHDCurlCurlwithProj (as LinearMHDCurlCurl, energy conserving shear Alfven step).",
)
parser.add_argument(
    "--precond",
    default=None,
    choices=["MassMatrixPreconditioner", "MultiGrid", "StiffnessPreconditioner", "HiptmairXu"],
    help="Preconditioner of the shear Alfven solve: of the mass matrix (default for LinearMHDCurlCurlwithProj), a geometric "
    "multigrid V-cycle of the whole Schur operator (LinearMHD only; experimental), the Kronecker (fast "
    "diagonalization) preconditioner of the whole Schur operator with the E x B auxiliary space correction "
    "(LinearMHD only, its default), or Hiptmair-Xu for the curl-curl operator (curl-curl models only; the default of "
    "LinearMHDCurlCurl).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 8, 4]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.5,
    help="Time step (default 0.5); the data is saved every 2 time units, or every step for dt > 2.",
)
args, _ = parser.parse_known_args()
if args.precond is None:
    args.precond = {"LinearMHD": "StiffnessPreconditioner", "LinearMHDCurlCurl": "HiptmairXu"}.get(
        args.model, "MassMatrixPreconditioner"
    )
only = {"MultiGrid": "LinearMHD", "StiffnessPreconditioner": "LinearMHD"}
if (
    args.precond in only
    and args.model != only[args.precond]
    or args.precond == "HiptmairXu"
    and args.model == "LinearMHD"
):
    parser.error(f"--precond {args.precond} is not available for --model {args.model}.")
if args.model != "LinearMHD":
    name = f"Low-mode TAE ({args.model})"
if args.test_run:
    args.num_elements = [8, 8, 4]

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
env = EnvironmentOptions(
    out_folders=args.out_folder, save_step=max(1, round(2.0 / args.dt)), sim_folder=args.sim_folder
)

# Time stepping
dt = args.dt
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 600.0)

# Geometry
domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, sfl=False, pol_period=1, tor_period=1)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=1.0, R0=5.0, B0=1.0, q_kind=0, p_kind=1, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0, p1=0.95, p2=0.05, beta=0.0018
)
# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options
# Boundary conditions at the walls (flux surfaces, B0.n = 0): LinearMHD (magnetic field in H(div)) imposes b.n = 0
# and u.n = 0 strongly (Dirichlet). The curl-curl models (magnetic field in H(curl)) use free boundary conditions:
# Dirichlet would force the tangential magnetic field to zero (unphysical, it damps the TAE). With free boundary
# conditions, u.n = 0 holds weakly in the shear Alfven step and p = 0 weakly in the magnetosonic step.
bcs_eta1 = ("dirichlet", "dirichlet") if args.model == "LinearMHD" else ("free", "free")
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(bcs_eta1, None, None))

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

model.propagators.shear_alf.options = model.propagators.shear_alf.Options(precond=args.precond)
model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# Background for (some) FEEC variables
# model.mhd.velocity.add_background(FieldsBackground())

# Perturbations for (some) FEEC variables
# The m = 1 and m = 2 harmonics of the TAE (n = 1: logical mode number -1 along eta3), Gaussian around eta1_c (the
# radius r_c = 0.5 of the gap), as 2-form components: divergence free for the poloidal amplitude amp / (2 pi m)
# ("d_exp" is the derivative of "exp")
eta1_c = (0.5 - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
sigma, amp = 0.15, 1e-3
ms = (1, 2)
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
