"""Toroidal Alfven eigenmode (TAE) with low mode numbers for the linear variational MHD model in q = sqrt(p)
(VariationalPerturbativeMHD_with_q with linear=True, without viscosity and resistivity): an n = 1 shear Alfven wave
in the gap of the Alfven continuum at the crossing of the m = 1 and m = 2 branches (q = 1.5, r = 0.5) in a tokamak of
aspect ratio 5. The same case as examples/LinearMHD/lowmode_tae, for the comparison of the variational discretization
with the (non-variational) LinearMHD models."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Low-mode TAE (VariationalPerturbativeMHD_with_q, linear)"
description = """
Toroidal Alfven eigenmode (TAE) with low mode numbers with the linear variational MHD model
VariationalPerturbativeMHD_with_q with linear=True, here without viscosity and resistivity, i.e. ideal linear MHD
around a static (zero-flow) equilibrium: shear Alfven waves together with the coupling to the magnetosonic waves
through the pressure and density perturbations. Its unknowns are the perturbations of the density, of q = sqrt(p) and
of the magnetic field with respect to the equilibrium (which enters through `equil`), and the velocity (in H1vec). The
linearized pressure perturbation is p - p0 = 2 q0 q~. No energetic particles: the TAE is excited by the initial
condition, not driven. Same setup as examples/LinearMHD/lowmode_tae, so that the
frequencies of the variational and of the LinearMHD discretizations can be compared.

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
benchmark of Koenies et al. 2018) and on-axis beta 0.0018, gamma = 5/3 (as in LinearMHD). The full torus is computed
(HollowTorus with tor_period = 1 and a hole of radius 0.1): the logical mode number -1 along eta3 is the physical
n = 1. The velocity is initialized with divergence-free m = 1 and m = 2 harmonics, localized around r = 0.5 (given as
2-form components, projected on H1vec) and then made discretely divergence-free in the sense of the scheme
(div Uv u = 0, as in LinearMHD where the commuting projection to H(div) is exactly divergence-free): the H1vec
projection is replaced by its closest (Mv norm) discretely divergence-free field, a change of the order of the
interpolation error (logged as delta_rel; switched off with --no-div-free). The projection is a CG solve without
preconditioner: on the 16 x 16 x 8 grid it stops at its iteration limit with |div u| reduced from 5.7e-5 to 2.6e-9.
The variational propagators run with model="linear_q" and their default
(Newton/Picard) solver options, except implicit_weights=False: the weights of the transport operators at t^n, i.e.
first order in time like the p model (no midpoint option) and LinearMHD (Crank-Nicolson substeps, Lie-Trotter
splitting). With split_sound=True the coupling of the velocity to B (shear-Alfven/fast waves, variat_qb) and to q
(sound waves, variat_sound) are advanced in two implicit substeps, like the shear-Alfven and magnetosonic steps of
LinearMHD: each substep is energy conserving and needs a single Picard iteration, with a smaller linear system.

Expected result. With n = 1 and m = 1 the gap is at q(r_c) = 3/2, i.e. r_c = 0.5, where n = 1 and |B| = 1.0
(flux-surface average), so v_A(r_c) = 1.0 v_unit and
    omega_TAE = 1/3 * 1.0/5 = 0.067 v_unit / x_unit,
a period of about 94 time units. The run up to t = 600 covers about 6.4 periods; the pproc script asserts the
measured frequency within 10% of omega_TAE (the formula gives the center of the gap, whose relative width is about
the inverse aspect ratio r_c/R0 = 0.1) and shows the energies, the components of the fields and the radial profiles
of the (m, n) harmonics. The total energy en_tot (with the auxiliary variables q^ and B^, en_thermo_2 and en_mag_2) is
conserved by the scheme.

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
    "--no-div-free",
    action="store_true",
    help="Keep the plain H1vec projection of the initial velocity (not exactly divergence-free in the scheme).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 8, 4]) and a short run with three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
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
from struphy.models import VariationalPerturbativeMHD_with_q

# adiabatic index (the one of LinearMHD)
gamma = 5.0 / 3.0

# Units
base_units = BaseUnits()

# Model instance: linear and ideal (no viscosity, no resistivity), as LinearMHD
# split_sound: shear-Alfven/fast and sound waves in two substeps, as in LinearMHD
model = VariationalPerturbativeMHD_with_q(
    base_units=base_units, linear=True, with_viscosity=False, with_resistivity=False, split_sound=True
)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.sqrt_p.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=4, sim_folder=args.sim_folder)

# Time stepping
dt = 0.5
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 600.0)

# Geometry
domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, sfl=False, pol_period=1, tor_period=1)

# Fluid equilibrium (enters the linear model through the projected equilibrium, not as a background of the variables)
equil = equils.AdhocTorus(
    a=1.0, R0=5.0, B0=1.0, q_kind=0, p_kind=1, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0, p1=0.95, p2=0.05, beta=0.0018
)
# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options
# Boundary conditions at the walls (flux surfaces, B0.n = 0), as in LinearMHD: b.n = 0 and u.n = 0 strongly. For the
# velocity in H1vec, Dirichlet acts (as for H(div)) only on the first component, u^1 = u.grad(eta1), so the tangential
# velocity is free (ideal slip wall). "free" would drop b.n = 0 and u.n = 0 as well.
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

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
# linear=True needs the "linear_q" variants of the variational propagators (checked by the model)

model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="linear_q", gamma=gamma)
# implicit_weights=False: first order in time, as the p model and LinearMHD (Lie-Trotter splitting)
model.propagators.variat_qb.options = model.propagators.variat_qb.Options(
    model="linear_q", gamma=gamma, implicit_weights=False
)
model.propagators.variat_sound.options = model.propagators.variat_sound.Options(
    model="linear_q", gamma=gamma, implicit_weights=False
)

# ------------------
# Initial conditions
# ------------------
# Linear model: the equilibrium is not part of the variables (it enters through `equil`), the perturbations of the
# density, q and magnetic field start at zero.

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

# Exactly divergence-free initial velocity: after the allocation (projection of the initial conditions), the velocity
# is projected onto the kernel of the discrete divergence div Uv of the variational scheme. Not on a restart, whose
# data overwrites the initial conditions.
if not args.no_div_free and not env.restart:
    from struphy.feec.variational_utilities import project_div_free_velocity

    model_post_allocate = model.post_allocate

    def post_allocate():
        model_post_allocate()
        info = project_div_free_velocity(model.mhd.velocity.spline.vector, sim.derham, sim.mass_ops, sim.basis_ops)
        logging.getLogger("struphy").info(
            f"Divergence-free initial velocity: |div u| {info['div_before']:.3e} -> {info['div_after']:.3e}, "
            f"delta_rel = {info['delta_rel']:.3e} ({info['niter']} CG iterations, success = {info['success']})."
        )

    model.post_allocate = post_allocate

if __name__ == "__main__":
    sim.run(profiling_activated=True)
