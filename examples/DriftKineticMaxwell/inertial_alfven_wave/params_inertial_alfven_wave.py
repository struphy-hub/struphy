"""Inertial Alfvén wave with the DriftKineticMaxwell model, with and without displacement current (`--lam 1` or
`--lam 0`) and with or without the E x B current (`--exb-current`)."""

import argparse
from pathlib import Path

# Command line options
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default=None, help="Name of the output folder (default: sim_lam<lam>[_exb]).")
parser.add_argument(
    "--lam",
    type=float,
    default=0.0,
    help="lam = v_th / c, coefficient lam^2 of the displacement current (0: quasi-neutral).",
)
parser.add_argument(
    "--exb-current", action="store_true", help="Include the E x B current of the guiding centers in Ampere's law."
)
parser.add_argument("--dt", type=float, default=0.5, help="Time step.")
parser.add_argument("--tend", type=float, default=80.0, help="End time.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.sim_folder is None:
    args.sim_folder = f"sim_lam{args.lam:g}" + ("_exb" if args.exb_current else "")
if args.test_run:
    args.tend = 2 * args.dt

name = "Inertial Alfven wave in the drift-kinetic Maxwell model"

description = f"""
Shear (inertial) Alfven wave in a cold guiding-center plasma with the DriftKineticMaxwell model: drift-kinetic
particles (PIC, Particles5D) coupled to Maxwell's equations through the implicit, energy conserving
DriftKineticMaxwellCoupling step, in a periodic slab with the uniform field B0 = e_z. Here lam = {args.lam:g}
(coefficient lam^2 of the displacement current), E x B current {"on" if args.exb_current else "off"}.

Physics. The model is normalized with the thermal velocity v_th of the species (t = x / v_th, E in units of
v_th B), with the species parameters epsilon = 1 / (Omega_c t), lam_s = Omega_c x / c and kappa_s = Omega_p x / c.
Ampere's law reads
    lam^2 dE/dt + chi n0 / |B0|^2 dE_perp/dt = curl B - c2 J_gc,
with the guiding-center current J_gc, lam = epsilon lam_s = v_th / c (here an option, 0 or 1), the polarization
coefficient chi = (epsilon kappa_s)^2 = v_th^2 / v_A^2 and c2 = epsilon kappa_s^2; the parallel force has
c3 = 1 / epsilon. The drift-kinetic ordering removes the polarization drift from the guiding-center motion; the chi
term adds it back. Without it, the quasi-neutral limit lam = 0 would be singular for flute modes. For a cold plasma
and k = (kx, 0, kz), the shear mode (E_x, E_z, B_y) obeys, without the E x B current and for lam = 0,
    omega^2 = (1 / chi) kz^2 / (1 + kx^2 d^2),   d^2 = 1 / P,   P = c2 c3 n0 = kappa_s^2,
the inertial Alfven wave with v_A^2 = 1 / chi and the inertial length d = 1 / kappa_s. For lam > 0, and with the
E x B current (which couples the shear and the fast wave through the Hall term of the single species), the
frequencies are the roots of the determinant of the 3x3 cold-plasma dispersion matrix (computed in the pproc
script).

Setup. epsilon = 0.5, lam_s = 2, kappa_s = 6 (chi = 9, v_A = 1/3, d = 1/6, c2 = 18, c3 = 2; lam = 1 would mean
v_th = c and only tests the displacement-current branch); box 1 x 1 x 4 with 8 x 1 x 8 cells and cubic splines in
x and z, kx = 2 pi, kz = pi / 2 (kx^2 d^2 = 1.1); cold guiding centers (vth = 1e-6, full-f weights, 50 markers per
cell, seed 1234), initial B_y = 1e-2 sin(kx x + kz z) (2-form component). Time step dt = {args.dt:g}
(omega dt = 0.18) up to t = {args.tend:g}, Lie-Trotter splitting of the coupling step with the parallel and the
E* x b0 guiding-center pushers. For lam = 0 the coupling step first projects the initial state onto the constraint
of the quasi-neutral limit (option initial_projection, default).

Expected result. Analytical frequencies of the shear mode: omega = 0.3616 (lam = 0), 0.2643 (lam = 0 with the
E x B current) and 0.3428 (lam = 1; the B_y perturbation then also excites the upper branch, omega = 8.70, with a
small amplitude). The pproc fits omega to the magnetic energy en_B ~ cos^2(omega t) and compares. Measured with
this example (serial, 2026-10-23): -1.0 % for lam = 0; with the verification test of the same setup -0.25 % with
the E x B current and -0.7 % for lam = 1 (dt = 0.1). The deviation decreases with the number of markers (-0.35 %
with 200 markers per cell). The total energy en_tot is conserved by the whole splitting up to the solver tolerance
(cold plasma, uniform B0): relative change 5e-6 with the default tolerance 1e-8, 1e-9 with 1e-12.

References:
- A. J. Brizard and T. S. Hahm, "Foundations of nonlinear gyrokinetic theory", Rev. Mod. Phys. 79, 421 (2007).
- C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation" (IOP, 1991).
"""

# ------------------
# Import Struphy API
# ------------------

import logging

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
    set_logging_level,
)

set_logging_level(logging.WARNING)

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import DriftKineticMaxwell

EPSILON, LAM, KAPPA = 0.5, 2.0, 6.0
LX, LZ = 1.0, 4.0
VTH = 1e-6

model = DriftKineticMaxwell(base_units=BaseUnits(kBT=1.0), epsilon=EPSILON, lam=LAM, kappa=KAPPA)

model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder)
time_opts = Time(dt=args.dt, Tend=args.tend, split_algo="LieTrotter")
domain = domains.Cuboid(r1=LX, r2=1.0, r3=LZ)
equil = equils.HomogenSlab(B0z=1.0)
grid = grids.TensorProductGrid(num_elements=(8, 1, 8))
derham_opts = DerhamOptions(degree=(3, 1, 3))

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

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(
        ppc=5 if args.test_run else 50,
        moments=(0.0, 0.0, VTH, VTH),
        B0=1.0,
        seed=1234,
    ),
    weights_params=WeightsParameters(control_variate=False),
    saving_params=SavingParameters(n_markers=0),
)

# ------------------
# Propagator options
# ------------------

prop = model.propagators.coupling
if args.exb_current:
    prop.options = prop.Options(lam=args.lam)
else:
    # without the E x B current the Schur system is symmetric
    prop.options = prop.Options(lam=args.lam, exb_current=False, solver="pcg")
model.propagators.push_parallel.options = model.propagators.push_parallel.Options()
# with the electric field (not a potential) only the explicit E* x b0 push is consistent
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo="explicit")

# ------------------
# Initial conditions
# ------------------

cold = dict(n=(1.0, None), vth_para=(VTH, None), vth_perp=(VTH, None), B0=1.0)
model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(**cold))
model.em_fields.b_field.add_perturbation(
    perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="2", comp=1, amps=(1e-2,))
)

if __name__ == "__main__":
    # post-processing and plots: pproc_inertial_alfven_wave.py
    sim.run()
