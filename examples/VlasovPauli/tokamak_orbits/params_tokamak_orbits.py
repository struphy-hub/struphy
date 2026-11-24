"""Guiding-center orbits (banana and passing) in an ad hoc tokamak, computed with classical Pauli particles
(VlasovPauli) and time steps of several cyclotron periods."""

# -----------------------------
# Description of the simulation
# -----------------------------

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--dt` the time step, `--epsilon` the normalized
# cyclotron period. `--test-run` selects few markers and a short run (three saved time steps) for the tests
# (test_examples.py). Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--dt", type=float, default=0.5, help="Time step (cyclotron period 2 pi epsilon).")
parser.add_argument("--epsilon", type=float, default=0.01, help="Normalized cyclotron frequency 1/(Omega_c t).")
parser.add_argument("--tend", type=float, default=150.0, help="End time (about one bounce period).")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()

name = "Guiding-center orbits in an ad hoc tokamak (VlasovPauli)"
description = f"""
Guiding-center orbits in an ad hoc tokamak computed with classical Pauli particles (Xiao and Qin 2021): full-orbit
markers with a magnetic moment, loaded without gyration and pushed with the Boris scheme with time steps of
several cyclotron periods. There is no self-consistent field: the markers move in the equilibrium field only.

Physics. A classical Pauli particle with magnetic moment mu moves as
    dx/dt = v,    dv/dt = (1/epsilon) v x B0 - mu grad|B0|.
The markers are drawn from a Maxwellian, mu = v_perp^2 / (2 |B0|) is computed for each marker and its
perpendicular velocity is set to zero. Then the magnetic moment of the perpendicular velocity,
mu' = |v x b0|^2 / (2 |B0|), stays close to zero (slow manifold) and the marker follows the guiding-center orbit of
a particle with magnetic moment mu: parallel streaming, mirror force, grad-B and curvature drifts. Particles with
small v_par / v are trapped on the low-field side (banana orbits), the others circulate (passing orbits). The
energy v^2/2 + mu |B0| is conserved.

Setup. Hollow torus (inner radius 0.1, minor radius 1, major radius R0 = 3) with an ad hoc tokamak equilibrium
(AdhocTorus, B0 = 1 on axis, circular flux surfaces). epsilon = {args.epsilon}, i.e. the cyclotron period is
2 pi epsilon = {2 * 3.14159265 * args.epsilon:.3g}; dt = {args.dt}. 200 markers from a Maxwellian with thermal
velocity 0.5, of which 50 are saved, up to t = {args.tend}. Strang splitting with PushVxB(algo="implicit"),
i.e. the Boris scheme of the classical Pauli particle (BAP2 in Xiao and Qin). Markers leaving the plasma at r = 1
or r = 0.1 are removed.

Expected result. Closed banana and passing orbits in the poloidal plane (for the default parameters, dt = 8
cyclotron periods), a bounded relative energy error without secular drift (at most 1.6e-2 for the default
parameters, second order in dt) and mu' / mu of order 1e-3 to 1e-2, largest for small mu (the markers stay on
the slow manifold). The verification test test_verif_VlasovPauli.py compares such orbits with
the guiding-center pushers of the GuidingCenter model (second-order convergence in dt). The pproc
plots the orbits in the (R, Z) plane, the relative energy error and max mu' of the saved markers against time.

References:
- J. Xiao and H. Qin, "Slow manifolds of classical Pauli particle enable structure-preserving geometric algorithms
  for guiding center dynamics", Comput. Phys. Commun. 265, 107981 (2021).
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    maxwellians,
)
from struphy.models import VlasovPauli

# ---------------------
# Instance of the model
# ---------------------

model = VlasovPauli(base_units=BaseUnits(kBT=1.0), epsilon=args.epsilon)
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder)

# Strang splitting: push_vxb (the last propagator) makes one full rotation step between two half kicks (Boris)
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=args.dt, Tend=2 * args.dt if args.test_run else args.tend, split_algo="Strang")

R0 = 3.0
domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=R0, tor_period=1)
equil = equils.AdhocTorus(a=1.0, R0=R0, B0=1.0)
grid = grids.TensorProductGrid(num_elements=(8, 16, 4))
derham_opts = DerhamOptions(degree=(3, 3, 1), bcs=(("free", "free"), None, None))

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
    loading_params=LoadingParameters(Np=40 if args.test_run else 200, seed=1234),
    boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
    saving_params=SavingParameters(n_markers=10 if args.test_run else 50),
    bufsize=1.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_v_mu.options = model.propagators.push_v_mu.Options()
# Crank-Nicolson rotation (Cayley transform, Boris scheme): correct drifts for dt larger than the cyclotron period
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="implicit")

# ------------------
# Initial conditions
# ------------------

vth = 0.5
background = maxwellians.Maxwellian3D(n=(1.0, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None))
model.kinetic_ions.var.add_background(background)

if __name__ == "__main__":
    sim.run()
