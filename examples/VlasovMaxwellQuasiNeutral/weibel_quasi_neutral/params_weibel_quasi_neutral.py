"""Weibel instability with the VlasovMaxwellQuasiNeutral model, with and without displacement current
(`--kappa 1` or `--kappa 0`), at a time step 20 times larger than in the VlasovMaxwellOneSpecies example."""

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
parser.add_argument("--sim-folder", default=None, help="Name of the output folder (default: sim_kappa<kappa>).")
parser.add_argument(
    "--kappa", type=float, default=0.0, help="Coefficient of the displacement current (1: Maxwell, 0: quasi-neutral)."
)
parser.add_argument("--dt", type=float, default=1.0, help="Time step.")
parser.add_argument("--tend", type=float, default=400.0, help="End time.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.sim_folder is None:
    args.sim_folder = f"sim_kappa{args.kappa:g}"
if args.test_run:
    args.tend = 2 * args.dt

name = "Weibel instability in the quasi-neutral limit"

description = f"""
Weibel instability with the VlasovMaxwellQuasiNeutral model: 1D-3V kinetic electrons (PIC) with Maxwell's
equations in a periodic box with a neutralizing ion background, solved with the implicit, energy conserving
VlasovMaxwellCoupling step. The electrons are hotter in v2 than in v1 (along the wave vector); this temperature
anisotropy makes a small transverse magnetic field B3 grow exponentially until magnetic trapping saturates it.
Here kappa = {args.kappa:g}.

Physics. With alpha = 1 and epsilon = -1 (electrons, time in units of 1/omega_p, velocities in units of c, lengths
in units of the skin depth c/omega_p) the model solves
    df/dt + v . grad_x f - (E + v x B) . grad_v f = 0,
    kappa dE/dt = curl B + int v f d^3v,
    dB/dt = -curl E.
kappa = 1 is the Vlasov-Maxwell system; kappa = 0 drops the displacement current (quasi-neutral, Darwin-type
limit), where E is the Lagrange multiplier of the constraint curl B = J. For a bi-Maxwellian with thermal speeds
vth1 (along k) and vth2, the transverse mode (E2, B3) obeys
    kappa omega^2 - k^2 - 1 + (vth2^2 / vth1^2) [1 + xi Z(xi)] = 0,   xi = omega / (sqrt(2) k vth1)
(the 1 is the plasma term, which stays for kappa = 0). The mode is purely growing, omega = i gamma, and
kappa gamma^2 ~ 8e-4 is negligible against k^2 + 1 = 2.56: the Weibel instability is a low-frequency, transverse
(div J = 0) mode that the quasi-neutral limit keeps.

Setup. As in the VlasovMaxwellOneSpecies Weibel example: periodic box of length 2 pi / k with k = 1.25, 32 cells
and cubic splines in x, Maxwellian electrons with n = 1 and vth1 = 0.02 / sqrt(2), vth2 = sqrt(12) vth1 (v3 = 0),
the seed field B3 = -1e-4 cos(k x), 100000 markers with full-f weights (seed 1234; the quasi-neutral limit needs
full-f weights). Time step dt = {args.dt:g} (20 times the dt = 0.05 of the VlasovMaxwellOneSpecies example; the
implicit coupling step is stable for dt omega_p > 1) up to t = {args.tend:g}, Lie-Trotter splitting.

Expected result. For k = 1.25 the dispersion relation gives gamma = 0.02784 for both kappa = 1 and kappa = 0 (the
growth rate of the amplitude of B3; the magnetic energy grows as exp(2 gamma t)). The energy en_H conserved by the
coupling step changes only by the splitting with the particle pushers. The pproc fits gamma to the magnetic energy
(between t = Tend/5 and 2 Tend/5), compares with the analytical value and plots the energies. Measured with this
setup (serial, 2026-10-23): gamma = 0.0249 for kappa = 1 and gamma = 0.0251 for kappa = 0, about 10 % below the
analytical value; without the initial projection (kappa = 0, option initial_projection=False) gamma = 0.0249, with a
period-2 oscillation of en_B in the first 80 time units, and with dt = 0.25 gamma = 0.0253, so the deviation is
mostly due to the marker noise and resolution, not to the large time step. The relative change of en_H stays below
4e-9.

For kappa = 0 the coupling step projects the initial state onto curl B = J (option initial_projection, default):
the constraint holds only at the mid-point of each step, so initial data that violate it (here by the marker noise
of J) would oscillate with period 2.

References:
- E. S. Weibel, "Spontaneously growing transverse waves in a plasma due to an anisotropic velocity distribution",
  Phys. Rev. Lett. 2, 83 (1959).
- M. Kraus, K. Kormann, P. J. Morrison and E. Sonnendruecker, "GEMPIC: geometric electromagnetic particle-in-cell
  methods", J. Plasma Phys. 83, 905830401 (2017).
"""

# ------------------
# Import Struphy API
# ------------------

import logging
import math

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
    set_logging_level,
)

set_logging_level(logging.WARNING)

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import VlasovMaxwellQuasiNeutral

model = VlasovMaxwellQuasiNeutral(base_units=BaseUnits(), alpha=1.0, epsilon=-1.0)

k = 1.25
B_pert_amp = -1e-4
vth1_background_val = 0.02 / math.sqrt(2)
vth2_background_val = vth1_background_val * math.sqrt(12)

model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder)
time_opts = Time(dt=args.dt, Tend=args.tend, split_algo="LieTrotter")
Lx = 2 * math.pi / k
domain = domains.Cuboid(r1=Lx)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))

sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)

# -------------------
# Particle parameters
# -------------------

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(
        Np=2000 if args.test_run else 100000,
        set_zero_velocity=(False, False, True),
        moments=(0.0, 0.0, 0.0, vth1_background_val, vth2_background_val, 1.0),
        seed=1234,
    ),
    weights_params=WeightsParameters(control_variate=False),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
    saving_params=SavingParameters(n_markers=0),
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.coupling.options = model.propagators.coupling.Options(kappa=args.kappa)
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# ------------------
# Initial conditions
# ------------------

model.kinetic_ions.var.add_background(
    maxwellians.Maxwellian3D(vth1=(vth1_background_val, None), vth2=(vth2_background_val, None))
)
model.em_fields.b_field.add_perturbation(perturbation=perturbations.ModesCos(amps=(B_pert_amp,), ls=(1,), comp=2))

if __name__ == "__main__":
    # post-processing and plots: pproc_weibel_quasi_neutral.py
    sim.run()
