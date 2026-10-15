"""Shear Alfvén wave with the GyrokineticMaxwellZLR model: a standing wave B_y ~ cos(2 pi z / L_z) at k_perp = 0
in a periodic slab with B_0 = e_z, ions and (heavy) electrons. The pproc checks the oscillation frequency of the
magnetic energy against omega = k_par / sqrt(1 + sum_s alpha_s^2 n_0 / B_0^2) (light units) and the energy conservation."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; the other flags set the resolution and the physics.
# pproc_shear_alfven.py takes the same flags. Unknown flags are ignored.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_shear_alfven", help="Name of the output folder.")
parser.add_argument(
    "--Np",
    type=int,
    default=8000,
    help="Markers per species (with fewer markers the noise of the magnetisation current masks the wave after a few periods).",
)
parser.add_argument("--dt", type=float, default=5.0, help="Time step (in units of x_hat / c).")
parser.add_argument("--tend", type=float, default=600.0, help="End time.")
parser.add_argument("--amp", type=float, default=0.05, help="Amplitude of each seeded mode of B_y (in units of B_0).")
parser.add_argument(
    "--modes",
    type=int,
    nargs="+",
    default=[1, 2, 3],
    help="Parallel mode numbers n of the seed B_y ~ sum_n cos(2 pi n z / L_z) (several modes show the dispersion branch).",
)
parser.add_argument("--n-orbits", type=int, default=16, help="Saved marker orbits per species (for the orbit plots).")
parser.add_argument("--mass-ratio", type=float, default=100.0, help="Ion-to-electron mass ratio.")
parser.add_argument("--vti", type=float, default=1e-3, help="Ion thermal speed in units of c.")
parser.add_argument("--vte", type=float, default=1e-2, help="Electron thermal speed in units of c.")
parser.add_argument("--no-drifts", action="store_true", help="Switch off the perpendicular drifts and their currents.")
parser.add_argument(
    "--oblique",
    action="store_true",
    help="3D variant: 8 x 8 x 32 cells and oblique seeds B_y ~ cos(2 pi x / L_x + 2 pi n z / L_z) (k_perp != 0), "
    "for the 3D views of the pproc.",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (Np = 200, 8 elements in z), for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()

num_elements = (8, 8, 32) if args.oblique else (4, 1, 32)
if args.test_run:
    args.Np = 200
    num_elements = (4, 1, 8)

name = "GyrokineticMaxwellZLR shear Alfvén wave"
description = f"""
Shear Alfvén wave with the gauge-invariant gyrokinetic Maxwell model in the zero-Larmor-radius limit
(Remmerswaal, Hatzky, Sonnendrücker, J. Plasma Phys. 91, E127, 2025).
Periodic slab 0.25 x 0.25 x 1 m with B_0 = 1 T e_z and n_0 = 1e20 / m^3 (light units: x_hat = 1 m, t_hat = x_hat / c).
Ions (m_i = m_p, v_ti = {args.vti} c) and electrons (m_i / m_e = {args.mass_ratio}, v_te = {args.vte} c),
{args.Np} markers per species, loaded at the same positions (marker-level quasi-neutrality).
Seed: B_y = {args.amp} sum_n cos({"2 pi x / L_x + " if args.oblique else ""}2 pi n z) for n in {args.modes}, standing
shear Alfvén waves{" (oblique, k_perp = 2 pi / L_x)" if args.oblique else " at k_perp = 0"}, which oscillate with
omega = k_par / sqrt(1 + sum_s alpha_s^2 n_0 / B_0^2) (undamped at k_perp = 0, since then E_par = 0).
Time step dt = {args.dt}, end time {args.tend}, drifts {"off" if args.no_drifts else "on"}.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# ------------------
# Import Struphy API
# ------------------

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
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import GyrokineticMaxwellZLR

# Units: x_hat = 1 m, B_hat = 1 T, n_hat = 1e20 / m^3 (v_hat = c)
base_units = BaseUnits(x=1.0, B=1.0, n=1.0)

# Model instance
model = GyrokineticMaxwellZLR(
    base_units=base_units,
    ion_mass_number=1.0,
    electron_mass_number=1.0 / args.mass_ratio,
    with_drifts=not args.no_drifts,
)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.kinetic_ions.var.save_data = True
model.kinetic_electrons.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder)

time_opts = Time(dt=args.dt, Tend=args.tend, split_algo="LieTrotter")

# Geometry: periodic slab, B_0 = e_z along the long direction
domain = domains.Cuboid(r1=0.25, r2=0.25, r3=1.0)

# Fluid equilibrium: homogeneous slab, provides B_0 and the background density n_0 of both species
equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.0)

# Grid and Derham options
grid = grids.TensorProductGrid(num_elements=num_elements)
derham_opts = DerhamOptions(degree=(1, 1, 2))

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

# The loading parameters set the thermal speeds of the sampling density in light units (the default unit
# moments would be v_th = c). The same seed for both species gives identical positions.
for species, vth in ((model.kinetic_ions, args.vti), (model.kinetic_electrons, args.vte)):
    species.set_markers(
        loading_params=LoadingParameters(Np=args.Np, seed=1234, moments=(0.0, 0.0, vth, vth), B0=1.0),
        boundary_params=BoundaryParameters(bc=("periodic", "periodic", "periodic")),
        saving_params=SavingParameters(n_markers=args.n_orbits),
    )

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.epar_ions.options = model.propagators.epar_ions.Options()
model.propagators.epar_electrons.options = model.propagators.epar_electrons.Options()
model.propagators.push_para_ions.options = model.propagators.push_para_ions.Options()
model.propagators.push_para_electrons.options = model.propagators.push_para_electrons.Options()
if not args.no_drifts:
    model.propagators.drifts.options = model.propagators.drifts.Options()

# ------------------
# Initial conditions
# ------------------
# b_field is the perturbation of the equilibrium field (B_0 comes from the equilibrium): no background.

# gyro-Maxwellian backgrounds with the thermal speeds of the loading
for species, vth in ((model.kinetic_ions, args.vti), (model.kinetic_electrons, args.vte)):
    species.var.add_background(
        maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(vth, None), vth_perp=(vth, None), B0=1.0),
    )

# shear Alfvén seed B_y ~ cos(2 pi z): the amplitude must exceed the marker noise of the magnetisation current
# (full-f, sum_s alpha_s^2 = c^2 / v_A^2 is large), which otherwise drives fields of comparable size
model.em_fields.b_field.add_perturbation(
    perturbations.ModesCos(
        ls=(1,) * len(args.modes) if args.oblique else None,
        ns=tuple(args.modes),
        amps=(args.amp,) * len(args.modes),
        Lx=domain.params["r1"],
        Lz=domain.params["r3"],
        given_in_basis="physical",
        comp=1,
    ),
)

if __name__ == "__main__":
    # post-processing and plots: pproc_shear_alfven.py
    sim.run()
