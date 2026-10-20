"""Kinetic drift wave in a slab with drift-kinetic ions and adiabatic electrons (DriftKineticElectrostaticAdiabatic);
the same setup as examples/VlasovPoissonElectrostaticAdiabatic/slab_drift_wave, for comparison with full-orbit ions."""

# -----------------------------
# Description of the simulation
# -----------------------------

name = "Slab drift wave (DriftKineticElectrostaticAdiabatic)"
description = """
Drift wave in a slab with a density gradient, with drift-kinetic (guiding-center, 5D) ions and adiabatic electrons
(DriftKineticElectrostaticAdiabatic). The physical setup is the one of the slab_drift_wave example of the full-orbit
model VlasovPoissonElectrostaticAdiabatic, so that the two models can be compared directly.

Physics. The guiding-center distribution f(X, v_par, mu, t) is advected by the parallel streaming, the E x B drift and
the parallel acceleration by E_par (uniform B0 = e_z, no curvature). The potential follows from the quasi-neutrality
equation with the ion polarization density and adiabatic electrons,
    -eps div_perp(n0 grad_perp phi) + n0 / (Z eps T0) phi = n_gc - n0.
As in the full-orbit example, the electrons respond adiabatically to all modes (which_geometry=None, no
flux-surface average): the two examples use the same electron model. In the shearless slab the electrons
could physically not respond to modes with k_z = 0 (no motion along B), which would be the average along the field
lines (which_geometry="cylindrical", average over z); the mode studied here has k_z != 0, where both agree.
A perturbation with wave vector
k = (k_x, k_y, k_z) in a density gradient kappa_n = -d ln n0 / dx propagates in the electron diamagnetic direction (+y).
In the fluid limit
    omega = omega_*e / (1 + k_perp^2 rho_s^2),   omega_*e = epsilon k_y Z T_e kappa_n,   rho_s = epsilon sqrt(Z T_e),
the polarization term being the long-wavelength limit (k_perp rho_i << 1) of the finite-Larmor-radius response of the
full-orbit model. The pproc solves the kinetic dispersion relations of both models, which differ at order
b = k_perp^2 rho_i^2 since the drift-kinetic model has no gyro-average.

Setup. Cuboid of size Lx x Ly x Lz = 2 x 2 pi/3 x 2 pi/0.05 with B0 = e_z (HomogenSlab), epsilon = 0.1, Z = 1,
T_i = T_e = 1. The density n0 ~ exp(-kappa_n x) has the uniform gradient kappa_n = 1; the guiding-center Maxwellian
n0(X) M(v_par, mu) is an exact equilibrium. The fluid equilibrium (ion density and electron pressure n T_e of the
adiabatic term, n0 / |B0|^2 of the polarization) is overwritten accordingly. The initial density perturbation
1e-2 n0 sin(pi x/Lx) cos(k_y y + k_z z) has k_x = pi/Lx, k_y = 3, k_z = 0.05, i.e. b = 0.115. Markers leaving at x = 0 or
x = Lx are removed and phi vanishes there; y and z are periodic. Delta-f with the control variate, explicit
guiding-center pushers, Strang splitting with dt = 0.25 (no cyclotron motion to resolve, unlike the full-orbit model
with dt = 0.05), grid of 8 x 8 x 4 cubic elements, end time Tend = 25.

Expected result. The initial (k_y, k_z) mode on the plane x = Lx/2 oscillates with omega close to the root of the
drift-kinetic dispersion relation, omega = 0.287 (full-orbit model: 0.262). The pproc measures the frequency from the
phase of the Fourier coefficient of phi, plots phi in the (x, y) plane at the first, middle and last saved time and,
with --compare <folder of a full-orbit run>, overlays the mode phase of the full-orbit example.

References:
- W. Horton, "Drift waves and transport", Rev. Mod. Phys. 71, 735 (1999).
- A. J. Brizard and T. S. Hahm, "Foundations of nonlinear gyrokinetic theory", Rev. Mod. Phys. 79, 421 (2007).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid.
# `--test-run` selects a small grid, few markers and a short run (three saved time steps) for the tests
# (test_examples.py). Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[8, 8, 4], help="Number of elements per direction.")
parser.add_argument("--ppc", type=int, default=100, help="Markers per cell.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 4, 4]), few markers and three saved time steps, for the tests "
    "(test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 4, 4]
    args.ppc = 5

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import cunumpy as xp

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
    equils,
    grids,
    maxwellians,
)
from struphy.initial.base import GenericPerturbation
from struphy.models import DriftKineticElectrostaticAdiabatic

# ---------------------
# Physical parameters (also used by the pproc)
# ---------------------

epsilon = 0.1
kappa_n = 1.0
Te = Ti = 1.0
Lx, ky, kz = 2.0, 3.0, 0.05
Ly, Lz = 2 * xp.pi / ky, 2 * xp.pi / kz
kx = xp.pi / Lx
amp = 1e-2

# ---------------------
# Instance of the model
# ---------------------

base_units = BaseUnits(kBT=1.0)
model = DriftKineticElectrostaticAdiabatic(base_units=base_units, epsilon=epsilon)

model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping: no cyclotron motion to resolve
dt = 0.25
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 25.0, split_algo="Strang")

# Geometry
domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=Ly, l3=0.0, r3=Lz)

# Fluid equilibrium: uniform B0 = e_z; density and pressure overwritten below
equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements))

# Derham options: phi = 0 at the walls in x
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

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

# the magnetic moment of the markers is sampled for the field strength B0 (default 2), here |B0| = 1
loading_params = LoadingParameters(ppc=args.ppc, seed=1234, B0=1.0)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(do_sort=True, boxes_per_dim=tuple(max(n // 2, 1) for n in args.num_elements))
saving_params = SavingParameters()

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=1.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(algo="explicit", evaluate_e_field=True)
# no flux-surface average: the electrons respond to all modes, as in the full-orbit example (with "cylindrical",
# the modes with k_z = 0 have no electron response; their ion convective cells, omega ~ 3, made the explicit push
# unstable at dt = 0.25)
model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options(which_geometry=None)
model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(algo="explicit", evaluate_e_field=True)

# ------------------
# Initial conditions
# ------------------

# n0 with mean one over the box (the guiding-center Maxwellian n0(X) M is an exact equilibrium in the slab)
C_n = kappa_n * Lx / (2 * xp.sinh(kappa_n * Lx / 2))


def n0_x(x):
    return C_n * xp.exp(-kappa_n * (x - Lx / 2))


def _etas(etas):
    """Logical coordinates from either (e1, e2, e3) or a single (3, ...) array."""
    if len(etas) == 1:
        return etas[0][0], etas[0][1], etas[0][2]
    return etas


def n_init(*etas):
    e1, _, _ = _etas(etas)
    return n0_x(Lx * e1)


def pert_func(*etas):
    e1, e2, e3 = _etas(etas)
    x, y, z = Lx * e1, Ly * e2, Lz * e3
    return amp * n0_x(x) * xp.sin(kx * x) * xp.cos(ky * y + kz * z)


# the adiabatic term uses n0 / T0 of the fluid equilibrium, with T0 = p / n0 = T_e, the polarization n0 / |B0|^2
def n0_xyz(x, y, z):
    return n0_x(x)


def p_xyz(x, y, z):
    return n0_x(x) * Te


equil.n_xyz = n0_xyz
equil.p_xyz = p_xyz

# the magnetic moment mu = v_perp^2 / (2 |B0|) needs the field strength of the slab, |B0| = 1 (the default is 2),
# the same as for the loading of the markers
vth = xp.sqrt(Ti)
background = maxwellians.GyroMaxwellian2D(n=(n_init, None), vth_para=(vth, None), vth_perp=(vth, None), B0=1.0)
model.kinetic_ions.var.add_background(background)
init = maxwellians.GyroMaxwellian2D(
    n=(n_init, GenericPerturbation(pert_func)), vth_para=(vth, None), vth_perp=(vth, None), B0=1.0
)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
