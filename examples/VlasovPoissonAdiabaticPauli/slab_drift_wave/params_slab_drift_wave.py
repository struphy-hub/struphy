"""Kinetic drift wave in a slab with guiding-center ions computed from classical Pauli particles and adiabatic
electrons (VlasovPoissonAdiabaticPauli); same setup as examples/VlasovPoissonAdiabatic/slab_drift_wave
(full-orbit ions) and examples/DriftKineticElectrostaticAdiabatic/slab_drift_wave (drift-kinetic ions)."""

# -----------------------------
# Description of the simulation
# -----------------------------

name = "Slab drift wave (VlasovPoissonAdiabaticPauli)"
description = """
Drift wave in a slab with a density gradient, with ions as classical Pauli particles (full-orbit markers with a
magnetic moment, loaded without gyration) and adiabatic electrons (VlasovPoissonAdiabaticPauli).

Physics. A classical Pauli particle (Xiao and Qin 2021) has an intrinsic magnetic moment mu and moves as
    dx/dt = v,    dv/dt = (1/epsilon) (E† + v x B0),    E† = -grad(phi + epsilon mu |B0|).
The markers are drawn from a Maxwellian, mu = v_perp^2 / (2 |B0|) is computed for each marker and its perpendicular
velocity is set to zero. Then mu' = |v x b0|^2 / (2 |B0|) stays close to zero (slow manifold) and the markers move
like guiding centers: parallel streaming, E x B, polarization (and grad-B and curvature) drifts, without gyration.
The electrons respond adiabatically, as in the full-orbit and drift-kinetic slab examples (no flux-surface average).
In the uniform B0 = e_z the magnetic-moment force vanishes. Since the ions do not gyrate, the perpendicular ion
response is that of cold ions, i.e. the polarization density of the drift-kinetic model, without the gyro-average
of the full-orbit model. The dispersion relation is that of the drift-kinetic model, with the polarization term
b / T_i replaced by b / (T_i (1 - epsilon^2 omega^2)) (cyclotron correction, negligible here), b = epsilon^2 k_perp^2 T_i.
Polarization and delta-f: the marker positions are guiding-center positions, and the polarization drift on the slow
manifold compresses the markers. The control variate subtracts f0 along the markers and thereby removes this
compression (the polarization density) from the deposited charge. It is added in the quasi-neutrality equation as
for guiding centers,
    n0 phi / (Z epsilon T0) - epsilon div(n0 / B0^2 grad_perp phi) = int f d^3v - n0
(poisson option polarization=True; --no-polarization switches it off: then omega = 0.317, the drift-kinetic root
without polarization, and a gradient-driven instability grows from t ~ 10).

Setup. As in examples/VlasovPoissonAdiabatic/slab_drift_wave: cuboid Lx x Ly x Lz = 2 x 2 pi/3 x 2 pi/0.05 with
B0 = e_z (HomogenSlab), epsilon = 0.1, Z = 1, T_i = T_e = 1, density n0 ~ exp(-kappa_n x) with kappa_n = 1, initial
perturbation 1e-2 n0 sin(pi x/Lx) cos(k_y y + k_z z) with k_y = 3, k_z = 0.05, phi = 0 at x = 0 and x = Lx,
reflecting walls in x, delta-f with the control variate, quasi-neutral Poisson solve with the polarization term,
8 x 8 x 4 cubic elements,
Tend = 25. Differences: the marker positions are guiding-center positions, so the background is n0(x) times a
Maxwellian at rest (guiding centers have no diamagnetic velocity); the control variate evaluates it at the gyrotropic
velocity of the magnetic moment (PauliParticles6D). PushVxB with algo="implicit" (Crank-Nicolson, i.e. the
Cayley rotation of the Boris scheme) and Strang splitting, such that the E x B drift is exact for any dt / epsilon.
The time step is set with --dt: the default dt = 0.05 resolves the cyclotron period 2 pi epsilon = 0.63
(as in the full-orbit example), dt = 0.5 and dt = 1 (8 and 16 cyclotron periods) do not.

Expected result. The initial (k_y, k_z) mode propagates in +y with omega close to the drift-kinetic root
(omega = 0.287; full-orbit model: 0.262). Runs on 2 processes gave omega = 0.2910 (dt = 0.05, 217 s),
0.2915 (dt = 0.5, 36 s) and 0.2939 (dt = 1, 20 s), with max mu' below 5e-3 (the thermal magnetic moment is
T_i / B0 = 1). After t = 25 a weak, gradient-driven instability (independent of dt) takes over: mu' and the field
energy grow, see the model documentation. The pproc measures the frequency from the phase of the (k_y, k_z) Fourier
coefficient of phi on the plane x = Lx/2, compares it with the three roots, and plots the slow-manifold diagnostic
mu_prime_max and phi in the (x, y) plane at the first, middle and last saved time.

References:
- J. Xiao and H. Qin, "Slow manifolds of classical Pauli particle enable structure-preserving geometric algorithms
  for guiding center dynamics", Comput. Phys. Commun. 265, 107981 (2021).
- W. Horton, "Drift waves and transport", Rev. Mod. Phys. 71, 735 (1999).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; `--num-elements` the grid; `--dt` the time step.
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
parser.add_argument("--dt", type=float, default=0.05, help="Time step (cyclotron period 2 pi epsilon = 0.63).")
parser.add_argument(
    "--no-polarization",
    action="store_true",
    help="Without the polarization term in the Poisson equation (the control variate then removes the "
    "polarization density).",
)
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
from struphy.models import VlasovPoissonAdiabaticPauli

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
model = VlasovPoissonAdiabaticPauli(base_units=base_units, epsilon=epsilon)

model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping: Strang splitting, such that push_vxb (the last propagator) makes one full rotation step
# between two half kicks (Boris scheme)
dt = args.dt
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

loading_params = LoadingParameters(ppc=args.ppc, seed=1234)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("reflect", "periodic", "periodic"))
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

model.propagators.push_eta.options = model.propagators.push_eta.Options()
# polarization density of the guiding centers (the control variate removes it from the deposited charge)
model.propagators.poisson.options = model.propagators.poisson.Options(
    quasi_neutral=True, polarization=not args.no_polarization
)
model.propagators.push_v_E.options = model.propagators.push_v_E.Options()
# Crank-Nicolson rotation (Cayley transform, Boris scheme): exact E x B drift for any dt / epsilon
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo="implicit")

# ------------------
# Initial conditions
# ------------------

# guiding-center equilibrium f0 = n0(X) M(v): the marker positions are guiding-center positions, so there is no
# diamagnetic shift of the Maxwellian (unlike the full-orbit example); density of mean one over the box
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


# the adiabatic term of the Poisson solve uses n0 / T0 of the fluid equilibrium, with T0 = p / n0 = T_e
def n0_xyz(x, y, z):
    return n0_x(x)


def p_xyz(x, y, z):
    return n0_x(x) * Te


equil.n_xyz = n0_xyz
equil.p_xyz = p_xyz

# gyrotropic Maxwellians (isotropic, at rest), as required by the classical Pauli particles
vth = xp.sqrt(Ti)
background = maxwellians.Maxwellian3D(n=(n_init, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None))
model.kinetic_ions.var.add_background(background)
init = maxwellians.Maxwellian3D(
    n=(n_init, GenericPerturbation(pert_func)), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None)
)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
