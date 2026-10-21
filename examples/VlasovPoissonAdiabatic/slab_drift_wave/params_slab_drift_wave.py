"""Kinetic drift wave in a slab with full-orbit ions and adiabatic electrons
(VlasovPoissonAdiabatic); the full-orbit counterpart of the drift-kinetic model
DriftKineticElectrostaticAdiabatic."""

# -----------------------------
# Description of the simulation
# -----------------------------

name = "Slab drift wave (VlasovPoissonAdiabatic)"
description = """
Drift wave in a slab with a density gradient, with full-orbit (6D) ions and adiabatic electrons
(VlasovPoissonAdiabatic).

Physics. The ions follow full Lorentz-force orbits,
    dx/dt = v,    dv/dt = (1/epsilon) (-grad phi + v x B0),
and the electrons respond adiabatically, n_e = Z n0 (1 + e phi / T_e). In the quasi-neutral limit the potential
follows from
    n0 phi / (Z epsilon T0) = int f d^3v - n0,
i.e. the electrons respond adiabatically to all modes (no flux-surface average, as in the drift-kinetic slab
example examples/DriftKineticElectrostaticAdiabatic/slab_drift_wave with which_geometry=None).
A perturbation with wave vector k = (k_x, k_y, k_z) in a density gradient kappa_n = -d ln n0 / dx (B0 = e_z)
propagates in the electron diamagnetic direction (+y) with the drift-wave frequency. In the fluid limit with
finite-Larmor-radius effects,
    omega = Gamma_0(b) omega_*e / (1 + (Z T_e / T_i) (1 - Gamma_0(b))),   omega_*e = epsilon k_y Z T_e kappa_n,
with b = k_perp^2 rho_i^2, rho_i = epsilon sqrt(T_i) and Gamma_0(b) = I_0(b) exp(-b). Parallel ion dynamics and ion
Landau damping enter through k_z. The pproc solves the full kinetic dispersion relation (sum over cyclotron
harmonics) and the one of the drift-kinetic model, which differ by O(b) since the drift-kinetic model has no
gyro-average.

Setup. Cuboid of size Lx x Ly x Lz = 2 x 2 pi/3 x 2 pi/0.05 with B0 = e_z (HomogenSlab), epsilon = 0.1, Z = 1,
T_i = T_e = 1. The density n0 ~ exp(-kappa_n x) has the uniform gradient kappa_n = 1. A Vlasov equilibrium must be a
function of the guiding-center position X = x + epsilon v_y; for the exponential profile
    f0 = n0(X) M(v) = n0(x) exp(epsilon^2 kappa_n^2 T_i / 2) M(v - u),   u = -epsilon kappa_n T_i e_y,
a Maxwellian shifted by the ion diamagnetic velocity (n0(x) M(v) alone is not stationary and would fill the
delta-f weights with noise of relative size epsilon kappa_n). The fluid equilibrium (ion density and electron
pressure n T_e of the adiabatic term) is overwritten accordingly. The initial
density perturbation 1e-2 n0 sin(pi x/Lx) cos(k_y y + k_z z) has k_x = pi/Lx, k_y = 3, k_z = 0.05, i.e. b = 0.115.
Markers are reflected at x = 0 and x = Lx, where phi vanishes; y and z are periodic. The reflection v_x -> -v_x keeps
X = x + epsilon v_y and |v|, so f0 stays an equilibrium. Delta-f with the control variate, quasi-neutral Poisson solve,
Strang splitting with dt = 0.05 (12 steps per cyclotron period), grid of 8 x 8 x 4 cubic elements, end time Tend = 25
(about one wave period; later the mode sinks into the PIC noise). In the quasi-neutral limit the ion-acoustic branch
omega = k c_s has no Debye cutoff, so the explicit time stepping needs dt c_s k_max < 2 with k_max ~ pi / (cell size);
with 16 elements in x the run is unstable for dt = 0.05.

Expected result. The initial (k_y, k_z) mode propagates in +y with omega close to the root of the full-orbit
dispersion relation, omega = 0.262 (drift-kinetic model: 0.287; a run gave 0.2619). The pproc measures the frequency
from the phase of the (k_y, k_z) Fourier coefficient of phi on the plane x = Lx/2 and plots phi in the (x, y) plane
at the first, middle and last saved time.

References:
- W. Horton, "Drift waves and transport", Rev. Mod. Phys. 71, 735 (1999).
- N. A. Krall and A. W. Trivelpiece, "Principles of Plasma Physics", McGraw-Hill (1973), chapter 9.
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
from struphy.models import VlasovPoissonAdiabatic

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
model = VlasovPoissonAdiabatic(base_units=base_units, epsilon=epsilon)

model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping: the drift (PushEta) and the Poisson solve come first, so both E-kicks of a Strang step
# use the potential of the current marker positions
dt = 0.05
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
model.propagators.poisson.options = model.propagators.poisson.Options(quasi_neutral=True)
model.propagators.push_v_E.options = model.propagators.push_v_E.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

# ------------------
# Initial conditions
# ------------------

# equilibrium f0 = n0(X) M(v) with X = x + epsilon v_y: for the exponential profile a shifted Maxwellian with the
# ion diamagnetic velocity u_y and density n0(x) exp(epsilon^2 kappa_n^2 T_i / 2), of mean one over the box
u_dia = -epsilon * kappa_n * Ti
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

vth = xp.sqrt(Ti)
background = maxwellians.Maxwellian3D(
    n=(n_init, None), u2=(u_dia, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None)
)
model.kinetic_ions.var.add_background(background)
init = maxwellians.Maxwellian3D(
    n=(n_init, GenericPerturbation(pert_func)), u2=(u_dia, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None)
)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
