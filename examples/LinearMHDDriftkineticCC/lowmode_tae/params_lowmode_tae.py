"""Toroidal Alfven eigenmode (TAE) with low mode numbers, driven unstable by energetic ions: the n = 1 TAE in the gap
at the crossing of the m = 1 and m = 2 shear Alfven branches (q = 1.5, r = 0.5) of the LinearMHD low-mode case,
excited by drift-kinetic energetic deuterons with a radial density gradient (LinearMHDDriftkineticCC, delta-f)."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[16, 16, 4], help="Number of elements per direction.")
parser.add_argument("--Np", type=int, default=50_000, help="Number of markers of the energetic ions.")
parser.add_argument(
    "--nh",
    type=float,
    default=0.005,
    help="Density of the energetic ions at the gap (r = 0.5), in units of the bulk. Much larger values (e.g. 0.02, an "
    "energetic-ion beta of 20 times the bulk beta) excite an energetic-particle mode below the gap instead of the TAE.",
)
parser.add_argument(
    "--kappa",
    type=float,
    default=4.0,
    help="Inverse gradient length -d ln(n_h)/dr of the energetic-ion density at the gap (the drive of the TAE).",
)
parser.add_argument(
    "--vth", type=float, default=1.0, help="Thermal speed of the energetic ions, in units of the Alfven speed."
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="Normalized cyclotron period of the energetic ions (their gyro-radius at the Alfven speed, in units of the "
    "minor radius); the default units would give 0.046, i.e. orbit widths of a large fraction of the minor radius.",
)
parser.add_argument("--dt", type=float, default=1.0, help="Time step.")
parser.add_argument("--tend", type=float, default=800.0, help="End time.")
parser.add_argument(
    "--filter",
    default="hybrid",
    choices=["hybrid", "three_point", "none"],
    help="Filter of the accumulated energetic-ion moments: the three-point binomial filter followed by a toroidal "
    "Fourier filter keeping only the n = 1 harmonic of the TAE (default, removes the marker noise in all other "
    "toroidal harmonics), the three-point filter alone, or none. The filters break the exact energy conservation of "
    "the coupling.",
)
parser.add_argument(
    "--seed",
    type=float,
    default=1e-3,
    help="Amplitude of the initial m = 1, 2 velocity perturbation (seed of the TAE).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 8, 4], 2000 markers) and a short run with three saved time steps, for "
    "the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [8, 8, 4]
    args.Np = 2000

name = "Low-mode TAE driven by energetic ions (LinearMHDDriftkineticCC)"
description = f"""
Toroidal Alfven eigenmode (TAE) with low mode numbers (n = 1, m = 1 and 2), driven unstable by energetic ions:
linear ideal MHD coupled to drift-kinetic energetic deuterons (current coupling, delta-f) with the model
LinearMHDDriftkineticCC. The MHD part is the low-mode TAE of examples/ShearAlfven/lowmode_tae; here the energetic
ions are the source of free energy: their radial density gradient at the gap drives the TAE, whose amplitude grows
exponentially.

Physics. The TAE (see the ShearAlfven low-mode example) has the frequency
    omega_TAE = n / (2m + 1) * v_A(r_c) / R0
in the gap of the shear Alfven continuum at q(r_c) = (2m + 1) / (2n). Energetic ions with parallel speed v_par
interact resonantly with the wave when the transit frequency matches, i.e. for v_par = v_A (main resonance) and
v_par = v_A / 3 (sideband resonance) in a large-aspect-ratio torus. The wave extracts energy from the radial gradient
of the energetic-ion distribution if the diamagnetic frequency exceeds the mode frequency,
    omega_*h ~ n q / r * rho_h v_th / L_n > omega_TAE,
with the gyro-radius rho_h = epsilon v_th (normalized cyclotron period epsilon) and the density gradient length L_n,
and loses energy to Landau damping by the velocity gradient (and to continuum damping). The growth rate is a small
fraction of omega_TAE, roughly proportional to the energetic-ion density.

Setup. Tokamak with circular flux surfaces and the MHD equilibrium of the ShearAlfven low-mode example (AdhocTorus,
R0 = 5, a = 1, B0 = 1, q(r) = 1.25 + r^2, flat bulk density, beta 0.0018) on the full torus (HollowTorus with a
hole of radius 0.1), grid {args.num_elements} with cubic splines, time step {args.dt} up to t = {args.tend}. The
compressional MHD (sound wave, background current and pressure steps of LinearMHD) is included. Energetic deuterons (mass number 2, so A_h/A_b = 2) with the
normalized cyclotron period epsilon = {args.epsilon} (gyro-radius at the Alfven speed in units of the minor radius;
set explicitly, since the default units give 0.046, i.e. drift-orbit widths of a large fraction of the minor radius
and large marker losses), represented by {args.Np} markers with delta-f weights (control variate): the
equilibrium is the canonical Maxwellian (a function of the constants of motion, an exact equilibrium of the
guiding-center motion, see CanonicalMaxwellian2D, with the exact canonical toroidal momentum psi_c) with the thermal speed vth = {args.vth} v_A and the density
    n_h(psi_c) = n_h0 * exp(-kappa D tanh((r(psi_c) - r_c) / D)),   D = 0.2,
where r(psi_c) is the minor radius of the flux surface psi = psi_c (canonical toroidal momentum), n_h0 = {args.nh} and
kappa = {args.kappa} (the inverse gradient length at r_c = 0.5). The guiding-center pushers use the discrete gradient
(second order) for the drifts and RK4 for the parallel motion (as in the reference paper for the torus), markers
leaving the plasma are removed. Filter of the accumulated moments: {args.filter} ("hybrid": three-point binomial filter and toroidal
Fourier filter keeping only n = 1, which removes the marker noise in the other toroidal harmonics).
The TAE is seeded by a small m = 1, 2 velocity perturbation localized at the gap (as in the ShearAlfven example).

Expected result. After the initial transient, the TAE (the (m, n) = (1, 1) and (2, 1) harmonics of the radial
velocity at r_c) oscillates with omega close to omega_TAE = 0.067 and grows exponentially. With the defaults, the run (to
t = 800) gave omega = 0.0619 (0.92 omega_TAE) and gamma = 0.0026 (gamma/omega = 0.04); four times the energetic-ion
density (--nh 0.02) excites an energetic-particle mode below the gap instead (omega = 0.046, gamma = 0.006). On 4 MPI
ranks the run takes about 2 s per time step, about 30 min for the default end time. The pproc script
measures omega (asserted within 10% of omega_TAE) and the growth rate gamma from the MHD kinetic energy
(proportional to exp(2 gamma t), asserted positive), and plots the energies (MHD, magnetic, energetic ions), the
fields and the radial profiles of the (m, n) harmonics.

References:
- W. W. Heidbrink, "Basic physics of Alfven instabilities in toroidal plasmas", Phys. Plasmas 15, 055501 (2008).
- G. Y. Fu and J. W. Van Dam, "Excitation of the toroidicity-induced shear Alfven eigenmode by fusion alpha particles
  in an ignited tokamak", Phys. Fluids B 1, 1949 (1989).
- A. Koenies et al., "Benchmark of gyrokinetic, kinetic MHD and gyrofluid codes for the linear calculation of fast
  particle driven TAE dynamics", Nucl. Fusion 58, 126027 (2018).
- B. K. Na, S. Possanner, X. Wang, "Structure-preserving discretization of MHD-driftkinetic current coupling"
  (the discretization of LinearMHDDriftkineticCC).
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------

import cunumpy as xp

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    ProfilingOptions,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import LinearMHDDriftkineticCC
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.filter import FilterParameters

# Units
base_units = BaseUnits()

# Model instance: energetic deuterons on a hydrogen bulk
model = LinearMHDDriftkineticCC(
    base_units=base_units, mhd_mass_number=1.0, hot_charge_number=1, hot_mass_number=2.0, hot_epsilon=args.epsilon
)

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=4, sim_folder=args.sim_folder, max_runtime=24 * 60)

# Time stepping
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=args.dt, Tend=2 * env.save_step * args.dt if args.test_run else args.tend)

# Geometry
a, R0 = 1.0, 5.0
domain = domains.HollowTorus(a1=0.1, a2=a, R0=R0, sfl=False, pol_period=1, tor_period=1)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=a, R0=R0, B0=1.0, q_kind=0, p_kind=1, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0, p1=0.95, p2=0.05, beta=0.0018
)

# Grid
# (no domain decomposition along eta3, needed by the toroidal Fourier filter)
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, False))

# Derham options: u.n = 0 and b.n = 0 at the walls (flux surfaces)
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

# -------------------
# Particle parameters
# -------------------

# the minor radius of the flux surface psi (monotonic psi(r) of the AdhocTorus)
R_TAB = xp.linspace(0.0, a, 2001)
PSI_TAB = equil.psi_r(R_TAB)
R_C, D = 0.5, 0.2


def minor_radius(psi):
    """Minor radius of the flux surface psi (constant beyond the axis and the plasma boundary)."""
    return xp.interp(psi, PSI_TAB, R_TAB)


def n_hot(psi_c):
    """Energetic-ion density as a function of the canonical toroidal momentum: decreasing through r_c with the
    inverse gradient length kappa."""
    r = minor_radius(psi_c)
    return args.nh * xp.exp(-args.kappa * D * xp.tanh((r - R_C) / D))


B_min = equil.params["B0"] * R0 / (R0 + a)
model.energetic_ions.set_markers(
    loading_params=LoadingParameters(Np=args.Np, seed=1234, moments=(0.0, 0.0, args.vth, args.vth), B0=B_min),
    # delta-f: the weights carry f - f0, with the canonical Maxwellian f0
    weights_params=WeightsParameters(control_variate=True),
    boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
    sorting_params=SortingParameters(do_sort=True, sorting_frequency=10),
    saving_params=SavingParameters(n_markers=0),
    # the discrete-gradient pushers sort at mid-points in each iteration
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

epsilon = model.energetic_ions.equation_params.epsilon
filter_params = FilterParameters(use_filter=None if args.filter == "none" else args.filter, modes=(1,))
props = model.propagators
props.push_bxe.options = props.push_bxe.Options(algo="discrete_gradient_2nd_order")
props.push_parallel.options = props.push_parallel.Options(algo="explicit", butcher=ButcherTableau(algo="rk4"))
props.shear_alf.options = props.shear_alf.Options(filter_params=filter_params)
props.cc5d_gradb.options = props.cc5d_gradb.Options(filter_params=filter_params)
# (CurrentCoupling5DDensity accumulates no vector: nothing to filter)
props.cc5d_curlb.options = props.cc5d_curlb.Options(filter_params=filter_params)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# energetic ions: canonical Maxwellian (equilibrium, the control variate of delta-f)
model.energetic_ions.var.add_background(
    # with the exact canonical toroidal momentum (shifted=False): the shifted variant is not a constant of motion for
    # a part of the trapped particles, which would drive delta-f in the steep gradient without any wave
    maxwellians.CanonicalMaxwellian2D(
        n=(n_hot, None), vth=(args.vth, None), equil=equil, epsilon=epsilon, shifted=False
    )
)

# seed of the TAE: the m = 1 and m = 2 harmonics (n = 1: logical mode number -1 along eta3) of the velocity, Gaussian
# around the gap, as divergence-free 2-form components (see the ShearAlfven low-mode example)
eta1_c = (R_C - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
sigma, amp = 0.15, args.seed
ms = (1, 2)
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesSin(
        ms=ms,
        ns=(-1, -1),
        amps=(amp, amp),
        pfuns=("exp", "exp"),
        pfun_params=([eta1_c, sigma], [eta1_c, sigma]),
        comp=0,
        given_in_basis="2",
    )
)
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesCos(
        ms=ms,
        ns=(-1, -1),
        amps=tuple(amp / (2 * xp.pi * m) for m in ms),
        pfuns=("d_exp", "d_exp"),
        pfun_params=([eta1_c, sigma], [eta1_c, sigma]),
        comp=1,
        given_in_basis="2",
    )
)

if __name__ == "__main__":
    # post-processing and plots: pproc_lowmode_tae.py
    sim.run(profiling_activated=True)
