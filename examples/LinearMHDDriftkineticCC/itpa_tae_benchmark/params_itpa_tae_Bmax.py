"""Toroidal Alfven eigenmode (TAE) of the ITPA benchmark driven unstable by energetic ions: the n = 6 TAE in the gap at
the crossing of the m = 10 and m = 11 shear Alfven branches (q = 1.75, r = 0.5) of the LinearMHD ITPA example, excited
by drift-kinetic energetic deuterons of 400 keV (LinearMHDDriftkineticCC, delta-f). Canonical Maxwellian with
shift="Bmax"; see params_itpa_tae_B0.py for the other shift."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse
import logging
from pathlib import Path

# Coefficients (c0, c1, c2, c3) of the energetic-ion density, fitted at 400 keV for shift="Bmax" to match the ITPA
# benchmark density profile. For another temperature or shift, refit them.
C_FIT = (0.51643, 0.13619, 0.10274, 0.56458)

# Command line options
# `--out-folder` and `--sim-folder` set the output folder (default sim_Bmax_<b-space>); `--b-space` the space of
# the magnetic field (default Hdiv); `--num-elements` the grid (default 24 96 16 with Hdiv, 24 44 16 with Hcurl;
# PR #887, issue #889); `--a1` the hole around the axis; `--degree` the spline degrees (as in the ShearAlfven and
# LinearMHD ITPA examples).
# `--test-run` selects a small grid and a short run (three saved time steps) for the tests (test_examples.py).
# Unknown flags are ignored so that a driver can forward other parameters as well.
# no -h here when imported by another script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument(
    "--sim-folder", default=None, help="Name of the output folder (default: sim_Bmax_<b-space>, e.g. sim_Bmax_Hdiv)."
)
parser.add_argument(
    "--num-elements",
    type=int,
    nargs=3,
    default=None,
    help="Number of elements per direction (default 24 96 16 with --b-space Hdiv, 24 44 16 with Hcurl: the H(div) "
    "shear Alfven step needs about 8 elements per poloidal wavelength of the m = 10 harmonic for the right TAE "
    "frequency, see the description).",
)
parser.add_argument(
    "--b-space",
    default="Hdiv",
    choices=["Hdiv", "Hcurl"],
    help="Space of the magnetic field (model flag b_space): Hdiv (default) or Hcurl (curl-curl shear Alfven step, "
    "1-form couplings and pushers, fewer poloidal elements, not yet validated by a full run), see the description.",
)
parser.add_argument(
    "--projected",
    action="store_true",
    help="With --b-space Hcurl: projection of the Lorentz force in the curl-curl term (model flag projected).",
)
parser.add_argument("--Np", type=int, default=1_000_000, help="Number of markers of the energetic ions.")
parser.add_argument(
    "--T-hot",
    type=float,
    default=400.0,
    help="Temperature of the energetic deuterons in keV (benchmark: 0-800 keV). The density coefficients C_FIT are "
    "fitted at 400 keV: refit them for any other temperature.",
)
parser.add_argument(
    "--nh-scale",
    type=float,
    default=1.0,
    help="Factor on the energetic-ion density profile of the benchmark (the growth rate is proportional to it).",
)
parser.add_argument("--dt", type=float, default=0.25, help="Time step.")
parser.add_argument("--tend", type=float, default=700.0, help="End time.")
parser.add_argument(
    "--a1", type=float, default=0.1, help="Radius of the hole around the magnetic axis (default 0.1, in units of a)."
)
parser.add_argument("--degree", type=int, nargs=3, default=[3, 3, 3], help="Spline degrees per direction.")
parser.add_argument(
    "--filter",
    default="fourier",
    choices=["fourier", "hybrid", "three_point", "none"],
    help="Filter of the accumulated energetic-ion moments: the toroidal Fourier filter keeping only the n = 6 harmonic "
    "of the TAE (default, removes the marker noise in all other toroidal harmonics), the three-point binomial filter "
    "followed by the Fourier filter, the three-point filter alone, or none. The filters break the exact energy "
    "conservation of the coupling.",
)
parser.add_argument(
    "--drift-algo",
    default="discrete_gradient_2nd_order",
    choices=["discrete_gradient_2nd_order", "discrete_gradient_1st_order", "discrete_gradient_1st_order_newton", "rk4"],
    help="Algorithm of the guiding-center drift push (PushGuidingCenterBxEstar1form, PushGuidingCenterBxEstar "
    "with --b-space Hdiv): a discrete gradient (energy "
    "conserving, iterative) or the explicit RK4.",
)
parser.add_argument(
    "--parallel-algo",
    default="discrete_gradient_1st_order_newton",
    choices=["discrete_gradient_2nd_order", "discrete_gradient_1st_order", "discrete_gradient_1st_order_newton", "rk4"],
    help="Algorithm of the parallel guiding-center push (PushGuidingCenterParallel1form, PushGuidingCenterParallel "
    "with --b-space Hdiv): a discrete gradient (energy "
    "conserving, iterative) or the explicit RK4.",
)
parser.add_argument(
    "--magnetosonic-cross-b0",
    action="store_true",
    help="Replace the background current, background pressure and sound steps by MagnetosonicCrossB0, which writes the "
    "current force and the pressure work through B0 x, like the shear Alfven step (model flag magnetosonic_cross_b0).",
)
parser.add_argument(
    "--marker-bc",
    default="refill",
    choices=["remove", "refill"],
    help="Radial boundary condition of the energetic-ion markers: refill the markers leaving the plasma on the same "
    "flux surface at the up-down mirrored poloidal angle (default; same velocity, so the constants of motion and f0 "
    "are nearly unchanged; for the HollowTorus with the AdhocTorus equilibrium), or remove them (with the profile of "
    "the benchmark, about 20%% of the markers are lost in the first 100 time units).",
)
parser.add_argument(
    "--seed",
    type=float,
    default=1e-3,
    help="Amplitude of the initial m = 10, 11 velocity perturbation (seed of the TAE).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 32, 4], 2000 markers) and a short run with three saved time steps, "
    "for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.sim_folder is None:
    args.sim_folder = f"sim_Bmax_{args.b_space}"
if args.projected and args.b_space != "Hcurl":
    parser.error("--projected requires --b-space Hcurl.")
if args.magnetosonic_cross_b0 and args.b_space != "Hdiv":
    parser.error("--magnetosonic-cross-b0 requires --b-space Hdiv.")
if args.num_elements is None:
    args.num_elements = [24, 96, 16] if args.b_space == "Hdiv" else [24, 44, 16]
if args.test_run:
    args.num_elements = [8, 32, 4]
    args.Np = 2000

if args.T_hot != 400.0:
    logging.getLogger("struphy").warning(
        f"C_FIT is fitted at 400 keV, not at T_hot = {args.T_hot} keV: refit it to match the benchmark density profile."
    )

name = (
    f"ITPA TAE benchmark driven by energetic ions (LinearMHDDriftkineticCC, shift=Bmax, "
    f"b_space={args.b_space}, filter {args.filter})"
)
description = f"""
Toroidal Alfven eigenmode (TAE) of the ITPA benchmark (Koenies et al. 2018), driven unstable by energetic ions:
linear ideal MHD coupled to drift-kinetic energetic deuterons (current coupling, delta-f) with the model
LinearMHDDriftkineticCC. The MHD part is the ITPA TAE of examples/LinearMHD/itpa_tae_benchmark (same equilibrium,
units and seed); here the energetic ions are the source of free energy: their radial density gradient at the gap
drives the TAE, whose amplitude grows exponentially.

Physics. The TAE (see the LinearMHD ITPA example) has the frequency
    omega_TAE = n / (2m + 1) * v_A(r_c) / R0
in the gap of the shear Alfven continuum at q(r_c) = (2m + 1) / (2n), here n = 6, m = 10, r_c = 0.5. Energetic ions
with parallel speed v_par interact resonantly with the wave when their transit frequency matches; in the benchmark,
the resonances at v_par = v_A/3 and v_A/5 dominate the energy transfer. The wave extracts energy from the radial
gradient of the energetic-ion density if the diamagnetic frequency exceeds the mode frequency, and loses energy to
Landau damping by the velocity gradient (and to continuum damping).

Setup (ITPA TAE benchmark, Koenies et al. 2018). Large-aspect-ratio tokamak with circular flux surfaces (AdhocTorus):
R0 = 10 m, a = 1 m, B0 = 3 T, q(r) = 1.71 + 0.16 r^2/a^2, hydrogen bulk with the flat density 2e19 m^-3 and on-axis
beta 0.0018 (units x = 1 m, B = 1 T, n = 2e19 m^-3, so v_unit = 4.88e6 m/s and t_unit = 0.205 us). Only n = 6 is
computed, on a sixth of the torus (HollowTorus with tor_period = 6 and a hole of radius {args.a1}; the logical
mode number -1 along eta3 is the physical n = 6), grid {args.num_elements} with spline degrees {args.degree}, time
step {args.dt} up to t = {args.tend}. The compressional MHD is included, with the background current,
background pressure and sound steps of LinearMHD or, with --magnetosonic-cross-b0, with MagnetosonicCrossB0, which
writes the current force and the pressure work through B0 x like the shear Alfven step:
{args.magnetosonic_cross_b0}.
The magnetic field is in {args.b_space} (--b-space). With H(curl), the shear Alfven step is the curl-curl
step coupled to the magnetization of the energetic ions (ShearAlfvenCurlCurlCurrentCoupling5D: weak products of the
Lorentz force, no basis projection) and the couplings and guiding-center pushers use the 1-form field (issue #889);
44 poloidal elements (about 4 per wavelength of the m = 10 harmonic), as in the ShearAlfven and LinearMHD versions
of this benchmark, which give the benchmark frequency with them (PR #887); the H(curl) setup of this example has not
yet been validated by a full run (issue #889). With H(div) (default, the more robust choice), the shear Alfven step
uses the commuting projection Pi^1[(B0 + b) x u] of the Lorentz drift. Its poloidal interpolation error makes the
effective poloidal mode number about 2-3% too small with 44 elements, which shifts the continuum near the gap (where
n q - m nearly cancels) so much that there is no TAE (PR #887): hence 96 poloidal elements with H(div). Boundary
conditions at the walls: b.n = 0 and u.n = 0 imposed strongly with H(div); free boundary conditions with H(curl)
(Dirichlet would force the tangential magnetic field to zero), where u.n = 0 holds only weakly, through the weak
induction equation of the shear Alfven step.
Energetic deuterons with the temperature T_h = {args.T_hot} keV (thermal speed sqrt(T_h/m_D) = {{vth:.3g}} v_unit,
about 0.3 v_A at 400 keV, gyro-radius about 3 cm in 3 T; the normalized cyclotron period epsilon follows from the
units), represented by {args.Np} markers with delta-f weights (control variate). The density profile is
    n_h = n0 c3 exp(-(c2/c1) tanh((r_c - c0) / c2)),
    r_c = sign(s) sqrt(|s|), s = (psi_c - psi_axis) / (psi_edge - psi_axis),
    n0 = 1.44131e17 m^-3, (c0, c1, c2, c3) = {C_FIT},
(times --nh-scale = {args.nh_scale}). The equilibrium is the canonical Maxwellian with shift="Bmax" (see
CanonicalMaxwellian2D). This shift is an exact equilibrium, but the parallel distribution is not
Maxwellian, so this is a different energetic-ion distribution from the benchmark even with a matching density.
The coefficients are fitted at 400 keV, so that the density in the radius of the position matches the benchmark profile
    n_h(s) = n0 c3 exp(-(c2/c1) tanh((sqrt(s) - c0) / c2)), c0 = 0.49123, c1 = 0.298228, c2 = 0.198739, c3 = 0.521298:
for another temperature, refit them. The benchmark used a distribution of the flux label of the position; with a
distribution of the canonical toroidal momentum its growth rate is about a factor of two smaller (NOVA-K, Sec. 4.4.1
of the paper). The guiding-center pushers use {args.drift_algo} for the drifts (--drift-algo) and
{args.parallel_algo} for the parallel motion (--parallel-algo), the current coupling is explicit, markers leaving the
plasma are removed (--marker-bc remove) or refilled on the same field line (--marker-bc refill): {args.marker_bc}.
Filter of the accumulated moments: {args.filter}.

Cost. With 1e5 markers on 4 MPI ranks (dt = 0.25, default pushers), a time step took about 16 s with --b-space Hdiv
(24 x 96 x 16) and about 19 s with --b-space Hcurl (24 x 44 x 16; about 8 s of it for the mass matrices of the total
field in the shear Alfven step, nonlinear=True by default; about 11 s with nonlinear=False). The marker noise of
the magnetization makes the curl-curl solve of the H(curl) step need about 60-70 Hiptmair-Xu iterations instead of
about 10 without energetic
ions (the H(div) solve: about 25), and the number grows with dt (about 300 at dt = 1). The guiding-center pushers and
the current coupling scale with the number of markers per rank (on 24 x 88 x 8 with 1e6 markers, dt = 1 and RK4 for
the parallel motion, a time step took about 26-30 s with H(div)): the run is meant for a cluster with many more ranks.
The TAE is seeded by a small m = 10, 11 velocity perturbation localized at the gap (as in the LinearMHD example).

Expected result. After the initial transient, the TAE (the (m, n) = (10, 6) and (11, 6) harmonics of the radial
velocity at r_c) oscillates with omega close to omega_TAE = 0.086 (4.2e5 rad/s; the benchmark codes give
4.0e5-4.3e5 rad/s) and grows exponentially. Without the Fourier filter, the marker noise in the other toroidal
harmonics (n = 0, 12, ...) is not removed and contaminates the energies. The pproc script measures omega (asserted
within 10% of omega_TAE and of the ideal MHD frequency 4.07e5 rad/s of the benchmark) and the growth rate gamma from
the MHD kinetic energy (asserted positive with a filter), and plots the energies (MHD, magnetic, energetic ions),
the fields and the radial profiles of the (m, n) harmonics; with --compare it overlays two runs.

References:
- A. Koenies et al., "Benchmark of gyrokinetic, kinetic MHD and gyrofluid codes for the linear calculation of fast
  particle driven TAE dynamics", Nucl. Fusion 58, 126027 (2018).
- W. W. Heidbrink, "Basic physics of Alfven instabilities in toroidal plasmas", Phys. Plasmas 15, 055501 (2008).
- B. K. Na, S. Possanner, X. Wang, "Structure-preserving discretization of MHD-driftkinetic current coupling"
  (the discretization of LinearMHDDriftkineticCC).
"""

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------

import cunumpy as xp

from struphy import (
    BaseUnits,
    BinningPlot,
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
from struphy.physics.physics import ConstantsOfNature
from struphy.pic.accumulation.filter import FilterParameters

# Units of the ITPA benchmark: x_unit = 1 m, B_unit = 1 T and the bulk density 2e19 m^-3 (hydrogen)
base_units = BaseUnits(x=1.0, B=1.0, n=0.2)

# Model instance: energetic deuterons on a hydrogen bulk (epsilon of the energetic ions from the units)
A_BULK, A_HOT = 1.0, 2.0
model = LinearMHDDriftkineticCC(
    base_units=base_units,
    mhd_mass_number=A_BULK,
    hot_charge_number=1,
    hot_mass_number=A_HOT,
    b_space=args.b_space,
    projected=args.projected,
    magnetosonic_cross_b0=args.magnetosonic_cross_b0,
)

# thermal speed sqrt(T_h/m_h) of the energetic ions, in units of the Alfven velocity unit B/sqrt(mu0 n A_b m_H)
con = ConstantsOfNature()
v_unit = base_units.B / xp.sqrt(base_units.n * 1e20 * A_BULK * con.mH * con.mu0)
vth = float(xp.sqrt(args.T_hot * 1e3 * con.e / (A_HOT * con.mH)) / v_unit)
description = description.replace("{vth:.3g}", f"{vth:.3g}")

# List all variables and decide whether to save their data
model.em_fields.b_field.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True
model.energetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, save_step=4, sim_folder=args.sim_folder, max_runtime=24 * 60)

# Time stepping
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=args.dt, Tend=2 * env.save_step * args.dt if args.test_run else args.tend, split_algo="LieTrotter")

# Geometry
a, R0 = 1.0, 10.0
domain = domains.HollowTorus(a1=args.a1, a2=a, R0=R0, sfl=False, pol_period=1, tor_period=6)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=a, R0=R0, B0=3.0, q_kind=0, p_kind=1, q0=1.71, q1=1.87, n1=0.0, n2=0.0, na=1.0, p1=0.95, p2=0.05, beta=0.0018
)

# Grid
# (no domain decomposition along eta3, needed by the toroidal Fourier filter)
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, False))

# Derham options: with H(div), u.n = 0 and b.n = 0 at the walls (flux surfaces) imposed strongly; with H(curl), free
# boundary conditions (Dirichlet would force the tangential magnetic field to zero); u.n = 0 is then imposed only
# weakly, by the weak induction equation of the shear Alfven step (see ShearAlfvenCurlCurlPropagator)
bcs_eta1 = ("dirichlet", "dirichlet") if args.b_space == "Hdiv" else ("free", "free")
derham_opts = DerhamOptions(degree=tuple(args.degree), bcs=(bcs_eta1, None, None))

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

# density in units of the bulk density 2e19 m^-3, as a function of the normalized canonical toroidal momentum
N0 = 1.44131e17 / (base_units.n * 1e20)
itpa_density = perturbations.ITPA_density(n0=N0, c=C_FIT)
psi_axis, psi_edge = equil.psi_range


def n_hot(psi_c):
    """Energetic-ion density as a function of the shifted canonical toroidal momentum."""
    s = (psi_c - psi_axis) / (psi_edge - psi_axis)
    return args.nh_scale * itpa_density(xp.sign(s) * xp.sqrt(xp.abs(s)))


# binned distributions for the density profile and the resonance plots of the pproc script
binplot_r = BinningPlot(slice="e1", n_bins=64, ranges=(0.0, 1.0))
binplot_r_vpar = BinningPlot(slice="e1_v1", n_bins=(64, 64), ranges=((0.0, 1.0), (-5 * vth, 5 * vth)))

model.energetic_ions.set_markers(
    # mu drawn with the local |B_0| (default B0="equil")
    loading_params=LoadingParameters(Np=args.Np, seed=1234, moments=(0.0, 0.0, vth, vth)),
    # delta-f: the weights carry f - f0, with the canonical Maxwellian f0
    weights_params=WeightsParameters(control_variate=True),
    # refill: the markers leaving through the walls are refilled on the same flux surface instead of removed
    boundary_params=BoundaryParameters(bc=(args.marker_bc, "periodic", "periodic")),
    sorting_params=SortingParameters(do_sort=True, sorting_frequency=10),
    saving_params=SavingParameters(n_markers=0, binning_plots=(binplot_r, binplot_r_vpar)),
    # the discrete-gradient pushers sort at mid-points in each iteration
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

epsilon = model.energetic_ions.equation_params.epsilon
use_filter = {"fourier": "fourier_in_tor", "hybrid": "hybrid", "three_point": "three_point", "none": None}[args.filter]
# the logical toroidal mode number 1 is the physical n = 6 (tor_period = 6)
filter_params = FilterParameters(use_filter=use_filter, modes=(1,))
props = model.propagators
if args.drift_algo == "rk4":
    props.push_bxe.options = props.push_bxe.Options(algo="explicit", butcher=ButcherTableau(algo="rk4"))
else:
    props.push_bxe.options = props.push_bxe.Options(algo=args.drift_algo)
if args.parallel_algo == "rk4":
    props.push_parallel.options = props.push_parallel.Options(algo="explicit", butcher=ButcherTableau(algo="rk4"))
else:
    props.push_parallel.options = props.push_parallel.Options(algo=args.parallel_algo)
props.shear_alf.options = props.shear_alf.Options(filter_params=filter_params)
props.cc5d_gradb.options = props.cc5d_gradb.Options(filter_params=filter_params)
# (CurrentCoupling5DDensity accumulates no vector: nothing to filter)
props.cc5d_curlb.options = props.cc5d_curlb.Options(filter_params=filter_params)
if args.magnetosonic_cross_b0:
    props.magnetosonic.options = props.magnetosonic.Options()

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# energetic ions: canonical Maxwellian (equilibrium, the control variate of delta-f)
model.energetic_ions.var.add_background(
    # shift="Bmax" is an exact equilibrium, but the parallel distribution is not Maxwellian
    maxwellians.CanonicalMaxwellian2D(n=(n_hot, None), vth=(vth, None), equil=equil, epsilon=epsilon, shift="Bmax")
)

# seed of the TAE: the m = 10 and m = 11 harmonics (n = 6: logical mode number -1 along eta3) of the velocity, Gaussian
# around the gap, as divergence-free 2-form components (see the LinearMHD ITPA example)
eta1_c = (0.5 - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
sigma, amp = 0.1, args.seed
ms = (10, 11)
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
    # post-processing and plots: pproc_itpa_tae.py
    sim.run(profiling_activated=True)
