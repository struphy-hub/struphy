"""Guiding centers in an ad hoc tokamak (full torus without hole, C1 polar splines): a canonical Maxwellian
is an exact equilibrium, a gyro-Maxwellian with the same profiles serves as a control. The pproc compares
the binned distributions with their initial profiles over a long time."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; the profiling driver passes its own
# (see `ProfilingCase.build_commands`).
# The other flags select the variant (initial condition, algorithms of the propagators), see the submit script;
# pproc_tokamak_equilibrium.py takes the same flags.
# Unknown flags are ignored so the driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument(
    "--init",
    default="canonical",
    choices=["canonical", "gyro"],
    help="Canonical Maxwellian (function of the constants of motion, an equilibrium) or gyro-Maxwellian "
    "(same profiles in the flux label of the position, not an equilibrium in a torus).",
)
parser.add_argument(
    "--algo",
    default="discrete_gradient_1st_order",
    choices=[
        "discrete_gradient_1st_order",
        "discrete_gradient_1st_order_newton",
        "discrete_gradient_2nd_order",
        "explicit",
    ],
    help="Algorithm of both guiding-center propagators (push_bxe and push_parallel).",
)
parser.add_argument(
    "--butcher",
    default="rk4",
    choices=["rk4", "forward_euler", "heun2", "rk2", "heun3", "3/8 rule"],
    help="Runge-Kutta method of the explicit algorithm.",
)
parser.add_argument("--epsilon", type=float, default=0.05, help="Normalized cyclotron period (size of the drifts).")
parser.add_argument("--ppc", type=int, default=25, help="Markers per cell.")
parser.add_argument(
    "--num-elements",
    type=int,
    nargs=3,
    default=[16, 48, 48],
    help="Number of elements per direction (eta1 is not decomposed; up to 256 processes for the default).",
)
parser.add_argument("--dt", type=float, default=0.2, help="Time step.")
parser.add_argument("--tend", type=float, default=100.0, help="End time.")
parser.add_argument("--save-step", type=int, default=20, help="Save (and bin) every save_step time steps.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 18, 18]; ppc = 5) and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    # the initial, a middle and a final saved time step
    args.tend = 2 * args.save_step * args.dt
    args.num_elements = [8, 18, 18]
    args.ppc = 5

name = f"GuidingCenter {args.init} Maxwellian in an ad hoc tokamak, {args.algo}"
description = f"""
Long-time equilibrium test of the GuidingCenter model in a full torus without hole (IGAPolarTorus, minor radius 1,
major radius 3, C1 polar splines) with an ad hoc tokamak equilibrium (AdhocTorus). Markers drift through the axis.
Initial condition: {args.init} Maxwellian. A canonical Maxwellian (density and thermal speed functions of the canonical
toroidal momentum psi_c) is an exact equilibrium; a gyro-Maxwellian with the same profiles in the flux label of the
position is not (radial drifts in the torus), it serves as a control.
The binned distributions in position (eta1, eta1-eta2) and velocity space (v_parallel, mu, eta1-v_parallel)
are compared with their initial profiles up to t={args.tend} (dt={args.dt}). The density vanishes at the edge r=a,
where markers on drift orbits leaving the plasma are removed; this small loss is not part of the equilibrium.
Propagators push_bxe and push_parallel: {args.algo} (Butcher {args.butcher} if explicit), epsilon={args.epsilon},
{args.ppc} markers per cell on a grid with {args.num_elements} elements.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------
import numpy as np

from struphy import (
    BaseUnits,
    BinningPlot,
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

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import GuidingCenter
from struphy.ode.utils import ButcherTableau

model = GuidingCenter(base_units=BaseUnits(), epsilon=args.epsilon)
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(
    out_folders=args.out_folder, sim_folder=args.sim_folder, save_step=args.save_step, max_runtime=24 * 60
)

time_opts = Time(dt=args.dt, Tend=args.tend, split_algo="LieTrotter")

# Grid (polar splines need a multiple of 3 elements in eta2); eta1 is not decomposed, such that the process
# with the pole has all rings of the polar splines and each process has at least degree elements per direction
num_elements = tuple(args.num_elements)
degree = (3, 3, 3)
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(False, True, True))

# Geometry: full torus without hole, the spline mapping has the resolution of the grid in the poloidal plane
a = 1.0
R0 = 3.0
domain = domains.IGAPolarTorus(num_elements=num_elements[:2], degree=degree[:2], a=a, R0=R0, sfl=False, tor_period=1)
derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)

# Ad hoc tokamak with circular concentric flux surfaces matching the domain (same a and R0)
B0 = 1.0
equil = equils.AdhocTorus(a=a, R0=R0, B0=B0, q_kind=0, q0=1.71, q1=1.87)

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

# profiles in the normalized poloidal flux s = (psi - psi_axis) / (psi_edge - psi_axis), in [0, 1]
PSI_AXIS, PSI_EDGE = (float(psi) for psi in equil.psi_range)
VTH_AXIS = 0.25
VTH_EDGE = 0.15


def n_of_s(s):
    # vanishes at the edge: markers on drift orbits that leave the plasma at r=a are removed (not replaced)
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s) ** 2


def vth_of_s(s):
    s = np.clip(s, 0.0, 1.0)
    return VTH_EDGE + (VTH_AXIS - VTH_EDGE) * (1.0 - s)


def n_canonical(psi_c):
    """Density as a function of the canonical toroidal momentum."""
    return n_of_s((psi_c - PSI_AXIS) / (PSI_EDGE - PSI_AXIS))


def vth_canonical(psi_c):
    """Thermal speed as a function of the canonical toroidal momentum."""
    return vth_of_s((psi_c - PSI_AXIS) / (PSI_EDGE - PSI_AXIS))


def n_position(*etas):
    """Density at the position, with the flux label s = eta1**2 (r = a * eta1, nearly constant q)."""
    eta1 = etas[0] if len(etas) == 3 else etas[0][:, 0]
    return n_of_s(eta1**2)


def vth_position(*etas):
    """Thermal speed at the position, with the flux label s = eta1**2."""
    eta1 = etas[0] if len(etas) == 3 else etas[0][:, 0]
    return vth_of_s(eta1**2)


B_min = B0 * R0 / (R0 + a)
v_max = 5.0 * VTH_AXIS
mu_max = 12.0 * VTH_AXIS**2 / B_min
binning_plots = (
    BinningPlot(slice="e1", n_bins=32, ranges=(0.0, 1.0)),
    BinningPlot(slice="e1_e2", n_bins=(32, 36), ranges=((0.0, 1.0), (0.0, 1.0))),
    BinningPlot(slice="v1", n_bins=64, ranges=(-v_max, v_max)),
    BinningPlot(slice="v2", n_bins=64, ranges=(0.0, mu_max)),
    BinningPlot(slice="e1_v1", n_bins=(32, 64), ranges=((0.0, 1.0), (-v_max, v_max))),
)

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(
        ppc=args.ppc, seed=1234, moments=(0.0, 0.0, VTH_AXIS, VTH_AXIS), B0=B_min, spatial="disc"
    ),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
    sorting_params=SortingParameters(do_sort=True, sorting_frequency=10),
    saving_params=SavingParameters(n_markers=0, binning_plots=binning_plots),
    # the discrete-gradient pushers sort at mid-points in each iteration
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

butcher = ButcherTableau(algo=args.butcher) if args.algo == "explicit" else None
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo=args.algo, butcher=butcher)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo=args.algo, butcher=butcher)

# ------------------
# Initial conditions
# ------------------

if args.init == "canonical":
    background = maxwellians.CanonicalMaxwellian2D(
        n=(n_canonical, None),
        vth=(vth_canonical, None),
        equil=equil,
        epsilon=args.epsilon,
    )
else:
    background = maxwellians.GyroMaxwellian2D(
        n=(n_position, None),
        vth_para=(vth_position, None),
        vth_perp=(vth_position, None),
        B0=equil.absB0,
    )
model.kinetic_ions.var.add_background(background)

if __name__ == "__main__":
    # post-processing and plots: pproc_tokamak_equilibrium.py
    sim.run(profiling_activated=True)
