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
# The other flags select the variant (algorithms of the propagators), see the submit script;
# pproc_guiding_center.py takes the same flags.
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
args, _ = parser.parse_known_args()

name = f"GuidingCenter gyro-Maxwellian equilibrium in a screw pinch, {args.algo}"
description = f"""
Long-time equilibrium test of the GuidingCenter model in a straight cylinder without hole
(IGAPolarCylinder, minor radius 1, length 2 pi R0, C1 polar splines) with a screw pinch equilibrium (ScrewPinch).
The guiding-center drifts keep the radius constant, hence a gyro-Maxwellian whose density and thermal speed
depend only on r (taken from the equilibrium) is an exact equilibrium. Markers move through the axis.
The binned distributions in position (eta1, eta1-eta2) and velocity space (v_parallel, mu, eta1-v_parallel)
are compared with their initial profiles (and the radial density with the exact one) up to t={args.tend} (dt={args.dt}).
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

# Geometry: cylinder without hole, the spline mapping has the resolution of the grid in the poloidal plane
a = 1.0
R0 = 5.0
domain = domains.IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], a=a, Lz=2 * np.pi * R0)
derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)

# Screw pinch with a peaked density; the pressure offset p0 keeps the thermal speed finite at r=a
equil = equils.ScrewPinch(a=a, R0=R0, B0=1.0, q0=1.05, q1=1.8, n1=2.0, n2=1.0, na=0.2, p0=0.005)

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

# radial profiles of the gyro-Maxwellian: density and thermal speed sqrt(p/n) of the equilibrium
r_eval = np.linspace(0.0, a, 101)
vth_max = float(np.max(np.sqrt(equil.p_r(r_eval) / equil.n_r(r_eval))))
B_min = 1.0


def n_profile(*etas):
    """Density n(r) of the screw pinch, r = a * eta1 (called with eta1, eta2, eta3 or one (n, 3) array)."""
    eta1 = etas[0] if len(etas) == 3 else etas[0][:, 0]
    r = a * eta1
    return 0.2 + 0.8 * (1.0 - r**2 / a**2)


def vth_profile(*etas):
    """Thermal speed sqrt(p(r)/n(r)) of the screw pinch."""
    eta1 = etas[0] if len(etas) == 3 else etas[0][:, 0]
    r = a * eta1
    q = 1.05 + (1.8 - 1.05) * r**2 / a**2
    p = 0.005 + 1.0**2 * a**2 * 1.05 / (2 * R0**2 * (1.8 - 1.05)) * (1 / q**2 - 1 / 1.8**2)
    return np.sqrt(p / n_profile(*etas))


# the profiles above must be those of the equilibrium
assert np.allclose(n_profile(r_eval / a, 0 * r_eval, 0 * r_eval), equil.n_r(r_eval))
assert np.allclose(vth_profile(r_eval / a, 0 * r_eval, 0 * r_eval), np.sqrt(equil.p_r(r_eval) / equil.n_r(r_eval)))

v_max = 5.0 * vth_max
mu_max = 12.0 * vth_max**2 / B_min
binning_plots = (
    BinningPlot(slice="e1", n_bins=32, ranges=(0.0, 1.0)),
    BinningPlot(slice="e1_e2", n_bins=(32, 36), ranges=((0.0, 1.0), (0.0, 1.0))),
    BinningPlot(slice="v1", n_bins=64, ranges=(-v_max, v_max)),
    BinningPlot(slice="v2", n_bins=64, ranges=(0.0, mu_max)),
    BinningPlot(slice="e1_v1", n_bins=(32, 64), ranges=((0.0, 1.0), (-v_max, v_max))),
)

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(
        ppc=args.ppc, seed=1234, moments=(0.0, 0.0, vth_max, vth_max), B0=1.0, spatial="disc"
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

background = maxwellians.GyroMaxwellian2D(
    n=(n_profile, None),
    vth_para=(vth_profile, None),
    vth_perp=(vth_profile, None),
    B0=equil.absB0,
)
model.kinetic_ions.var.add_background(background)

# exact radial profile of the binned distribution (normalized to the binned data): the binning divides by the
# spatial Jacobian only, the velocity-space Jacobian |B0| of (v_parallel, mu) remains
exact = {"e1_density": lambda e1: n_profile(e1, 0 * e1, 0 * e1) / equil.absB0(np.stack((e1, 0 * e1, 0 * e1), axis=1))}

if __name__ == "__main__":
    # post-processing and plots: pproc_guiding_center.py
    sim.run(profiling_activated=True)
