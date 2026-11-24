"""Particles in a static, uniform magnetic field (Vlasov model, periodic cube): a Maxwellian with constant
moments is an exact equilibrium. The pproc compares the binned distributions with their initial and exact
profiles over a long time, for several pushers and splitting schemes."""

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
# The other flags select the variant (algorithms of the propagators, loading), see the submit script;
# pproc_maxwellian_uniform_field.py takes the same flags.
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
    "--vxb-algo", default="analytic", choices=["analytic", "implicit"], help="Algorithm of the push_vxb propagator."
)
parser.add_argument(
    "--butcher",
    default="rk4",
    choices=["rk4", "forward_euler", "heun2", "rk2", "heun3", "3/8 rule"],
    help="Runge-Kutta method of the push_eta propagator.",
)
parser.add_argument("--split", default="LieTrotter", choices=["LieTrotter", "Strang"], help="Splitting algorithm.")
parser.add_argument(
    "--loading", default="pseudo_random", choices=["pseudo_random", "sobol_standard"], help="Marker loading."
)
parser.add_argument("--ppc", type=int, default=100, help="Markers per cell.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[16, 16, 16], help="Number of elements per direction.")
parser.add_argument("--dt", type=float, default=0.05, help="Time step.")
parser.add_argument("--tend", type=float, default=100.0, help="End time (about 16 cyclotron periods).")
parser.add_argument("--save-step", type=int, default=20, help="Save (and bin) every save_step time steps.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [8, 8, 8]; ppc = 10) and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    # the initial, a middle and a final saved time step
    args.tend = 2 * args.save_step * args.dt
    args.num_elements = [8, 8, 8]
    args.ppc = 10

name = f"Vlasov Maxwellian equilibrium in a periodic cube, push_vxb {args.vxb_algo}, push_eta {args.butcher}"
description = f"""
Long-time equilibrium test of the Vlasov model (charged particles in a static magnetic field, no self-consistent
fields): a homogeneous Maxwellian in a uniform magnetic field must stay stationary. It tests the particle pushers,
the splitting schemes and the marker loading over many cyclotron periods.

Physics. The distribution f(x, v, t) solves the Vlasov equation
    df/dt + v . grad_x f + (v x B0) . grad_v f = 0,
whose characteristics are the particle orbits dx/dt = v, dv/dt = v x B0 (in units where the cyclotron frequency
of the base units is one). In the uniform field B0 = e_z the particles gyrate with the cyclotron period 2 pi
around the field lines at the Larmor radius |v_perp| (vth = 1 for a thermal particle) and stream freely along z,
conserving |v_perp| and v_z. Any spatially homogeneous distribution that depends on v only through |v_perp| and
v_z is therefore an exact equilibrium, in particular the Maxwellian
    f0 = n / (2 pi)^(3/2) / vth^3 * exp(-|v|^2 / (2 vth^2)).
Numerically, the Lie-Trotter or Strang splitting into the rotation of v (push_vxb, analytic: exact rotation;
implicit) and the free streaming of x (push_eta, explicit Runge-Kutta) must preserve this equilibrium.

Setup. Maxwellian with constant moments (n=1, u=0, vth=1) in a periodic cube (Cuboid, side 2 pi) with a uniform
magnetic field B0 = e_z (HomogenSlab). Binned distributions in position (eta1, eta1-eta2) and velocity (v1, v3,
v1-v2) up to t={args.tend} (dt={args.dt}, {args.split} splitting), i.e. about {args.tend / (2 * 3.141592653589793):.1f}
cyclotron periods. Propagators: push_vxb ({args.vxb_algo}), push_eta ({args.butcher}). Loading {args.loading},
{args.ppc} markers per cell on a grid with {args.num_elements} elements.

Expected result. The binned distributions stay at their initial and exact profiles (uniform in position,
Gaussian in each velocity component): the relative L2 deviation from the exact profiles stays at the level of the
Monte-Carlo noise of the binning (lower for the quasi-random sobol_standard loading), without a systematic drift.
The kinetic energy is invariant under the exact (analytic) rotation of v and under the free streaming; the
pproc shows its relative drift.

References:
- F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements
  with particles-in-cell", J. Comput. Phys. 433, 110143 (2021).
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
from struphy.models import Vlasov
from struphy.ode.utils import ButcherTableau

model = Vlasov(base_units=BaseUnits())
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

env = EnvironmentOptions(
    out_folders=args.out_folder, sim_folder=args.sim_folder, save_step=args.save_step, max_runtime=24 * 60
)

# Time stepping (the cyclotron period is 2 pi in units of the model)
time_opts = Time(dt=args.dt, Tend=args.tend, split_algo=args.split)

# Geometry: periodic cube of side 2 pi
L = 2 * np.pi
domain = domains.Cuboid(r1=L, r2=L, r3=L)

# Uniform magnetic field along z
equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)

# Grid and Derham options (only the projection of the magnetic field)
num_elements = tuple(args.num_elements)
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, True))
derham_opts = DerhamOptions(degree=(2, 2, 2))

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

vth = 1.0
v_max = 5.0 * vth
binning_plots = (
    BinningPlot(slice="e1", n_bins=32, ranges=(0.0, 1.0)),
    BinningPlot(slice="e1_e2", n_bins=(32, 32), ranges=((0.0, 1.0), (0.0, 1.0))),
    BinningPlot(slice="v1", n_bins=64, ranges=(-v_max, v_max)),
    BinningPlot(slice="v3", n_bins=64, ranges=(-v_max, v_max)),
    BinningPlot(slice="v1_v2", n_bins=(64, 64), ranges=((-v_max, v_max), (-v_max, v_max))),
)

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=args.ppc, loading=args.loading, seed=1234),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(bc=("periodic", "periodic", "periodic")),
    sorting_params=SortingParameters(do_sort=True, sorting_frequency=10),
    saving_params=SavingParameters(n_markers=0, binning_plots=binning_plots),
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_vxb.options = model.propagators.push_vxb.Options(algo=args.vxb_algo)
model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=ButcherTableau(algo=args.butcher))

# ------------------
# Initial conditions
# ------------------

background = maxwellians.Maxwellian3D(n=(1.0, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None))
model.kinetic_ions.var.add_background(background)


def gaussian(v):
    return np.exp(-(v**2) / (2 * vth**2))


# exact equilibrium profiles of the binned distributions (normalized to the binned data)
exact = {
    "e1_density": lambda e1: np.ones_like(e1),
    "e1_e2_density": lambda e1, e2: np.ones_like(e1),
    "v1_density": gaussian,
    "v3_density": gaussian,
    "v1_v2_density": lambda v1, v2: gaussian(v1) * gaussian(v2),
}

if __name__ == "__main__":
    # post-processing and plots: pproc_maxwellian_uniform_field.py
    sim.run(profiling_activated=True)
