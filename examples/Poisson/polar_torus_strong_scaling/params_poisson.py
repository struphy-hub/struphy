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
# The other flags select the variant (spline space and preconditioner), see the submit script; pproc_poisson.py
# takes the same flags.
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
    "--splines",
    default="polar",
    choices=["polar", "tensor"],
    help="C1 polar splines on a full torus (IGAPolarTorus) or tensor-product splines on a torus with a hole (HollowTorus).",
)
parser.add_argument(
    "--precond",
    default="None",
    choices=[
        "None",
        "MassMatrixPreconditioner",
        "PolarMassMatrixPreconditioner",
        "StiffnessPreconditioner",
        "MultiGrid",
    ],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument(
    "--poloidal-solver",
    default="schur",
    choices=["schur", "lu", "ilu"],
    help="Poloidal solver of the polar preconditioners and projectors.",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 66, 64], help="Number of elements per direction.")
parser.add_argument(
    "--hole-bc",
    default="free",
    choices=["free", "dirichlet"],
    help="Boundary condition at the hole of the tensor-product domain (default free; the solution does not vanish there).",
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [16, 18, 16]), for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [16, 18, 16]

precond = None if args.precond == "None" else args.precond

name = f"Poisson strong scaling on a torus, {args.splines} splines, preconditioner {precond}"
description = f"""
Strong scaling test for the Poisson equation on a full torus of minor radius 1 and major radius 4, comparing C1 polar
splines (IGAPolarTorus, no hole) with tensor-product splines on a torus with a hole (HollowTorus, a1=1e-2).
The manufactured solution is a Gaussian ring whose gradient is largest on the magnetic axis r=0.
Homogeneous Dirichlet boundary conditions are set at r=1 (boundary condition {args.hole_bc} at the hole).
The linear system is solved with CG, preconditioner {precond} (poloidal solver {args.poloidal_solver}).
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    ProfilingOptions,
    Simulation,
    Time,
    domains,
    grids,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import Poisson

# Units
base_units = BaseUnits()

# Model instance
model = Poisson(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.em_fields.source.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    out_folders=args.out_folder,
    sim_folder=args.sim_folder,
    restart=False,
)

# Time stepping
time_opts = Time()

# Grid (polar splines need a multiple of 3 elements in eta2)
num_elements = tuple(args.num_elements)
degree = (3, 3, 3)
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, True))

# Geometry and Derham options
a = 1.0
R0 = 4.0
if args.splines == "polar":
    # the spline mapping has the resolution of the grid in the poloidal plane
    domain = domains.IGAPolarTorus(
        num_elements=num_elements[:2], degree=degree[:2], a=a, R0=R0, sfl=False, tor_period=1
    )
    derham_opts = DerhamOptions(
        degree=degree,
        bcs=(("free", "dirichlet"), None, None),
        polar_splines=True,
        polar_poloidal_solver=args.poloidal_solver,
    )
else:
    a1 = 1e-2
    domain = domains.HollowTorus(a1=a1, a2=a, R0=R0, sfl=False, pol_period=1, tor_period=1)
    derham_opts = DerhamOptions(degree=degree, bcs=((args.hole_bc, "dirichlet"), None, None))

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Profiling options
profiling_opts = ProfilingOptions(
    use_line_profiler=True,
)

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

# ------------------
# Propagator options
# ------------------

from struphy.linear_algebra.solver import SolverParameters

# options of the polar preconditioners
precond_params = {}
if args.splines == "polar" and precond in ("PolarMassMatrixPreconditioner", "StiffnessPreconditioner"):
    precond_params["poloidal_solver"] = args.poloidal_solver

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver="pcg",
    precond=precond,
    precond_params=precond_params,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np
import sympy as sym

from struphy.initial.base import GenericPerturbation

# The exact solution is phi = g(R, z) * h: a Gaussian ring g of width w in the poloidal plane (R = sqrt(x**2 + y**2)),
# centered at distance w / sqrt(2) from the magnetic axis (R0, 0), where its gradient is largest, so the gradient is
# largest on the axis r = 0, and a toroidal modulation h = 1 + x / (2 R). It is negligible at r = 1 (Dirichlet boundary).
# Tensor-product splines cannot represent a non-zero gradient across the axis (they live on a domain
# with a hole and a boundary condition there, see --hole-bc); polar splines can.
# The right-hand side -Laplace(phi) is computed symbolically. Both are given as functions of
# the Cartesian coordinates (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
w = 0.2
x0 = w / np.sqrt(2.0)
R_ = sym.sqrt(x_**2 + y_**2)
phi_sym = sym.exp(-((R_ - R0 - x0) ** 2 + z_**2) / w**2) * (1 + x_ / (2 * R_))
rhs_sym = -(sym.diff(phi_sym, x_, 2) + sym.diff(phi_sym, y_, 2) + sym.diff(phi_sym, z_, 2))
phi_fun = sym.lambdify((x_, y_, z_), phi_sym, "numpy")
rhs_lam = sym.lambdify((x_, y_, z_), rhs_sym, "numpy")


def exact_solution(x, y, z):
    return phi_fun(x, y, z) + 0.0 * x


def rhs_fun(x, y, z):
    return rhs_lam(x, y, z) + 0.0 * x


def distance_to_axis(x, y, z):
    return np.sqrt((np.sqrt(x**2 + y**2) - R0) ** 2 + z**2)


model.em_fields.source.add_perturbation(GenericPerturbation(rhs_fun, given_in_basis="physical"))


if __name__ == "__main__":
    # post-processing and plots: pproc_poisson.py
    sim.run(profiling_activated=True, one_time_step=True)
