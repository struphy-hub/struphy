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
# `--precond` selects the preconditioner (the variant), see the submit script; pproc_curl_curl.py takes the
# same flags.
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
    "--precond",
    default="None",
    choices=["None", "HiptmairXu", "MultiGrid", "StiffnessPreconditioner"],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 64, 64], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [16, 16, 16]), for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [16, 16, 16]

precond = None if args.precond == "None" else args.precond

# preconditioner -> (name suffix, linear solver) for the name and the description
PRECONDS = {
    None: ("no preconditioner", "unpreconditioned CG"),
    "HiptmairXu": (
        "Hiptmair-Xu preconditioner",
        "CG preconditioned by the auxiliary space preconditioner of Hiptmair and Xu (HiptmairXuPreconditioner)",
    ),
    "MultiGrid": (
        "multigrid preconditioner",
        "CG preconditioned by geometric multigrid with the hybrid smoother of Hiptmair",
    ),
    "StiffnessPreconditioner": (
        "stiffness preconditioner",
        "CG preconditioned by the Kronecker approximation of the inverse curl-curl operator with gradient correction (StiffnessPreconditioner)",
    ),
}

name = f"Curl-curl strong scaling on hollow torus, {PRECONDS[precond][0]}"
description = f"""
Strong scaling test for the curl-curl problem curl curl E + sigma E = J on a full hollow torus (HollowTorus, a1=1e-2, a2=1, R0=4).
The manufactured solution is a Gaussian vector field E = g(x) c in Cartesian coordinates, with a constant
vector c and a Gaussian g centered at x0 inside the domain, exciting the full spectrum.
Homogeneous Dirichlet boundary conditions (vanishing tangential E) are set in direction eta1.
The linear system is solved with {PRECONDS[precond][1]}.
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
from struphy.models import CurlCurl

# Units
base_units = BaseUnits()

# Model instance
model = CurlCurl(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
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

# Geometry
a1 = 1e-2
a2 = 1.0
R0 = 4.0
tor_period = 1
domain = domains.HollowTorus(a1=a1, a2=a2, R0=R0, sfl=False, pol_period=1, tor_period=tor_period)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(2, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profilinig options
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

# coefficient of the mass term; small values make the kernel of the curl (the gradients) hard to precondition
sigma = 1e-2

solver_params = SolverParameters(tol=1e-8, maxiter=10000 if precond is None else 3000, info=True, recycle=False)
model.propagators.curl_curl.options = model.propagators.curl_curl.Options(
    sigma=sigma,
    solver="pcg",
    precond=precond,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np
import sympy as sym

from struphy.initial.base import GenericPerturbation

# The exact solution is the vector field E = g(x) c in Cartesian coordinates, with a constant vector c and a
# localized Gaussian g (below). It is negligible at the boundary, hence satisfies the boundary conditions
# up to that accuracy. Being localized, it excites the whole spectrum of the curl-curl operator, including
# gradients, so CG has to work for convergence.
# The right-hand side J = curl curl E + sigma E is computed symbolically. Both are given in Cartesian
# components (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
# Gaussian following the torus, centered at mid radius on the outboard side (theta = phi = 0, R = R0 + 0.5):
# width 0.15 in the poloidal plane and 1.5 along the toroidal arc length (Rc * phi, with phi**2 replaced by
# the smooth periodic 2 * (1 - cos(phi))), where the elements are longest (2*pi*R / 64 ~ 0.44)
Rc = R0 + 0.5
R_ = sym.sqrt(x_**2 + y_**2)
g = sym.exp(-((R_ - Rc) ** 2 + z_**2) / 0.15**2 - 2 * Rc**2 * (1 - x_ / R_) / 1.5**2)
c = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)


def curl(F):
    """Curl of a Cartesian vector field (list of sympy expressions)."""
    return [
        sym.diff(F[2], y_) - sym.diff(F[1], z_),
        sym.diff(F[0], z_) - sym.diff(F[2], x_),
        sym.diff(F[1], x_) - sym.diff(F[0], y_),
    ]


E_sym = [ck * g for ck in c]
J_sym = [cc + sigma * e for cc, e in zip(curl(curl(E_sym)), E_sym)]
E_fun = [sym.lambdify((x_, y_, z_), e, "numpy") for e in E_sym]
J_fun = [sym.lambdify((x_, y_, z_), j, "numpy") for j in J_sym]


def exact_solution(comp):
    return lambda x, y, z: E_fun[comp](x, y, z) + 0.0 * x


def rhs(comp):
    return lambda x, y, z: J_fun[comp](x, y, z) + 0.0 * x


for comp in range(3):
    model.em_fields.source.add_perturbation(GenericPerturbation(rhs(comp), given_in_basis="physical", comp=comp))


if __name__ == "__main__":
    # post-processing and plots: pproc_curl_curl.py
    sim.run(profiling_activated=True, one_time_step=True)
