"""Curl-curl problem curl curl E + sigma E = J on a hollow cylinder (HollowCylinder), the elliptic problem of an
implicit time step of Maxwell's equations (or of eddy-current and magnetostatic problems), with a manufactured
Gaussian vector field, solved with CG and a choice of preconditioners. The pproc reports the max. relative
error of the right-hand side and the solution."""

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
# `--precond` selects the preconditioner (the variant), see the submit script; pproc_gaussian_hollow_cylinder.py takes the
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

name = f"Curl-curl on hollow cylinder, {PRECONDS[precond][0]}"
description = f"""
Curl-curl problem curl curl E + sigma E = J for the vector field E (e.g. the electric field) with a given
source J on a hollow cylinder, with a manufactured localized solution, to test the accuracy and the
preconditioners of the solver in curvilinear (polar) coordinates.

Physics. The curl-curl problem is the elliptic core of low-frequency and time-implicit electromagnetics.
A backward Euler step of Maxwell's equations, dE/dt = c^2 curl B - J/eps0 and dB/dt = -curl E,
leads after elimination of the new B to
    curl curl E + E / (c dt)^2 = (terms of the old time step and the current),
i.e. sigma = 1/(c dt)^2. The same operator appears in eddy-current problems (sigma = mu0 sigma_c / dt with the
electric conductivity sigma_c), and for sigma -> 0 it becomes the magnetostatic operator curl curl A = mu0 J of the
vector potential. Here sigma = 1e-2 is small: on the kernel of the curl, the gradient fields grad psi, the
operator reduces to sigma times the identity, so its spectrum has a cluster of small eigenvalues on all length
scales which point smoothers and preconditioners designed for the Laplacian do not reduce. The Hiptmair-Xu
preconditioner and the hybrid smoother of Hiptmair (multigrid) treat this kernel separately in the H1 space of
the potentials. E lives in the spline space V1 of H(curl) of the discrete de Rham complex (FEEC), so the discrete
gradients are exactly in the kernel of the discrete curl, as in the continuous case. The manufactured solution
E = g c has both a curl-free and a divergence-free part (div E = c . grad g, curl E = grad g x c); being
localized, it excites the whole spectrum of the operator, including the near-kernel of the gradients.

Setup. Hollow cylinder (HollowCylinder, a1=1e-2, a2=1, Lz=4), splines of degree (2, 2, 3), sigma = 1e-2.
The manufactured solution is a Gaussian vector field E = g(x) c in Cartesian coordinates, with a constant
vector c and a Gaussian g centered at x0 inside the domain, exciting the full spectrum.
Homogeneous Dirichlet boundary conditions (vanishing tangential E) are set in direction eta1.
The linear system is solved with {PRECONDS[precond][1]}.

Expected result. The pproc script compares the computed E and the projected right-hand side J with the
exact fields and prints the max. relative errors; at the default resolution (not with --test-run) it asserts that
they are below 3e-2 (J) and 3e-2 (E). The number of CG iterations shows the efficiency of the
preconditioner.

References:
- R. Hiptmair and J. Xu, "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces",
  SIAM J. Numer. Anal. 45, 2483 (2007).
- R. Hiptmair, "Multigrid method for Maxwell's equations", SIAM J. Numer. Anal. 36, 204 (1998).
- A. Buffa, J. Rivas, G. Sangalli and R. Vazquez, "Isogeometric discrete differential forms in three
  dimensions", SIAM J. Numer. Anal. 49, 818 (2011).
- F. Holderied, S. Possanner and X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite
  elements with particles-in-cell", J. Comput. Phys. 433, 110143 (2021).
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
Lz = 4.0
domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

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
# Gaussian of width 0.15 centered at mid radius, (0.5, 0, 2)
g = sym.exp(-((x_ - 0.5) ** 2 + y_**2 + (z_ - 2.0) ** 2) / 0.15**2)
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
    # post-processing and plots: pproc_gaussian_hollow_cylinder.py
    sim.run(profiling_activated=True, one_time_step=True)
