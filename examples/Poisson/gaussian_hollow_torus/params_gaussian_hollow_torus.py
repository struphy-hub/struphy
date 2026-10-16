"""Poisson equation -Laplace phi = rho (electrostatic potential of a charge density) on a hollow torus
(HollowTorus) with a manufactured Gaussian solution (homogeneous Dirichlet boundary conditions in eta1), solved
with CG and a choice of preconditioners. The pproc reports the max. relative error of the right-hand side and the
solution."""

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
# `--precond` selects the preconditioner (the variant), see the submit script; pproc_gaussian_hollow_torus.py takes the
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
    choices=["None", "MultiGrid", "StiffnessPreconditioner"],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[128, 128, 128], help="Number of elements per direction.")
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
    "MultiGrid": ("multigrid preconditioner", "CG preconditioned by geometric multigrid"),
    "StiffnessPreconditioner": (
        "stiffness preconditioner",
        "CG preconditioned by the Kronecker approximation of the inverse Laplacian (StiffnessPreconditioner)",
    ),
}

name = f"Poisson on hollow torus, {PRECONDS[precond][0]}"
description = f"""
Poisson equation -Laplace phi = rho for the electrostatic potential phi of a given charge density rho on a
full hollow torus, with a manufactured localized solution, to test the accuracy and the preconditioners of the
solver in tokamak-like toroidal geometry.

Physics. Poisson's equation -Laplace phi = rho determines the electrostatic potential phi, and
thereby the electric field E = -grad phi, of a given charge density rho (Gauss's law div E = rho for a curl-free
E, in Struphy units). It is the field equation of electrostatic models such as Vlasov-Poisson and the prototype
of the elliptic problems in Struphy. The Poisson model solves -div(D0 grad phi) + n0 phi = rho; here D0 is the
identity and the stabilization term is switched off (stab_eps = 0), i.e. the pure Laplacian with phi in the
spline space V0 of H1 of the discrete de Rham complex (FEEC).
The Gaussian potential corresponds to a localized charge distribution, a positive core inside a
negative shell with zero net charge (the integral of Laplace phi vanishes), hence there is no long-range 1/r tail.
Being localized, it excites the whole spectrum of the Laplacian, whose condition number grows like h^-2 with the
mesh size h, so unpreconditioned CG needs many iterations.

Setup. Full hollow torus (HollowTorus, a1=1e-2, a2=1, R0=4), splines of degree (1, 2, 3).
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1.
The linear system is solved with {PRECONDS[precond][1]}.

Expected result. The pproc script compares the computed phi and the projected right-hand side rho with
the exact fields and prints the max. relative errors; at the default resolution (not with --test-run) it asserts
that they are below 2.2e-2 (rho) and 1.3e-2 (phi). The number of CG iterations shows the efficiency of the
preconditioner.

References:
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
    FieldsBackground,
    ProfilingOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
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
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profilinig options
profiling_opts = ProfilingOptions(
    label=name,
    use_line_profiler=False,
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

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver="pcg",
    precond=precond,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np

from struphy.initial.base import GenericPerturbation

# The exact solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5).
# Its value at the Dirichlet boundary eta1 = 0, 1 is exp(-0.25 / w**2) ~ 1e-11, and it is periodic
# in eta2 and eta3 up to the same accuracy. Being localized, it excites the whole spectrum of the
# Laplacian, so CG has to work for convergence. Both the solution and the right-hand side
# -Laplace(phi) are 0-forms given as functions of the logical coordinates (given_in_basis="0");
# the pushed-forward solution at x = F(eta) is phi(eta).
w = 0.1
eta0 = (0.5, 0.5, 0.5)


def gaussian_derivatives(e1, e2, e3):
    """The Gaussian and its first and second derivatives with respect to eta1, eta2, eta3."""
    eta = (e1, e2, e3)
    g = np.exp(-sum((e - c) ** 2 for e, c in zip(eta, eta0)) / w**2)
    d = [-2 * (e - c) / w**2 * g for e, c in zip(eta, eta0)]
    dd = [(4 * (e - c) ** 2 / w**4 - 2 / w**2) * g for e, c in zip(eta, eta0)]
    return g, d, dd


def exact_solution(e1, e2, e3):
    return gaussian_derivatives(e1, e2, e3)[0]


def laplacian(e1, e2, e3):
    """Physical Laplacian of the exact solution, in logical coordinates.

    HollowTorus (sfl=False): r = a1 + (a2 - a1) * eta1, theta = 2*pi*eta2, phi = 2*pi*eta3 / tor_period,
    R = R0 + r*cos(theta), metric dr**2 + r**2 dtheta**2 + R**2 dphi**2, and
    Laplace(u) = u_rr + (R + r*cos(theta)) / (r*R) * u_r + u_theta,theta / r**2 - sin(theta) / (r*R) * u_theta
    + u_phi,phi / R**2.
    """
    g, d, dd = gaussian_derivatives(e1, e2, e3)
    da = a2 - a1
    r = a1 + da * e1
    theta = 2 * np.pi * e2
    R = R0 + r * np.cos(theta)
    dphi = 2 * np.pi / tor_period
    return (
        dd[0] / da**2
        + (R + r * np.cos(theta)) / (r * R) * d[0] / da
        + dd[1] / (2 * np.pi * r) ** 2
        - np.sin(theta) / (r * R) * d[1] / (2 * np.pi)
        + dd[2] / (dphi * R) ** 2
    )


def rhs_fun(e1, e2, e3):
    return -laplacian(e1, e2, e3)


rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="0")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    # post-processing and plots: pproc_gaussian_hollow_torus.py
    sim.run(profiling_activated=True, one_time_step=True)
