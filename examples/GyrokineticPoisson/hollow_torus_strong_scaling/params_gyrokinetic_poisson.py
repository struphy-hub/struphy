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
# `--precond` selects the preconditioner (the variant), see the submit script;
# pproc_gyrokinetic_poisson.py takes the same flags.
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
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 64, 64], help="Number of elements per direction.")
args, _ = parser.parse_known_args()

precond = None if args.precond == "None" else args.precond

# preconditioner -> (name suffix, linear solver) for the name and the description
PRECONDS = {
    None: ("no preconditioner", "unpreconditioned CG"),
    "MultiGrid": ("multigrid preconditioner", "CG preconditioned by geometric multigrid"),
    "StiffnessPreconditioner": (
        "stiffness preconditioner",
        "CG preconditioned by the Kronecker approximation of the inverse operator (StiffnessPreconditioner)",
    ),
}

name = f"GyrokineticPoisson strong scaling on hollow torus, {PRECONDS[precond][0]}"
description = f"""
Strong scaling test for the gyrokinetic Poisson equation with adiabatic electrons on a full hollow torus
(HollowTorus, a1=1e-2, a2=1, R0=3) in an ad hoc tokamak equilibrium (AdhocTorus, a=1, R0=3).
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1.
The linear system is solved with {PRECONDS[precond][1]}.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)


import numpy as np

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
    equils,
    grids,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import GyrokineticPoisson

# Units
base_units = BaseUnits()

# Model instance
epsilon = 1.0
charge_number = 1
model = GyrokineticPoisson(base_units=base_units, epsilon=epsilon, Z=charge_number)

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
R0 = 3.0
tor_period = 1
domain = domains.HollowTorus(a1=a1, a2=a2, R0=R0, sfl=False, pol_period=1, tor_period=tor_period)

# MHD equilibrium with circular concentric flux surfaces matching the domain (a = a2, same R0);
# the non-flat density and the ad hoc pressure keep n0/T0 = n0**2/p and n0/|B0|**2 of order one
B0 = 2.0
q0 = 1.71
q1 = 1.87
n1 = 2.0
n2 = 1.0
na = 0.2
beta = 0.179
p0 = 1.0
p1 = 0.1
p2 = 0.1
equil = equils.AdhocTorus(
    a=a2,
    R0=R0,
    B0=B0,
    q_kind=0,
    q0=q0,
    q1=q1,
    n1=n1,
    n2=n2,
    na=na,
    p_kind=1,
    p0=p0,
    p1=p1,
    p2=p2,
    beta=beta,
)

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

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

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.gyrokinetic_poisson.options = model.propagators.gyrokinetic_poisson.Options(
    which_geometry="toroidal",
    solver="pcg",
    precond=precond,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
from scipy.special import erf

from struphy.initial.base import GenericPerturbation

# The exact solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5).
# Its value at the Dirichlet boundary eta1 = 0, 1 is exp(-0.25 / w**2) ~ 1e-11, and it is periodic
# in eta2 and eta3 up to the same accuracy. Being localized, it excites the whole spectrum of the
# operator, so CG has to work for convergence. Both the solution and the right-hand side are 0-forms
# given as functions of the logical coordinates (given_in_basis="0"); the pushed-forward solution
# at x = F(eta) is phi(eta).
#
# The model solves (see GyrokineticPoisson.doc_pde)
#
#   1/(charge_number*epsilon**2) * n0/T0 * (phi - <phi>) - div(n0/|B0|**2 * (I - b0 b0^T) grad(phi)) = rho / epsilon,
#
# where <phi> is the average over eta2 and eta3 weighted with n0/T0 * sqrt(g). In logical coordinates the
# diffusion term reads -1/sqrt(g) * d_i(C^ij d_j phi), with C^ij = sqrt(g) * n0/|B0|**2 * (G^ij - b^i b^j),
# the inverse metric G^ij and the contravariant components b^i of b0.
w = 0.1
eta0 = (0.5, 0.5, 0.5)


def gaussian_derivatives(e1, e2, e3):
    """The Gaussian, its first and (pure) second derivatives and the mixed derivative d2 d3."""
    eta = (e1, e2, e3)
    g = np.exp(-sum((e - c) ** 2 for e, c in zip(eta, eta0)) / w**2)
    d = [-2 * (e - c) / w**2 * g for e, c in zip(eta, eta0)]
    dd = [(4 * (e - c) ** 2 / w**4 - 2 / w**2) * g for e, c in zip(eta, eta0)]
    d23 = 4 * (e2 - eta0[1]) * (e3 - eta0[2]) / w**4 * g
    return g, d, dd, d23


def exact_solution(e1, e2, e3):
    return gaussian_derivatives(e1, e2, e3)[0]


def equilibrium_coefficients(e1, e2):
    """sqrt(g), n0/T0 and C^11, C^22, C^23, C^33 (C^12 = C^13 = 0) as functions of (eta1, eta2).

    HollowTorus (sfl=False): r = a1 + (a2 - a1) * eta1, theta = 2*pi*eta2, phi = 2*pi*eta3 / tor_period,
    R = R0 + r*cos(theta), with the scale factors h1 = a2 - a1, h2 = 2*pi*r, h3 = 2*pi*R / tor_period.
    AdhocTorus (q_kind=0, p_kind=1): B0 = grad(psi) x grad(phi) - B0*R0 grad(phi) with
    dpsi/dr = B0 * r / (q * sqrt(1 - r**2/R0**2)), hence |B_pol| = dpsi/dr / R and |B_tor| = B0*R0 / R,
    q = q0 + (q1 - q0) * r**2 / a**2, n0 = na + (1 - na) * (1 - (r/a)**n1)**n2 and
    p = beta * B0**2 / 2 * (p0 - p1 * r**2/a**2 - p2 * r**4/a**4), T0 = p/n0.
    Only operations defined for complex numbers are used, such that the derivatives can be computed
    with the complex step.
    """
    da = a2 - a1
    r = a1 + da * e1
    theta = 2 * np.pi * e2
    R = R0 + r * np.cos(theta)
    q = q0 + (q1 - q0) * r**2 / a2**2
    n = na + (1 - na) * (1 - (r / a2) ** n1) ** n2
    p = beta * B0**2 / 2 * (p0 - p1 * r**2 / a2**2 - p2 * r**4 / a2**4)
    b_pol = B0 * r / (q * np.sqrt(1 - r**2 / R0**2)) / R
    b_tor = B0 * R0 / R
    abs_b_sq = b_pol**2 + b_tor**2

    h1, h2, h3 = da, 2 * np.pi * r, 2 * np.pi * R / tor_period
    sqrt_g = h1 * h2 * h3
    # contravariant components of the unit vector b0 = B0 / |B0| (b^1 = 0), signs from the
    # orientation of the mapping (only the sign of b^2 * b^3 enters)
    b2 = b_pol / np.sqrt(abs_b_sq) / h2
    b3 = b_tor / np.sqrt(abs_b_sq) / h3

    weight = sqrt_g * n / abs_b_sq
    c11 = weight / h1**2
    c22 = weight * (1 / h2**2 - b2**2)
    c23 = -weight * b2 * b3
    c33 = weight * (1 / h3**2 - b3**2)
    return sqrt_g, n**2 / p, c11, c22, c23, c33


def complex_step(f, x, h=1e-30):
    """Derivative f'(x) with the complex step, exact up to machine precision for analytic f."""
    return f(x + 1j * h).imag / h


def weighted_average(e1, e2, e3):
    """Average of the exact solution over eta2 and eta3, weighted with n0/T0 * sqrt(g).

    The weight is proportional to f(eta1) * R = f(eta1) * (R0 + r*cos(2*pi*eta2)), hence the average is
    g1 * int3 * (R0 * int2 + r * int_cos) / R0 with the integrals over [0, 1] of the Gaussians
    g2 and g3 (int2 = int3) and of g2 * cos(2*pi*eta2) (int_cos).
    """
    r = a1 + (a2 - a1) * e1
    g1 = np.exp(-((e1 - eta0[0]) ** 2) / w**2)
    int3 = w * np.sqrt(np.pi) * erf(0.5 / w)
    int_cos = -w * np.sqrt(np.pi) * np.exp(-((np.pi * w) ** 2)) * erf(0.5 / w - 1j * np.pi * w).real
    return g1 * int3 * (R0 * int3 + r * int_cos) / R0 + 0 * e2 + 0 * e3


def gyrokinetic_operator(e1, e2, e3):
    """Left-hand side of the gyrokinetic Poisson equation applied to the exact solution."""
    g, d, dd, d23 = gaussian_derivatives(e1, e2, e3)
    sqrt_g, n_over_t, c11, c22, c23, c33 = equilibrium_coefficients(e1, e2)
    dc11 = complex_step(lambda x: equilibrium_coefficients(x, e2)[2], e1)
    dc22 = complex_step(lambda x: equilibrium_coefficients(e1, x)[3], e2)
    dc23 = complex_step(lambda x: equilibrium_coefficients(e1, x)[4], e2)
    div = dc11 * d[0] + c11 * dd[0] + dc22 * d[1] + c22 * dd[1] + dc23 * d[2] + 2 * c23 * d23 + c33 * dd[2]
    return n_over_t / (charge_number * epsilon**2) * (g - weighted_average(e1, e2, e3)) - div / sqrt_g


def rhs_fun(e1, e2, e3):
    return epsilon * gyrokinetic_operator(e1, e2, e3)


rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="0")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    # post-processing and plots: pproc_gyrokinetic_poisson.py
    sim.run(profiling_activated=True, one_time_step=True)
