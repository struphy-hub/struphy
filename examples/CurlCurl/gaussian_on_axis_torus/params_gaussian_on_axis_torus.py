"""Curl-curl problem curl curl E + sigma E = J (implicit time step of Maxwell's equations, eddy-current or
magnetostatic problems) on a torus with a manufactured field whose gradient is largest on the magnetic axis.
Compares C1 polar splines (no hole) with tensor-product splines on a torus with a small hole; the pproc reports
the max. relative error, also close to the axis."""

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
# The other flags select the variant (spline space and preconditioner), see the submit script; pproc_gaussian_on_axis_torus.py
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
        "HiptmairXu",
    ],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument(
    "--poloidal-solver",
    default="schur",
    choices=["schur", "lu", "ilu"],
    help="Poloidal solver of the polar preconditioners.",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 66, 64], help="Number of elements per direction.")
parser.add_argument(
    "--hole-bc",
    default="dirichlet",
    choices=["free", "dirichlet"],
    help="Boundary condition at the hole of the tensor-product domain (default dirichlet; the natural boundary condition contradicts the solution, whose curl is largest on the axis, and the error is amplified by 1/sigma).",
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

name = f"Curl-curl on a torus, {args.splines} splines, preconditioner {precond}"
description = f"""
Curl-curl problem curl curl E + sigma E = J for the vector field E (e.g. the electric field) with a given
source J on a full torus of minor radius 1 and major radius 4, with a manufactured solution that is largest near
the magnetic axis, comparing C1 polar splines with tensor-product splines.

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

The case tests the treatment of the coordinate singularity at the magnetic axis r = 0 of the torus,
the center of the polar coordinates (r, theta) of the cross-section: physical fields, such as the potential and
the electric field of a plasma column, are smooth there and in general have a non-zero value and gradient on the
axis. Tensor-product splines in (r, theta) are not smooth across the axis, so they are used on a domain with a
small hole and a boundary condition at the hole. C1 polar splines, linear combinations of the tensor-product
splines of the first two rings of control points around the pole, are C1-smooth across the axis and need no hole.
The manufactured solution is placed such that its gradient is largest on the axis, where the two approaches
differ most.

Setup. Comparing C1 polar splines (IGAPolarTorus, no hole) with tensor-product splines on a torus with a hole
(HollowTorus, a1=1e-2), splines of degree 3, sigma = 1e-2. The manufactured solution E = g c (constant Cartesian
vector c) has a Gaussian ring g whose gradient is largest on the magnetic axis r=0.
Homogeneous Dirichlet boundary conditions (vanishing tangential E) are set at r=1
(boundary condition {args.hole_bc} at the hole).
The linear system is solved with CG, preconditioner {precond} (poloidal solver {args.poloidal_solver}).

Expected result. The pproc script compares the computed E and the projected right-hand side J
with the exact fields and prints the max. relative errors, in the whole domain and close to the axis (r < 0.1),
where the tensor-product splines on the domain with a hole are expected to be less accurate than the polar
splines. No thresholds are asserted. The number of CG iterations shows the efficiency of the preconditioner.

References:
- D. Toshniwal, H. Speleers, R. R. Hiemstra and T. J. R. Hughes, "Multi-degree smooth polar splines: A
  framework for geometric modeling and isogeometric analysis", Comput. Methods Appl. Mech. Engrg. 316, 1005
  (2017).
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

# Grid
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

# coefficient of the mass term; small values make the kernel of the curl (the gradients) hard to precondition
sigma = 1e-2

# options of the polar preconditioners
precond_params = {}
if args.splines == "polar" and precond in ("PolarMassMatrixPreconditioner", "StiffnessPreconditioner"):
    precond_params["poloidal_solver"] = args.poloidal_solver

solver_params = SolverParameters(tol=1e-8, maxiter=10000, info=True, recycle=False)
model.propagators.curl_curl.options = model.propagators.curl_curl.Options(
    sigma=sigma,
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

# The exact solution is the vector field E = g c in Cartesian coordinates, with a constant vector c and
# g = Gaussian ring of width w in the poloidal plane (R = sqrt(x**2 + y**2)), centered at distance w / sqrt(2) from
# the magnetic axis (R0, 0), where its gradient is largest, so the gradient is largest on the axis r = 0, times
# a toroidal modulation 1 + x / (2 R).
# It is negligible at r = 1, hence satisfies the boundary conditions up to that accuracy.
# Tensor-product splines cannot represent non-zero fields and gradients across the axis (they live on a domain
# with a hole and a boundary condition there, see --hole-bc); polar splines can.
# The right-hand side J = curl curl E + sigma E is computed symbolically. Both are given in Cartesian
# components (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
w = 0.2
x0 = w / np.sqrt(2.0)
R_ = sym.sqrt(x_**2 + y_**2)
g = sym.exp(-((R_ - R0 - x0) ** 2 + z_**2) / w**2) * (1 + x_ / (2 * R_))
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


def distance_to_axis(x, y, z):
    return np.sqrt((np.sqrt(x**2 + y**2) - R0) ** 2 + z**2)


for comp in range(3):
    model.em_fields.source.add_perturbation(GenericPerturbation(rhs(comp), given_in_basis="physical", comp=comp))


if __name__ == "__main__":
    # post-processing and plots: pproc_gaussian_on_axis_torus.py
    sim.run(profiling_activated=True, one_time_step=True)
