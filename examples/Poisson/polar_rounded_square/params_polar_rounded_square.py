"""Poisson equation -Laplace phi = rho on a rounded square with high-order (homotopy) polar splines of regularity r.
Manufactured solution of PolarFEEC.jl example 05; the pproc reports the L2 error, convergence_polar_rounded_square.py
runs the h-convergence study (rate p + 1)."""

# -----------------------------
# Description of the simulation
# -----------------------------

import argparse
from pathlib import Path

# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--degree", type=int, default=3, help="Spline degree p in eta1 and eta2.")
parser.add_argument(
    "--regularity", type=int, default=None, help="Regularity r of the polar splines at the pole (default: p)."
)
parser.add_argument("--cells", type=int, default=6, help="Number n of radial cells (6n angular cells).")
parser.add_argument(
    "--precond",
    default="StiffnessPreconditioner",
    choices=["None", "MassMatrixPreconditioner", "StiffnessPreconditioner"],
    help="Preconditioner of the CG solver.",
)
parser.add_argument("--test-run", action="store_true", help="Small resolution (n = 4, p = 2), for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.cells, args.degree = 4, 2
if args.regularity is None:
    args.regularity = args.degree

precond = None if args.precond == "None" else args.precond

name = f"Poisson on a rounded square, homotopy polar splines (p = {args.degree}, r = {args.regularity})"
description = f"""
Poisson equation -Laplace phi = rho on the rounded square g > 0, g = 1 - x^2 - y^2 - eps Re((x + i y)^4),
eps = 0.15, with high-order polar splines of regularity r = {args.regularity} at the centre.

Physics. Poisson's equation determines the electrostatic potential phi of a charge density rho. The domain is
parametrized by a disk-like map whose logical edge eta1 = 0 collapses to the centre of the cross-section (the axis
of the cylinder); tensor-product splines are not smooth there. The polar splines of R. Remmerswaal (homotopy
construction, see struphy.polar.homotopy) replace the r + 1 innermost rings of coefficients by projected polynomials
of degree r: the 0-forms are C^r at the centre, conforming, and form a de Rham complex with commuting projections.

Setup. RoundedSquareCylinder(eps=0.15), {args.cells} x {6 * args.cells} x 1 cells, splines of degree {args.degree}
(one periodic cell in eta3: a 2D problem), DerhamOptions(polar_splines=True,
polar_regularity={args.regularity}), homogeneous Dirichlet conditions at eta1 = 1. The manufactured solution of
PolarFEEC.jl example 05 is phi = g (1 + sin(2x + y)), which vanishes on the boundary; rho = -Laplace phi.
CG with preconditioner {precond}.

Expected result. The pproc script prints the L2 error of phi; the error converges like h^(p + 1)
(convergence_polar_rounded_square.py), as with the homotopy solver of PolarFEEC.jl.

References:
- R. Remmerswaal, High-order FEEC discretization on 2D and 3D domains with a disk-like coordinate singularity
  (draft), and the Julia package PolarFEEC.jl (https://gitlab.mpcdf.mpg.de/rwr/polarfeec.jl).
- P. Jiang et al., Polar spline discretizations with high-order regularity at the origin, J. Comput. Phys. 564,
  115129 (2026).
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

# ------------------
# Import Struphy API
# ------------------
from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids
from struphy.models import Poisson

base_units = BaseUnits()
model = Poisson(base_units=base_units)
model.em_fields.phi.save_data = True
model.em_fields.source.save_data = True

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)
time_opts = Time()

num_elements = (args.cells, 6 * args.cells, 1)
degree = (args.degree, args.degree, 1)
grid = grids.TensorProductGrid(num_elements=num_elements)

EPS = 0.15
domain = domains.RoundedSquareCylinder(a=1.0, eps=EPS, Lz=1.0)
derham_opts = DerhamOptions(
    degree=degree,
    bcs=(("free", "dirichlet"), None, None),
    polar_splines=True,
    polar_regularity=args.regularity,
    nquads=(args.degree + 3, args.degree + 3, 2),
)

sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=None,
    grid=grid,
    derham_opts=derham_opts,
)

# ------------------
# Propagator options
# ------------------
from struphy.linear_algebra.solver import SolverParameters

model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver="pcg",
    precond=precond,
    solver_params=SolverParameters(tol=1e-12, maxiter=5000, info=True, recycle=False),
)

# ------------------
# Initial conditions
# ------------------
import numpy as np

from struphy.initial.base import GenericPerturbation


def level(x, y):
    """g, its gradient and its Laplacian (-4, since Re((x + i y)^4) is harmonic)."""
    g = 1 - x**2 - y**2 - EPS * (x**4 - 6 * x**2 * y**2 + y**4)
    return g, (-2 * x - 4 * EPS * (x**3 - 3 * x * y**2), -2 * y - 4 * EPS * (y**3 - 3 * x**2 * y)), -4.0


def exact_solution(x, y, z):
    g, _, _ = level(x, y)
    return g * (1 + np.sin(2 * x + y)) + 0.0 * z


def rhs_fun(x, y, z):
    g, (gx, gy), lap_g = level(x, y)
    s, sx, sy, lap_s = 1 + np.sin(2 * x + y), 2 * np.cos(2 * x + y), np.cos(2 * x + y), -5 * np.sin(2 * x + y)
    return -(g * lap_s + s * lap_g + 2 * (gx * sx + gy * sy)) + 0.0 * z


# The source variable is saved for plotting. The solver gets the source as a callable (a logical 0-form proxy): its
# L2 load is integrated against the test functions with homogeneous Dirichlet conditions, whereas a source in the
# space V0 would have its boundary coefficients removed (rho does not vanish on the boundary).
model.em_fields.source.add_perturbation(GenericPerturbation(rhs_fun, given_in_basis="physical"))


def rhs_logical(e1, e2, e3):
    return rhs_fun(*domain(e1, e2, e3))


model.propagators.poisson.rho = rhs_logical


if __name__ == "__main__":
    # post-processing: pproc_polar_rounded_square.py
    sim.run(one_time_step=True)
