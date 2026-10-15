"""Curl-curl problem curl curl A = J on a rounded square with high-order (homotopy) polar splines of regularity r, the
Dirichlet case (A x n = 0) of PolarFEEC.jl example 05; the pproc reports the L2 error of curl A (rate p)."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--degree", type=int, default=3, help="Spline degree p in eta1 and eta2.")
parser.add_argument("--regularity", type=int, default=None, help="Regularity r of the polar splines (default: p).")
parser.add_argument("--cells", type=int, default=6, help="Number n of radial cells (6n angular cells).")
parser.add_argument("--sigma", type=float, default=1e-10, help="Coefficient of the stabilizing mass term.")
parser.add_argument("--test-run", action="store_true", help="Small resolution (n = 4, p = 2), for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.cells, args.degree = 4, 2
if args.regularity is None:
    args.regularity = args.degree

name = f"Curl-curl on a rounded square, homotopy polar splines (p = {args.degree}, r = {args.regularity})"
description = f"""
Curl-curl problem (curl Phi, curl A) + sigma (Phi, A) = (Phi, J) for all Phi in H0(curl) on the rounded square
g > 0, g = 1 - x^2 - y^2 - eps Re((x + i y)^4), eps = 0.15, with A x n = 0 on the boundary and high-order polar
splines of regularity r = {args.regularity} at the centre.

Physics. The curl-curl problem determines the vector potential A of a current J (magnetostatics); A is unique only
up to gradients, so the gauge-independent field B = curl A is compared. J = curl* B = (d_y B, -d_x B) with the
manufactured B = sin(2x) cos(y) (odd in x, zero mean: the compatibility condition of the Dirichlet problem). The
domain is parametrized by a disk-like map with its pole at the centre of the cross-section; the polar splines of
R. Remmerswaal (homotopy construction) are conforming and C^(r-1) at the pole. In 2D the paper solves this problem
with one 2-form mass solve and the homotopy operator ("a free two-dimensional lunch", struphy.polar.homotopy); here
the stabilized stiffness system of the CurlCurl model is solved (sigma = {args.sigma}), which gives the same curl A.

Setup. RoundedSquareCylinder(eps=0.15), {args.cells} x {6 * args.cells} x 1 cells, degree {args.degree},
DerhamOptions(polar_method="homotopy", polar_regularity={args.regularity}), Dirichlet conditions at eta1 = 1.

Expected result. The L2 error of curl A converges like h^p (PolarFEEC.jl example 05).
"""

import logging

import numpy as np

from struphy import set_logging_level

set_logging_level(logging.INFO)

from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids
from struphy.models import CurlCurl

model = CurlCurl(base_units=BaseUnits())
model.em_fields.e_field.save_data = True

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)
grid = grids.TensorProductGrid(num_elements=(args.cells, 6 * args.cells, 1))
EPS = 0.15
domain = domains.RoundedSquareCylinder(a=1.0, eps=EPS, Lz=1.0)
derham_opts = DerhamOptions(
    degree=(args.degree, args.degree, 1),
    bcs=(("free", "dirichlet"), None, None),
    polar_splines=True,
    polar_method="homotopy",
    polar_regularity=args.regularity,
    nquads=(args.degree + 3, args.degree + 3, 2),
)

sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=Time(),
    domain=domain,
    equil=None,
    grid=grid,
    derham_opts=derham_opts,
)

from struphy.linear_algebra.solver import SolverParameters

model.propagators.curl_curl.options = model.propagators.curl_curl.Options(
    sigma=args.sigma,
    solver="pcg",
    precond="StiffnessPreconditioner",
    solver_params=SolverParameters(tol=1e-12, maxiter=5000, info=True, recycle=False),
)


def exact_b(x, y, z):
    """B_z = curl A of the Dirichlet problem."""
    return np.sin(2 * x) * np.cos(y) + 0.0 * z


def current(x, y):
    """J = (d_y B, -d_x B)."""
    return -np.sin(2 * x) * np.sin(y), -2 * np.cos(2 * x) * np.cos(y)


def _current_logical(i):
    """Logical 1-form component i of J: (DF^T J)_i."""

    def component(e1, e2, e3):
        x, y, _ = domain(e1, e2, e3)
        DF = np.asarray(domain.jacobian(e1, e2, e3)).reshape((3, 3) + np.shape(x))
        jx, jy = current(x, y)
        return DF[0, i] * jx + DF[1, i] * jy

    return component


# the load is integrated against the Dirichlet test functions (a source in V1 would lose its boundary coefficients)
model.propagators.curl_curl.j = tuple(_current_logical(i) for i in range(3))

if __name__ == "__main__":
    # post-processing: pproc_polar_rounded_square.py
    sim.run(one_time_step=True)
