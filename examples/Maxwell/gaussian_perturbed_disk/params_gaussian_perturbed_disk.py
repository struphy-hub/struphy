"""1-form wave equation (Maxwell in 2D) on the perturbed disk with high-order (homotopy) polar splines of regularity r:
a Gaussian pulse of B_z = curl A next to the polar axis crosses the axis (the 1-form wave study of the polar FEEC talk
of R. Remmerswaal)."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--regularity", type=int, default=3, help="Regularity r of the polar splines at the axis.")
parser.add_argument("--degree", type=int, default=5, help="Spline degree p in eta1 and eta2.")
parser.add_argument("--cells", type=int, default=8, help="Number of radial cells N^x.")
parser.add_argument("--angular-cells", type=int, default=16, help="Number of angular cells N^y.")
parser.add_argument(
    "--enrichment",
    default="cochain_chain",
    choices=["cochain_chain", "cochain", "none"],
    help="Enrichment of the polar spaces (the talk compares cochain_chain with cochain).",
)
parser.add_argument("--final-time", type=float, default=1.0, help="Final time.")
parser.add_argument(
    "--cfl", type=float, default=0.9, help="Fraction of the leapfrog stability limit 2/sqrt(lambda_max)."
)
parser.add_argument("--test-run", action="store_true", help="Small resolution and a few time steps, for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.cells, args.angular_cells, args.degree, args.regularity, args.final_time = 4, 12, 3, 2, 0.05

name = f"1-form wave on the perturbed disk, polar splines p = {args.degree}, r = {args.regularity}"
description = f"""
Maxwell's equations dE/dt = curl B, dB/dt = -curl E in the cross-section of a cylinder, i.e. the 1-form wave
equation d^2 A / dt^2 + curl curl A = 0 for E = -dA/dt, B = curl A, on the perturbed disk
R = eta1 (1 + 0.2 eta1 cos(6 pi eta2)), with E x n = 0 on the boundary, E = 0 and
B_z = exp(-((x - 0.05)^2 + y^2) / (2 * 0.04^2)) - mean at t = 0.

Physics. The pulse of B_z = curl A is centred next to the axis of the cylinder (the pole of the polar coordinates),
so the wave crosses the coordinate singularity. The tensor-product 1-form space is not conforming there; the
high-order polar splines (R. Remmerswaal, homotopy construction, DerhamOptions(polar_splines=True, polar_regularity=r)) are
conforming for r >= 1 and C^(r-1) at the axis. The talk compares r = 0 (non-conforming) with higher regularities and
the enrichments "cochain" (gradients of polynomials only, too poor) and "cochain_chain" (with the homotopy images of
polynomial 2-forms). The mean of B_z is conserved (E x n = 0) and is subtracted, since it does not propagate.

Setup. PerturbedDiskCylinder(eps=0.2, k=3), {args.cells} x {args.angular_cells} x 1 cells, splines of degree
{args.degree}, polar regularity r = {args.regularity}, enrichment {args.enrichment}. Model Maxwell with the
symplectic leapfrog (algo="leapfrog") of the talk; the time step is {args.cfl} times the stability limit
2 / sqrt(lambda_max) of these spaces (computed with struphy.polar.homotopy).

Expected result. The pproc plots B_z at the first, middle and last snapshot and the energies; the discrete
electromagnetic energy is conserved up to the leapfrog oscillation of order dt^2.

References:
- R. Remmerswaal, High-order FEEC discretization on 2D and 3D domains with a disk-like coordinate singularity
  (draft); talk at the NMPP seminar 2026; PolarFEEC.jl (https://gitlab.mpcdf.mpg.de/rwr/polarfeec.jl).
"""

import logging

import numpy as np

from struphy import set_logging_level

set_logging_level(logging.INFO)

from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids
from struphy.initial.base import GenericPerturbation
from struphy.models import Maxwell
from struphy.polar.homotopy.demos.waves import WaveProblem, gaussian

base_units = BaseUnits()
model = Maxwell(base_units=base_units)
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True

# time step: a fraction of the leapfrog stability limit of these polar spaces
problem = WaveProblem(
    1,
    args.regularity,
    degree=args.degree,
    radial_cells=args.cells,
    angular="bspline",
    angular_cells=args.angular_cells,
    enrichment=args.enrichment,
)
dt = args.cfl / 0.9 * problem.dt
dt = args.final_time / np.ceil(args.final_time / dt)

env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)
time_opts = Time(dt=dt, Tend=args.final_time)
grid = grids.TensorProductGrid(num_elements=(args.cells, args.angular_cells, 1))
domain = domains.PerturbedDiskCylinder(R1=1.0, eps=0.2, k=3, Lz=1.0)
derham_opts = DerhamOptions(
    degree=(args.degree, args.degree, 1),
    bcs=(("free", "dirichlet"), None, None),
    polar_splines=True,
    polar_regularity=args.regularity,
    polar_enrichment=args.enrichment,
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

from struphy.linear_algebra.solver import SolverParameters

model.propagators.maxwell.options = model.propagators.maxwell.Options(
    algo="leapfrog",
    precond="PolarMassMatrixPreconditioner",
    solver_params=SolverParameters(tol=1e-12, maxiter=500),
)

# mean of the Gaussian over the cross-section (Gauss quadrature in the logical coordinates)
_x, _w = np.polynomial.legendre.leggauss(8)
_e1 = ((np.arange(64)[:, None] + (_x[None, :] + 1) / 2) / 64).ravel()
_w1 = np.tile(_w / 128, 64)
_e3 = np.array([0.5])
_X, _Y, _ = (c.reshape(_e1.size, _e1.size) for c in domain(_e1, _e1, _e3))
_detj = np.abs(domain.jacobian_det(_e1, _e1, _e3)).reshape(_e1.size, _e1.size)
_weights = np.outer(_w1, _w1) * _detj
MEAN = float(np.sum(_weights * gaussian(_X, _Y)) / np.sum(_weights))


def exact_initial_bz(x, y, z):
    """Initial B_z (physical) with zero mean."""
    return gaussian(x, y) - MEAN + 0.0 * z


def initial_b_logical_z(e1, e2, e3):
    """Logical 2-form component b_3 = (det(DF) DF^{-1} B)_3 = det(DF) B_z / L_z (finite at the axis)."""
    x, y, z = domain(e1, e2, e3)
    return domain.jacobian_det(e1, e2, e3) / domain.params["Lz"] * exact_initial_bz(x, y, z)


model.em_fields.b_field.add_perturbation(GenericPerturbation(initial_b_logical_z, given_in_basis="2", comp=2))

if __name__ == "__main__":
    # post-processing: pproc_gaussian_perturbed_disk.py
    sim.run()
