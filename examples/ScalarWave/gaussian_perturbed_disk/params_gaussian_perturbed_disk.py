"""0-form wave equation on the perturbed disk with high-order (homotopy) polar splines of regularity r: a Gaussian
pulse next to the polar axis crosses the axis (the 0-form wave study of the polar FEEC talk of R. Remmerswaal)."""

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
parser.add_argument("--final-time", type=float, default=1.0, help="Final time.")
parser.add_argument(
    "--cfl", type=float, default=0.9, help="Fraction of the leapfrog stability limit 2/sqrt(lambda_max)."
)
parser.add_argument("--test-run", action="store_true", help="Small resolution and a few time steps, for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.cells, args.angular_cells, args.degree, args.regularity, args.final_time = 4, 12, 3, 2, 0.05

name = f"0-form wave on the perturbed disk, polar splines p = {args.degree}, r = {args.regularity}"
description = f"""
0-form (scalar) wave equation d^2 phi / dt^2 - Laplace phi = 0 on the perturbed disk
R = eta1 (1 + 0.2 eta1 cos(6 pi eta2)), with phi = 0 on the boundary and a Gaussian pulse
exp(-((x - 0.05)^2 + y^2) / (2 * 0.04^2)) at rest next to the polar axis.

Physics. The pulse is centred next to the axis of the cylinder (the pole of the polar coordinates of the
cross-section), so the expanding wave crosses the coordinate singularity. Tensor-product splines are discontinuous
there; the high-order polar splines (R. Remmerswaal, homotopy construction, DerhamOptions(polar_splines=True, polar_regularity=r))
are C^r at the axis. The talk compares r = 0 (continuity) with higher regularities: low regularity leaves spurious
waves and kinks at the axis, and limits the convergence of nearly resolved waves.

Setup. PerturbedDiskCylinder(eps=0.2, k=3), {args.cells} x {args.angular_cells} x 1 cells, splines of degree
{args.degree}, polar regularity r = {args.regularity}, Dirichlet condition at eta1 = 1. Model ScalarWave with the
symplectic leapfrog (velocity Verlet) of the talk; the time step is {args.cfl} times the stability limit
2 / sqrt(lambda_max) of these spaces (computed with struphy.polar.homotopy).

Expected result. The pproc plots phi at the first, middle and last snapshot and the energy; the staggered discrete
energy is conserved. For the convergence in h of the talk (maximum error during the final 10%), run several
resolutions and regularities (see struphy.polar.homotopy.demos.waves).

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
from struphy.models import ScalarWave
from struphy.polar.homotopy.demos.waves import WaveProblem, gaussian

base_units = BaseUnits()
model = ScalarWave(base_units=base_units)
model.em_fields.phi.save_data = True
model.em_fields.psi.save_data = True

# time step: a fraction of the leapfrog stability limit of these polar spaces
problem = WaveProblem(
    0,
    args.regularity,
    degree=args.degree,
    radial_cells=args.cells,
    angular="bspline",
    angular_cells=args.angular_cells,
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

model.propagators.wave.options = model.propagators.wave.Options(
    precond="PolarMassMatrixPreconditioner",
    solver_params=SolverParameters(tol=1e-12, maxiter=500),
)


def initial_phi(x, y, z):
    return gaussian(x, y) + 0.0 * z


model.em_fields.phi.add_perturbation(GenericPerturbation(initial_phi, given_in_basis="physical"))

if __name__ == "__main__":
    # post-processing: pproc_gaussian_perturbed_disk.py
    sim.run()
