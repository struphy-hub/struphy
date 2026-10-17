"""3D 1-form wave equation (Maxwell) in a stellarator-like tube with high-order (homotopy) polar splines: a localized
magnetic pulse next to the magnetic axis (the 3D wave study of the polar FEEC talk of R. Remmerswaal)."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--regularity", type=int, default=2, help="Regularity r of the polar splines at the axis.")
parser.add_argument("--degree", type=int, default=4, help="Spline degree p in all directions.")
parser.add_argument("--n", type=int, default=5, help="Mesh parameter: n x 2n x 8n cells.")
parser.add_argument("--final-time", type=float, default=0.2, help="Final time.")
parser.add_argument("--dt", type=float, default=2e-3, help="Time step (leapfrog: below 2 / sqrt(lambda_max)).")
parser.add_argument("--test-run", action="store_true", help="Small resolution and a few time steps, for the tests.")
args, _ = parser.parse_known_args()
if args.test_run:
    args.n, args.degree, args.regularity, args.final_time, args.dt = 3, 2, 1, 0.004, 0.002

name = f"3D 1-form wave in a stellarator-like tube, polar splines p = {args.degree}, r = {args.regularity}"
description = f"""
Maxwell's equations dE/dt = curl B, dB/dt = -curl E (the 1-form wave equation for E = -dA/dt, B = curl A) in the
stellarator-like tube of the 3D wave and curl-curl studies of the talk: a helical magnetic axis with three periods,
a three-petal cross-section of radius 0.3 that turns once around the torus. E x n = 0 on the boundary.

Physics. The initial field B = curl A_0 is a magnetic pulse next to the magnetic axis (the line eta1 = 0 of the polar
coordinates), A_0 = a(x) grad(zeta)-like with the Gaussian a of width 0.04 * 0.3 around the cross-section point of
logical coordinates (0.05, 0) and of width 0.1 along the axis (as the pulse of the talk). The wave crosses the axis,
where the tensor-product splines are not conforming; the high-order polar splines (DerhamOptions(polar_splines=
True, polar_regularity=r)) are the total complex of the 2D polar splines of the cross-section and the splines along the axis.
Since B_0 is the exact curl of a 1-form given as logical components, its commuting projection is the discrete curl
of the projection of A_0: div B = 0 holds exactly in polar coefficients.

Setup. HelicalTube("stellarator"), {args.n} x {2 * args.n} x {8 * args.n} cells, splines of degree {args.degree},
polar regularity r = {args.regularity}, Maxwell with the symplectic leapfrog (algo="leapfrog"), dt = {args.dt}.

Expected result. The pproc plots |B| on the poloidal cut through the pulse at the first, middle and last snapshot,
and the energies. The talk compares r = 1, ..., 4 for degree-4 splines (convergence in h of the maximum error over
the final 10%; the resolutions n = 8 ... 32 are long runs).

Not stated in the talk and chosen here: the initial field is curl A_0 of the Gaussian pulse (the talk uses the
L2-orthogonal projection of a 2-form pulse onto the range of curl).

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

base_units = BaseUnits()
model = Maxwell(base_units=base_units)
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True

num_elements = (args.n, 2 * args.n, 8 * args.n)
degree = (args.degree,) * 3
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)
time_opts = Time(dt=args.dt, Tend=args.final_time)
grid = grids.TensorProductGrid(num_elements=num_elements)
domain = domains.HelicalTube("stellarator", num_elements=(8, 24, 64), degree=(4, 4, 4))
derham_opts = DerhamOptions(
    degree=degree,
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

model.propagators.maxwell.options = model.propagators.maxwell.Options(
    algo="leapfrog",
    precond="PolarMassMatrixPreconditioner",
    solver_params=SolverParameters(tol=1e-10, maxiter=500),
)

# ------------------
# Initial condition: B_0 = curl A_0, A_0 = (0, 0, a) in logical 1-form components
# ------------------
g = domain.geometry
WIDTH, AXIAL_WIDTH = 0.04 * 0.3, 0.1


def section(e1, e2):
    radius = e1 * (1 + g["eps"] * e1 * np.cos(2 * np.pi * g["k"] * e2))
    return radius * g["a"] * np.cos(2 * np.pi * e2), radius * g["b"] * np.sin(2 * np.pi * e2)


CENTER = section(0.05, 0.0)
_speed = 2 * np.pi * np.sqrt((g["R0"] + g["delta"]) ** 2 + (g["periods"] * g["delta_z"]) ** 2)


def potential(e1, e2, e3):
    """Logical component a_3 of A_0: the Gaussian pulse (cross-section and arc length along the axis)."""
    x2, y2 = section(e1, e2)
    arclength = _speed * (np.mod(e3 + 0.5, 1.0) - 0.5)
    return np.exp(
        -((x2 - CENTER[0]) ** 2 + (y2 - CENTER[1]) ** 2) / (2 * WIDTH**2) - arclength**2 / (2 * AXIAL_WIDTH**2)
    )


def _derivative(i, h=1e-7):
    def d(e1, e2, e3):
        plus, minus = [e1, e2, e3], [e1, e2, e3]
        plus[i] = plus[i] + h
        minus[i] = minus[i] - h
        return (potential(*plus) - potential(*minus)) / (2 * h)

    return d


# curl of (0, 0, a) in logical components: (d_2 a, -d_1 a, 0)
_d1, _d2 = _derivative(0), _derivative(1)
model.em_fields.b_field.add_perturbation(
    GenericPerturbation(lambda e1, e2, e3: _d2(e1, e2, e3), given_in_basis="2", comp=0)
)
model.em_fields.b_field.add_perturbation(
    GenericPerturbation(lambda e1, e2, e3: -_d1(e1, e2, e3), given_in_basis="2", comp=1)
)

if __name__ == "__main__":
    # post-processing: pproc_gaussian_helical_tube.py
    sim.run()
