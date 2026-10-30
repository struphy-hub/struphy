"""Gyrokinetic Poisson equation (quasineutrality with the ion polarization density and adiabatic electrons) on a
torus in an ad hoc tokamak equilibrium, with a manufactured solution whose gradient is largest on the magnetic
axis. Compares C1 polar splines (no hole) with tensor-product splines on a torus with a small hole; the pproc
reports the max. relative error, also close to the axis."""

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
# The other flags select the variant (spline space and preconditioner), see the submit script;
# pproc_gaussian_on_axis_torus.py takes the same flags.
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
    ],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument(
    "--poloidal-solver",
    default="schur",
    choices=["schur", "lu", "ilu"],
    help="Poloidal solver of the polar preconditioners and projectors.",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[64, 66, 64], help="Number of elements per direction.")
parser.add_argument(
    "--hole-bc",
    default="free",
    choices=["free", "dirichlet"],
    help="Boundary condition at the hole of the tensor-product domain (default free; the solution does not vanish there).",
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

name = f"GyrokineticPoisson on a torus, {args.splines} splines, preconditioner {precond}"
description = f"""
Gyrokinetic Poisson equation (quasineutrality) with adiabatic electrons for the electrostatic potential phi
of a given gyrocenter charge density rho on a full torus of minor radius 1 and major radius 3 in an ad hoc
tokamak equilibrium, with a manufactured solution that is largest near the magnetic axis, comparing C1 polar
splines with tensor-product splines.

Physics. In gyrokinetic models the electrostatic potential follows from quasineutrality instead of
Poisson's equation: the ion gyrocenter density plus the ion polarization density equals the electron density.
In the long-wavelength limit the polarization density is proportional to div(n0 / B0^2 grad_perp phi), the
gyrokinetic counterpart of the polarization drift (Lee 1983). With adiabatic (Boltzmann) electrons, which respond
to the potential along the field lines, the electron density perturbation is proportional to
n0/Te (phi - <phi>); the flux-surface average <phi> is removed since the electrons cannot respond adiabatically
to the zonal (surface-averaged) potential. In Struphy units this is the GyrokineticPoisson equation
    1/(Z eps^2) n0/T0 (phi - <phi>) - div(n0/|B0|^2 grad_perp phi) = rho / eps,
with grad_perp = (I - b0 b0^T) grad, b0 = B0/|B0|, the gyrokinetic parameter eps (here 1) and the ion charge
number Z (here 1); it is the field equation of the DriftKineticElectrostaticAdiabatic model. n0, T0 = p/n0
(with the pressure p) and B0 are taken from the MHD equilibrium. The operator is strongly anisotropic, since
along b0 only the adiabatic term acts, and the average <phi> couples all points of a surface; both make it harder
to precondition than the Laplacian. phi lives in the spline space V0 of H1 of the discrete de Rham complex (FEEC).

The case tests the treatment of the coordinate singularity at the magnetic axis r = 0 of the torus,
the center of the polar coordinates (r, theta) of the cross-section: physical fields, such as the potential and
the electric field of a plasma column, are smooth there and in general have a non-zero value and gradient on the
axis. Tensor-product splines in (r, theta) are not smooth across the axis, so they are used on a domain with a
small hole and a boundary condition at the hole. C1 polar splines, linear combinations of the tensor-product
splines of the first two rings of control points around the pole, are C1-smooth across the axis and need no hole.
The manufactured solution is placed such that its gradient is largest on the axis, where the two approaches
differ most.

Setup. Ad hoc tokamak equilibrium (AdhocTorus, a=1, R0=3) with circular flux surfaces; comparing C1 polar
splines (IGAPolarTorus, no hole) with tensor-product splines on a torus with a hole (HollowTorus, a1=1e-2),
splines of degree 3; <phi> is the average over eta2 and eta3 (poloidal and toroidal angle), the flux-surface
average.
The manufactured solution is a Gaussian ring whose gradient is largest on the magnetic axis r=0, with a non-zero
flux-surface average, so that the average term is tested as well.
Homogeneous Dirichlet boundary conditions are set at r=1 (boundary condition {args.hole_bc} at the hole).
The linear system is solved with CG, preconditioner {precond} (poloidal solver {args.poloidal_solver}).

Expected result. The pproc script compares the computed phi and the projected right-hand side rho
with the exact fields and prints the max. relative errors, in the whole domain and close to the axis (r < 0.1),
where the tensor-product splines on the domain with a hole are expected to be less accurate than the polar
splines. No thresholds are asserted. The number of CG iterations shows the efficiency of the preconditioner.

References:
- W. W. Lee, "Gyrokinetic approach in particle simulation", Phys. Fluids 26, 556 (1983).
- D. Toshniwal, H. Speleers, R. R. Hiemstra and T. J. R. Hughes, "Multi-degree smooth polar splines: A
  framework for geometric modeling and isogeometric analysis", Comput. Methods Appl. Mech. Engrg. 316, 1005
  (2017).
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
import numpy as np

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

# Grid (polar splines need a multiple of 3 elements in eta2)
num_elements = tuple(args.num_elements)
degree = (3, 3, 3)
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, True))

# Geometry and Derham options
a = 1.0
R0 = 3.0
if args.splines == "polar":
    # the spline mapping has the resolution of the grid in the poloidal plane
    domain = domains.IGAPolarTorus(
        num_elements=num_elements[:2], degree=degree[:2], a=a, R0=R0, sfl=False, tor_period=1
    )
    derham_opts = DerhamOptions(
        degree=degree,
        bcs=(("free", "dirichlet"), None, None),
        polar_splines=True,
        polar_poloidal_solver=args.poloidal_solver,
    )
else:
    a1 = 1e-2
    domain = domains.HollowTorus(a1=a1, a2=a, R0=R0, sfl=False, pol_period=1, tor_period=1)
    derham_opts = DerhamOptions(degree=degree, bcs=((args.hole_bc, "dirichlet"), None, None))

# MHD equilibrium with circular concentric flux surfaces matching the domain (same a and R0);
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
    a=a,
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

# options of the polar preconditioners
precond_params = {}
if args.splines == "polar" and precond in ("PolarMassMatrixPreconditioner", "StiffnessPreconditioner"):
    precond_params["poloidal_solver"] = args.poloidal_solver

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.gyrokinetic_poisson.options = model.propagators.gyrokinetic_poisson.Options(
    which_geometry="toroidal",
    solver="pcg",
    precond=precond,
    precond_params=precond_params,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import sympy as sym

from struphy.initial.base import GenericPerturbation

# The model solves (see GyrokineticPoisson.doc_pde)
#
#   1/(charge_number*epsilon**2) * n0/T0 * (phi - <phi>) - div(n0/|B0|**2 * (I - b0 b0^T) grad(phi)) = rho / epsilon,
#
# where <phi> is the average over eta2 and eta3 (the poloidal angle theta and the toroidal angle) weighted with
# n0/T0 * sqrt(g), sqrt(g) ~ r * R. The exact solution is phi = g(R, z) * h: a Gaussian ring g of width w in the
# poloidal plane (R = sqrt(x**2 + y**2)), centered at distance w / sqrt(2) from the magnetic axis (R0, 0), where its
# gradient is largest, so the gradient is largest on the axis r = 0, and the toroidal modulation h = 1 + x / (2 R).
# The weight is a function of r only, hence <phi>(r) = int g R dtheta / int R dtheta (computed by quadrature in theta;
# h averages to 1).
# phi is negligible at r = 1 (Dirichlet boundary). Tensor-product splines cannot represent a non-zero gradient
# across the axis (they live on a domain with a hole and a boundary condition there, see --hole-bc); polar
# splines can. The right-hand side rho is computed symbolically from the equilibrium in Cartesian coordinates
# (checked against the struphy equilibrium below); phi and rho are given as functions of the Cartesian
# coordinates (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
w = 0.2
x0 = w / np.sqrt(2.0)
R_ = sym.sqrt(x_**2 + y_**2)
g_sym = sym.exp(-((R_ - R0 - x0) ** 2 + z_**2) / w**2)
phi_sym = g_sym * (1 + x_ / (2 * R_))

# AdhocTorus (q_kind=0, p_kind=1) in Cartesian coordinates: B0 = grad(psi) x grad(phi) - B0*R0 grad(phi), with
# dpsi/dr = B0 * r / (q * sqrt(1 - r**2/R0**2)), r = sqrt((R - R0)**2 + z**2), q = q0 + (q1 - q0) * r**2 / a**2,
# i.e. B_R = -psi_z / R, B_phi = -B0 * R0 / R, B_z = psi_R / R; n0 = na + (1 - na) * (1 - (r/a)**n1)**n2 and
# p = B0**2 * beta / 2 * (p0 - p1 * r**2/a**2 - p2 * r**4/a**4)
r2 = (R_ - R0) ** 2 + z_**2
q = q0 + (q1 - q0) * r2 / a**2
dpsi_over_r = B0 / (q * sym.sqrt(1 - r2 / R0**2))
B_R = -dpsi_over_r * z_ / R_
B_phi = -B0 * R0 / R_
B_z = dpsi_over_r * (R_ - R0) / R_
n_sym = na + (1 - na) * (1 - (r2 / a**2) ** (n1 / 2)) ** n2
p_sym = B0**2 * beta / 2 * (p0 - p1 * r2 / a**2 - p2 * r2**2 / a**4)
B_sym = [B_R * x_ / R_ - B_phi * y_ / R_, B_R * y_ / R_ + B_phi * x_ / R_, B_z]


def gyrokinetic_rho(phi_sym, avg_sym, n_sym, p_sym, B_sym):
    """rho = epsilon * (n0/T0 / (Z eps**2) * (phi - <phi>) - div(n0/|B0|**2 * (I - b0 b0^T) grad(phi)))."""
    X = (x_, y_, z_)
    abs_B_sq = sum(b**2 for b in B_sym)
    grad = [sym.diff(phi_sym, xi) for xi in X]
    b_dot_grad = sum(b * gr for b, gr in zip(B_sym, grad)) / abs_B_sq
    flux = [n_sym / abs_B_sq * (gr - b * b_dot_grad) for b, gr in zip(B_sym, grad)]
    div = sum(sym.diff(f, xi) for f, xi in zip(flux, X))
    return epsilon * (n_sym**2 / p_sym / (charge_number * epsilon**2) * (phi_sym - avg_sym) - div)


# rho without the average, which is added numerically below
rho_sym = gyrokinetic_rho(phi_sym, 0, n_sym, p_sym, B_sym)
phi_lam = sym.lambdify((x_, y_, z_), phi_sym, "numpy")
rho_lam = sym.lambdify((x_, y_, z_), rho_sym, "numpy")
n_over_t_lam = sym.lambdify((x_, y_, z_), n_sym**2 / p_sym, "numpy")

# <phi> depends on r only: tabulated once (trapezoidal rule in theta, exponentially accurate for periodic
# integrands) and interpolated
_theta = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
_r_tab = np.linspace(0.0, a, 2001)
_R_tab = R0 + _r_tab[:, None] * np.cos(_theta)
_g_tab = np.exp(-((_r_tab[:, None] * np.cos(_theta) - x0) ** 2 + (_r_tab[:, None] * np.sin(_theta)) ** 2) / w**2)
_avg_tab = np.sum(_g_tab * _R_tab, axis=1) / np.sum(_R_tab, axis=1)


def average(x, y, z):
    """Flux-surface average <phi>(r) = int g R dtheta / int R dtheta at the points (x, y, z)."""
    r = np.sqrt((np.sqrt(x**2 + y**2) - R0) ** 2 + z**2)
    return np.interp(r, _r_tab, _avg_tab)


# the symbolic equilibrium must match the struphy equilibrium (points in the torus)
_rng = np.random.default_rng(0)
_r, _t, _p = _rng.uniform(1e-3, 0.99, 50), _rng.uniform(0.0, 2 * np.pi, 50), _rng.uniform(0.0, 2 * np.pi, 50)
_pts = ((R0 + _r * np.cos(_t)) * np.cos(_p), (R0 + _r * np.cos(_t)) * np.sin(_p), _r * np.sin(_t))
_check = [
    (sym.lambdify((x_, y_, z_), n_sym, "numpy"), equil.n_xyz),
    (sym.lambdify((x_, y_, z_), p_sym, "numpy"), equil.p_xyz),
] + [(sym.lambdify((x_, y_, z_), B_sym[k], "numpy"), lambda x, y, z, k=k: equil.b_xyz(x, y, z)[k]) for k in range(3)]
for _sym_fun, _equil_fun in _check:
    assert np.allclose(_sym_fun(*_pts), _equil_fun(*_pts), rtol=1e-12, atol=1e-12), "Symbolic equilibrium mismatch."


def exact_solution(x, y, z):
    return phi_lam(x, y, z) + 0.0 * x


def rhs_fun(x, y, z):
    avg = average(x, y, z)
    return rho_lam(x, y, z) - epsilon * n_over_t_lam(x, y, z) / (charge_number * epsilon**2) * avg + 0.0 * x


def distance_to_axis(x, y, z):
    return np.sqrt((np.sqrt(x**2 + y**2) - R0) ** 2 + z**2)


model.em_fields.source.add_perturbation(GenericPerturbation(rhs_fun, given_in_basis="physical"))


if __name__ == "__main__":
    # post-processing and plots: pproc_gaussian_on_axis_torus.py
    sim.run(profiling_activated=True, one_time_step=True)
