# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse

# Command line options
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# The other flags select the variant (spline space and preconditioner), see the submit script.
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
parser.add_argument(
    "--splines",
    default="polar",
    choices=["polar", "tensor"],
    help="C1 polar splines on a disc (IGAPolarCylinder) or tensor-product splines on a cylinder with a hole (HollowCylinder).",
)
parser.add_argument("--precond", default="None", help="Name of the preconditioner (None for no preconditioner).")
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
args, _ = parser.parse_known_args()

precond = None if args.precond == "None" else args.precond

name = f"Poisson strong scaling on a cylinder, {args.splines} splines, preconditioner {precond}"
description = f"""
Strong scaling test for the Poisson equation on a cylinder of radius 1 and length 4, comparing C1 polar splines
(IGAPolarCylinder, no hole) with tensor-product splines on a cylinder with a hole (HollowCylinder, a1=1e-2).
The manufactured solution is a Gaussian whose gradient is largest on the axis r=0.
Homogeneous Dirichlet boundary conditions are set at r=1 (boundary condition {args.hole_bc} at the hole).
The linear system is solved with CG, preconditioner {precond} (poloidal solver {args.poloidal_solver}).
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
    sim_folder=f"sim_{args.id:02d}",
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
Lz = 4.0
if args.splines == "polar":
    # the spline mapping has the resolution of the grid in the poloidal plane
    domain = domains.IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], a=a, Lz=Lz)
    derham_opts = DerhamOptions(
        degree=degree,
        bcs=(("free", "dirichlet"), None, None),
        polar_splines=True,
        polar_poloidal_solver=args.poloidal_solver,
    )
else:
    a1 = 1e-2
    domain = domains.HollowCylinder(a1=a1, a2=a, Lz=Lz)
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

# options of the polar preconditioners
precond_params = {}
if args.splines == "polar" and precond in ("PolarMassMatrixPreconditioner", "StiffnessPreconditioner"):
    precond_params["poloidal_solver"] = args.poloidal_solver

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
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

# The exact solution is phi = g(x, y) * h(z): a Gaussian g of width w in the cross-section, centered at distance
# w / sqrt(2) from the axis, where its gradient is largest, so the gradient is largest on the axis r = 0,
# and a periodic modulation h along the axis. It is negligible at r = 1 (Dirichlet boundary).
# Tensor-product splines cannot represent a non-zero gradient across the axis (they live on a domain
# with a hole and a boundary condition there, see --hole-bc); polar splines can.
# The right-hand side -Laplace(phi) is computed symbolically. Both are given as functions of
# the Cartesian coordinates (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
w = 0.2
x0 = w / np.sqrt(2.0)
phi_sym = sym.exp(-((x_ - x0) ** 2 + y_**2) / w**2) * (1 + sym.cos(2 * sym.pi * z_ / Lz) / 2)
rhs_sym = -(sym.diff(phi_sym, x_, 2) + sym.diff(phi_sym, y_, 2) + sym.diff(phi_sym, z_, 2))
phi_fun = sym.lambdify((x_, y_, z_), phi_sym, "numpy")
rhs_lam = sym.lambdify((x_, y_, z_), rhs_sym, "numpy")


def exact_solution(x, y, z):
    return phi_fun(x, y, z) + 0.0 * x


def rhs_fun(x, y, z):
    return rhs_lam(x, y, z) + 0.0 * x


def distance_to_axis(x, y, z):
    return np.sqrt(x**2 + y**2)


model.em_fields.source.add_perturbation(GenericPerturbation(rhs_fun, given_in_basis="physical"))


if __name__ == "__main__":
    # switches for debugging and testing
    estimate_mem = False
    run = True
    pproc = True
    save_figs = True

    if estimate_mem:
        sim.estimate_mem(print_report=True)
        exit()

    if run:
        out = sim.run(profiling_activated=True, one_time_step=True)
    else:
        # use this for working with existing sim data from a previous run
        out = sim.output

    if pproc:
        out.pproc(create_vtk=True, parallel=True)

    from matplotlib import pyplot as plt

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error on the cross-section at eta3 = 0.5 and the
        half plane at eta2 = 0.5, as functions of (x, y) and (x, z)."""
        fig = plt.figure(figsize=(12, 12))
        idx3 = (slice(None), slice(None), num.shape[2] // 2)
        idx2 = (slice(None), num.shape[1] // 2, slice(None))
        for col, (idx, h, v, labels) in enumerate(
            [(idx3, X, Y, ("x", "y")), (idx2, X, Z, ("x", "z"))],
        ):
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 2, 2 * row + col + 1)
                plt.pcolormesh(h[idx], v[idx], data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                plt.title(f"{name} {what}")
        fig.tight_layout()
        return fig

    if sim.rank == 0:
        # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
        # cell centres of the simulation grid (serial Derham, safe on rank 0 only).
        rhs = out.evaluate("em_fields/source", t=0).values

        phi_data = out.evaluate("em_fields/phi", t=-1)
        phi = phi_data.values
        X, Y, Z = (phi_data.coords[c].values for c in ("X", "Y", "Z"))

        rhs_exact = rhs_fun(X, Y, Z)
        phi_exact = exact_solution(X, Y, Z)

        fig_rhs = plot_slices(rhs, rhs_exact, "RHS")
        fig_phi = plot_slices(phi, phi_exact, "Phi")

        rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
        rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))
        # error close to the axis, where the gradient of the solution is largest
        near_axis = distance_to_axis(X, Y, Z) < 0.1
        rel_err_phi_axis = np.max(np.abs(phi - phi_exact)[near_axis]) / np.max(np.abs(phi_exact))

        print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
        print(f"Max relative error in Phi: {rel_err_phi:.2e}")
        print(f"Max relative error in Phi close to the axis (r < 0.1): {rel_err_phi_axis:.2e}")

        import json
        import os

        # `out.path_out` is the out's (absolute) output folder; `sim_folder` alone is a bare
        # name resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-out<id>`.
        results_dir = os.path.join(out.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
        np.save(os.path.join(results_dir, "rel_err_phi_axis.npy"), rel_err_phi_axis)
        np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
        np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)
        with open(os.path.join(results_dir, "variant.json"), "w") as f:
            json.dump(
                {
                    "splines": args.splines,
                    "precond": args.precond,
                    "poloidal_solver": args.poloidal_solver,
                    "hole_bc": args.hole_bc,
                },
                f,
            )

        if save_figs:
            fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
            fig_phi.savefig(os.path.join(results_dir, "phi_slices.png"))
        else:
            plt.show()
