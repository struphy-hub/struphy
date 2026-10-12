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
    default="dirichlet",
    choices=["free", "dirichlet"],
    help="Boundary condition at the hole of the tensor-product domain (default dirichlet; the natural boundary condition contradicts the solution, whose curl is largest on the axis, and the error is amplified by 1/sigma).",
)
args, _ = parser.parse_known_args()

precond = None if args.precond == "None" else args.precond

name = f"Curl-curl strong scaling on a cylinder, {args.splines} splines, preconditioner {precond}"
description = f"""
Strong scaling test for the curl-curl problem curl curl E + sigma E = J on a cylinder of radius 1 and length 4,
comparing C1 polar splines (IGAPolarCylinder, no hole) with tensor-product splines on a cylinder with a hole
(HollowCylinder, a1=1e-2). The manufactured solution E = g c (constant Cartesian vector c) has a Gaussian g whose
gradient is largest on the axis r=0.
Homogeneous Dirichlet boundary conditions (vanishing tangential E) are set at r=1 (boundary condition {args.hole_bc} at the hole).
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
# g(x, y, z) = Gaussian of width w in the cross-section, centered at distance w / sqrt(2) from the axis, where its
# gradient is largest, so the gradient is largest on the axis r = 0, times a periodic modulation along the axis.
# It is negligible at r = 1, hence satisfies the boundary conditions up to that accuracy.
# Tensor-product splines cannot represent non-zero fields and gradients across the axis (they live on a domain
# with a hole and a boundary condition there, see --hole-bc); polar splines can.
# The right-hand side J = curl curl E + sigma E is computed symbolically. Both are given in Cartesian
# components (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
w = 0.2
x0 = w / np.sqrt(2.0)
g = sym.exp(-((x_ - x0) ** 2 + y_**2) / w**2) * (1 + sym.cos(2 * sym.pi * z_ / Lz) / 2)
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
    return np.sqrt(x**2 + y**2)


for comp in range(3):
    model.em_fields.source.add_perturbation(GenericPerturbation(rhs(comp), given_in_basis="physical", comp=comp))


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
        out.pproc(create_vtk=True, parallel=True, physical=True)

    from matplotlib import pyplot as plt

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error of the magnitude on the cross-section at eta3 = 0.5 and the
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
                plt.pcolormesh(h[idx], v[idx], np.nan_to_num(data), shading="gouraud")
                plt.colorbar()
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                plt.title(f"|{name}| {what}")
        fig.tight_layout()
        return fig

    if sim.rank == 0:
        # Cartesian components of the fields at the cell centres of the simulation grid
        e_xyz = out.evaluate("em_fields/e_field_xyz", t=-1)
        j_xyz = out.evaluate("em_fields/source_xyz", t=0)
        X, Y, Z = (e_xyz.coords[c_].values for c_ in ("X", "Y", "Z"))

        e = np.stack([e_xyz.isel(component=k).values for k in range(3)])
        j = np.stack([j_xyz.isel(component=k).values for k in range(3)])
        e_exact = np.stack([exact_solution(k)(X, Y, Z) for k in range(3)])
        j_exact = np.stack([rhs(k)(X, Y, Z) for k in range(3)])

        fig_rhs = plot_slices(np.linalg.norm(j, axis=0), np.linalg.norm(j_exact, axis=0), "J")
        fig_e = plot_slices(np.linalg.norm(e, axis=0), np.linalg.norm(e_exact, axis=0), "E")

        # the Cartesian components are not defined on the pole (singular mapping), which is left out
        valid = np.isfinite(e).all(axis=0) & np.isfinite(j).all(axis=0)
        rel_err_rhs = np.max(np.abs(j - j_exact)[:, valid]) / np.max(np.abs(j_exact))
        rel_err_e = np.max(np.abs(e - e_exact)[:, valid]) / np.max(np.abs(e_exact))
        # error close to the axis, where the gradient of the solution is largest
        near_axis = valid & (distance_to_axis(X, Y, Z) < 0.1)
        rel_err_e_axis = np.max(np.abs(e - e_exact)[:, near_axis]) / np.max(np.abs(e_exact))

        print(f"Max relative error in J: {rel_err_rhs:.2e}")
        print(f"Max relative error in E: {rel_err_e:.2e}")
        print(f"Max relative error in E close to the axis (r < 0.1): {rel_err_e_axis:.2e}")

        import json
        import os

        # `out.path_out` is the out's (absolute) output folder; `sim_folder` alone is a bare
        # name resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-out<id>`.
        results_dir = os.path.join(out.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_e.npy"), rel_err_e)
        np.save(os.path.join(results_dir, "rel_err_e_axis.npy"), rel_err_e_axis)
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
            fig_e.savefig(os.path.join(results_dir, "e_slices.png"))
        else:
            plt.show()
