# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Curl-curl strong scaling on hollow torus, multigrid preconditioner"
description = """
Strong scaling test for the curl-curl problem curl curl E + sigma E = J on a full hollow torus (HollowTorus, a1=1e-2, a2=1, R0=4).
The manufactured solution is a Gaussian vector field E = g(x) c in Cartesian coordinates, with a constant
vector c and a Gaussian g centered at x0 inside the domain, exciting the full spectrum.
Homogeneous Dirichlet boundary conditions (vanishing tangential E) are set in direction eta1.
The linear system is solved with CG preconditioned by geometric multigrid with the hybrid smoother of Hiptmair.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import argparse

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
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
args, _ = parser.parse_known_args()

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    restart=False,
)

# Time stepping
time_opts = Time()

# Geometry
a1 = 1e-2
a2 = 1.0
R0 = 4.0
tor_period = 1
domain = domains.HollowTorus(a1=a1, a2=a2, R0=R0, sfl=False, pol_period=1, tor_period=tor_period)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(64, 64, 64), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(2, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profilinig options
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

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.curl_curl.options = model.propagators.curl_curl.Options(
    sigma=sigma,
    solver="pcg",
    precond="MultiGrid",
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np
import sympy as sym

from struphy.initial.base import GenericPerturbation

# The exact solution is the vector field E = g(x) c in Cartesian coordinates, with a constant vector c and a
# localized Gaussian g (below). It is negligible at the boundary, hence satisfies the boundary conditions
# up to that accuracy. Being localized, it excites the whole spectrum of the curl-curl operator, including
# gradients, so CG has to work for convergence.
# The right-hand side J = curl curl E + sigma E is computed symbolically. Both are given in Cartesian
# components (given_in_basis="physical").
x_, y_, z_ = sym.symbols("x y z", real=True)
# Gaussian following the torus, centered at mid radius on the outboard side (theta = phi = 0, R = R0 + 0.5):
# width 0.15 in the poloidal plane and 1.5 along the toroidal arc length (Rc * phi, with phi**2 replaced by
# the smooth periodic 2 * (1 - cos(phi))), where the elements are longest (2*pi*R / 64 ~ 0.44)
Rc = R0 + 0.5
R_ = sym.sqrt(x_**2 + y_**2)
g = sym.exp(-((R_ - Rc) ** 2 + z_**2) / 0.15**2 - 2 * Rc**2 * (1 - x_ / R_) / 1.5**2)
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
        """Numerical solution, exact solution and error of the magnitude on the logical slices eta_k = 0.5 (k = 1, 2, 3)."""
        fig = plt.figure(figsize=(16, 12))
        for k in range(3):
            idx = tuple(num.shape[k] // 2 if i == k else slice(None) for i in range(3))
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 3, 3 * row + k + 1)
                plt.pcolormesh(data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(f"eta{[2, 1, 1][k]} index")
                plt.ylabel(f"eta{[3, 3, 2][k]} index")
                plt.title(f"|{name}| {what}, slice at eta{k + 1} = 0.5")
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

        rel_err_rhs = np.max(np.abs(j - j_exact)) / np.max(np.abs(j_exact))
        rel_err_e = np.max(np.abs(e - e_exact)) / np.max(np.abs(e_exact))

        print(f"Max relative error in J: {rel_err_rhs:.2e}")
        print(f"Max relative error in E: {rel_err_e:.2e}")

        assert rel_err_rhs < 8e-3, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_e < 2e-2, (
            f"The computed solution does not match the exact solution, max rel error = {rel_err_e}."
        )

        import os

        # `out.path_out` is the out's (absolute) output folder; `sim_folder` alone is a bare
        # name resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-out<id>`.
        results_dir = os.path.join(out.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_e.npy"), rel_err_e)
        np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
        np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

        if save_figs:
            fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
            fig_e.savefig(os.path.join(results_dir, "e_slices.png"))
        else:
            plt.show()
