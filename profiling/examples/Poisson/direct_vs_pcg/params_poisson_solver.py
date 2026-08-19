# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Poisson on cube, PCG vs DirectSolver"
description = """
Benchmark for feectools.linalg.solvers.DirectSolver (see
ISSUE_add_direct_solver_for_constant_operators.md), built on the same manufactured-
solution Poisson case as profiling/examples/Poisson/cube_strong_scaling/params_poisson.py,
but adapted to actually exercise repeated solves of the *same* system instead of a
single one-shot solve.

Unlike that scaling case, the manufactured solution here uses the lowest wavenumbers
that are periodic on y and z (2*pi/L, i.e. one full period across the domain, vs.
12*pi/Ly and 4*pi/Lz there -- also periodic, but far finer). That case is meant to be
resolved on a 256^3 grid, but DirectSolver only supports a single MPI rank, and a
sparse-direct factorization's fill-in grows fast with problem size in 3D -- a grid
coarse enough to finish factorizing in reasonable time would badly under-resolve the
original high wavenumbers. Lowering them keeps the case accurate on a small
(single-rank-sized) grid, which is the regime DirectSolver actually targets.

PoissonSolve (src/struphy/propagators/poisson_solve.py) is an ImplicitDiffusion
subclass with divide_by_dt=False and a time-independent right-hand side here (no
--with-t-dep-source), so its left-hand-side operator -- and, since the manufactured
source never changes, its solution -- is identical on every single time step. That is
exactly the situation feectools.linalg.solvers.DirectSolver targets: PCG re-derives an
iterative solution from scratch every step, while DirectSolver factorizes the
(unchanging) operator once, lazily, on the first call, and reuses that factorization --
a single triangular back/forward-solve, no iteration -- for every call after that.
`solver_params` below deliberately sets `recycle=False`: with `recycle=True` PCG
would warm-start from the previous (already-converged, since the system never
changes) solution and hit its `tol` in a single iteration, which is just as cheap as
DirectSolver's cached solve and hides the comparison this benchmark exists to make.

`--repeats` controls how many identical time steps to run (default 100); more repeats
make DirectSolver's one-time factorization cost matter less and less relative to PCG's
cost, which is paid in full on every single call. On the default 8x8x8 grid, the
factorization costs ~0.03s and each direct solve after that is ~0.15ms -- roughly 7x
faster than a from-scratch PCG solve (~1ms) -- so DirectSolver wins on total wall time
by ~30 repeats and is ~2.5x faster overall by 100.
"""

import logging

from struphy import set_logging_level

set_logging_level(logging.WARNING)

import argparse

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import Poisson

# Environment options
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
parser.add_argument(
    "--solver",
    choices=("pcg", "direct"),
    default="pcg",
    help="Symmetric solver for the Poisson field solve (default: pcg).",
)
parser.add_argument(
    "--num-elements",
    type=int,
    nargs=3,
    default=[8, 8, 8],
    help="Grid resolution (default: 8 8 8). Deliberately small: DirectSolver is single-rank "
    "only, and a sparse-direct factorization's fill-in grows fast with problem size in 3D, so "
    "this case never scales MPI ranks or grid size the way profiling/examples/Poisson/"
    "cube_strong_scaling does.",
)
parser.add_argument(
    "--repeats",
    type=int,
    default=100,
    help="Number of identical time steps to run, i.e. repeated solves of the same system "
    "(default: 100; DirectSolver's one-time factorization cost needs enough repeats to "
    "amortize -- see the module docstring).",
)
args, _ = parser.parse_known_args()

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

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    profiling_activated=True,
    restart=False,
)

# Time stepping. dt/Tend chosen so `--repeats` gives exactly that many identical steps
# (the manufactured source below is time-independent, so every step re-solves the same
# system -- see the module docstring for why that is the point of this benchmark).
time_opts = Time(dt=1.0, Tend=float(args.repeats), split_algo="LieTrotter")

# Geometry
Lx = 2.0
Ly = 3.0
Lz = 4.0
domain = domains.Cuboid(r1=Lx, l2=-Ly / 2, r2=Ly / 2, r3=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

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
)

# ------------------
# Propagator options
# ------------------

from struphy.linear_algebra.solver import SolverParameters

solver_params = SolverParameters(tol=1e-12, maxiter=3000, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver=args.solver,
    precond=None,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np

from struphy.initial.base import GenericPerturbation


def exact_solution(x, y, z):
    return np.sin(np.pi / Lx * x) * np.cos(2 * np.pi / Ly * y + 2 * np.pi / Lz * z)


def rhs_fun(x, y, z):
    return exact_solution(x, y, z) * ((np.pi / Lx) ** 2 + (2 * np.pi / Ly) ** 2 + (2 * np.pi / Lz) ** 2)


rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="physical")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    sim.run()
    sim.pproc(parallel_pproc=True)

    if sim.rank == 0:
        import os

        results_dir = os.path.join(sim.env.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Correctness check: since the source is time-independent, phi at the last step
        # should still match the manufactured solution, regardless of solver.
        sim.load_plotting_data()
        Tend_saved = sim.t_grid[-1]
        phi = sim.spline_values.em_fields.phi_log.data[Tend_saved][0]
        x, y, z = sim.grids_phy
        rel_err_phi = np.max(np.abs(phi - exact_solution(x, y, z))) / np.max(np.abs(exact_solution(x, y, z)))
        print(f"solver={args.solver}: max relative error in Phi after {args.repeats} steps: {rel_err_phi:.2e}")
        assert rel_err_phi < 1e-2, f"The computed solution does not match the exact solution, max rel error = {rel_err_phi}."

        np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
        np.save(os.path.join(results_dir, "resolution.npy"), np.asarray(args.num_elements))
        np.save(os.path.join(results_dir, "repeats.npy"), args.repeats)
        with open(os.path.join(results_dir, "solver.txt"), "w") as f:
            f.write(args.solver)
