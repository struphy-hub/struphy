"""Poisson equation -Laplace phi = rho (electrostatic potential of a charge density) on a cube with a
manufactured Gaussian solution (homogeneous Dirichlet boundary conditions in eta1), solved with CG and a choice of
preconditioners. The pproc reports the max. relative error of the right-hand side and the solution."""

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
# `--precond` selects the preconditioner (the variant), see the submit script; pproc_gaussian_cube.py takes the
# same flags.
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
    "--precond",
    default="None",
    choices=["None", "MultiGrid", "StiffnessPreconditioner"],
    help="Name of the preconditioner (None for no preconditioner).",
)
parser.add_argument("--num-elements", type=int, nargs=3, default=[128, 128, 128], help="Number of elements per direction.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (num_elements = [16, 16, 16]), for the tests (test_examples.py).",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1,
    help="Number of time steps, i.e. Poisson solves with the same matrices (assembled once in the setup).",
)
parser.add_argument(
    "--solver",
    default="pcg",
    choices=["pcg", "pcg_sr"],
    help="CG variant: 'pcg_sr' does one global reduction per iteration instead of two (for many ranks/GPUs).",
)
parser.add_argument(
    "--gpu-timing",
    action="store_true",
    help="Setup of the CPU vs GPU cases: one thread per rank, CUDA-event device times per region on the CuPy "
    "backend instead of the line profiler, and a run label naming the backend and the rank count.",
)
parser.add_argument(
    "--no-output",
    action="store_true",
    help="Write only the profiling data: no field or restart data (the pproc script then does nothing).",
)
parser.add_argument(
    "--pinning",
    default=None,
    help="Placement of the ranks set by the submit script (compact, spread, balanced); only added to the profiling label.",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.num_elements = [16, 16, 16]

if args.gpu_timing:
    # One thread per MPI rank on both backends: OMP_NUM_THREADS=1, and the same for OpenBLAS/MKL, which would
    # otherwise start one thread per core in every rank. Set before NumPy is imported (OpenBLAS reads it when loaded).
    import os

    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

precond = None if args.precond == "None" else args.precond

# preconditioner -> (name suffix, linear solver) for the name and the description
PRECONDS = {
    None: ("no preconditioner", "unpreconditioned CG"),
    "MultiGrid": ("multigrid preconditioner", "CG preconditioned by geometric multigrid"),
    "StiffnessPreconditioner": (
        "stiffness preconditioner",
        "CG preconditioned by the Kronecker approximation of the inverse Laplacian (StiffnessPreconditioner)",
    ),
}

name = f"Poisson on 3D cube, {PRECONDS[precond][0]}"
description = f"""
Poisson equation -Laplace phi = rho for the electrostatic potential phi of a given charge density rho on a 3D
cube (Cuboid), with a manufactured localized solution, to test the accuracy and the preconditioners of the solver.

Physics. Poisson's equation -Laplace phi = rho determines the electrostatic potential phi, and
thereby the electric field E = -grad phi, of a given charge density rho (Gauss's law div E = rho for a curl-free
E, in Struphy units). It is the field equation of electrostatic models such as Vlasov-Poisson and the prototype
of the elliptic problems in Struphy. The Poisson model solves -div(D0 grad phi) + n0 phi = rho; here D0 is the
identity and the stabilization term is switched off (stab_eps = 0), i.e. the pure Laplacian with phi in the
spline space V0 of H1 of the discrete de Rham complex (FEEC).
The Gaussian potential corresponds to a localized charge distribution, a positive core inside a
negative shell with zero net charge (the integral of Laplace phi vanishes), hence there is no long-range 1/r tail.
Being localized, it excites the whole spectrum of the Laplacian, whose condition number grows like h^-2 with the
mesh size h, so unpreconditioned CG needs many iterations.

Setup. Cube [0, 2] x [-1.5, 1.5] x [0, 4] (Cuboid), splines of degree (1, 2, 3).
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1, periodic ones in eta2 and eta3.
The linear system is solved with {PRECONDS[precond][1]}.

Expected result. The pproc script compares the computed phi and the projected right-hand side rho with
the exact fields and prints the max. relative errors; at the default resolution (not with --test-run) it asserts
that they are below 1.7e-2 (rho) and 1.2e-2 (phi). The number of CG iterations shows the efficiency of the
preconditioner.

References:
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
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    ProfilingOptions,
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

# Units
base_units = BaseUnits()

# Model instance
model = Poisson(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = not args.no_output
model.em_fields.source.save_data = not args.no_output

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    out_folders=args.out_folder,
    sim_folder=args.sim_folder,
    restart=False,
    save_step=args.steps,  # save the first and the last snapshot only
    save_restart=not args.no_output,
)

# Time stepping: with --steps > 1 every step solves the Poisson problem again (from x0 = 0, recycle=False) with the
# matrices assembled once in the setup, as in a time-dependent simulation; the solution is the same at every step.
dt = 0.01
time_opts = Time() if args.steps == 1 else Time(dt=dt, Tend=round(args.steps * dt, 14))

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

# Profiling options
import os

import cunumpy as xp
from cunumpy.kernel_testing import fake_cupy_active
from maybempi import MPI, SerialMPI

num_ranks = 1 if isinstance(MPI, SerialMPI) else MPI.COMM_WORLD.Get_size()
# on a real GPU: the CuPy backend, not cunumpy's fake CuPy (which runs on the CPU)
on_gpu = xp.get_backend() == "cupy" and not fake_cupy_active()

# OpenMP threads per rank and their placement, as set by the job script (see ProfilingCase.launch)
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1").split(",")[0])
omp_proc_bind = os.environ.get("OMP_PROC_BIND")

# scope-profiler run label, the name of the run in the reports and comparison plots, e.g.
# "CPU: 28 MPI ranks", "CPU: 8 MPI ranks, spread", "CPU: 1 MPI rank, 16 threads, close" and
# "GPU: 4 MPI ranks, 4 GPUs"
profiling_label = (
    f"{'GPU' if on_gpu else 'CPU'}: {num_ranks} MPI rank{'s' if num_ranks != 1 else ''}"
    + (f", {num_ranks} GPU{'s' if num_ranks != 1 else ''}" if on_gpu else "")
    + (f", {num_threads} threads" if num_threads > 1 and not on_gpu else "")
    + (f", {args.pinning}" if args.pinning is not None else "")
    + (f", {omp_proc_bind}" if omp_proc_bind and not on_gpu else "")
)
if args.gpu_timing:
    # CUDA-event device times per region on the CuPy backend (not on cunumpy's fake CuPy, whose CUDA events have no
    # elapsed time). No line profiler: it would distort the CPU/GPU comparison.
    profiling_opts = ProfilingOptions(
        label=profiling_label,
        use_gpu_timing=on_gpu,
        gpu_timing_backend="cupy",
    )
else:
    profiling_opts = ProfilingOptions(
        label=profiling_label,
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

solver_params = SolverParameters(tol=1e-8, maxiter=3000, info=True, recycle=False)
model.propagators.poisson.options = model.propagators.poisson.Options(
    stab_eps=0.0,
    solver=args.solver,
    precond=precond,
    solver_params=solver_params,
)

# ------------------
# Initial conditions
# ------------------
import numpy as np

from struphy.initial.base import GenericPerturbation

# The exact solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5).
# Its value at the Dirichlet boundary eta1 = 0, 1 is exp(-0.25 / w**2) ~ 1e-11, and it is periodic
# in eta2 and eta3 up to the same accuracy. Being localized, it excites the whole spectrum of the
# Laplacian, so CG has to work for convergence. Both the solution and the right-hand side
# -Laplace(phi) are 0-forms given as functions of the logical coordinates (given_in_basis="0");
# the pushed-forward solution at x = F(eta) is phi(eta).
w = 0.1
eta0 = (0.5, 0.5, 0.5)


def gaussian_derivatives(e1, e2, e3):
    """The Gaussian and its first and second derivatives with respect to eta1, eta2, eta3."""
    eta = (e1, e2, e3)
    g = np.exp(-sum((e - c) ** 2 for e, c in zip(eta, eta0)) / w**2)
    d = [-2 * (e - c) / w**2 * g for e, c in zip(eta, eta0)]
    dd = [(4 * (e - c) ** 2 / w**4 - 2 / w**2) * g for e, c in zip(eta, eta0)]
    return g, d, dd


def exact_solution(e1, e2, e3):
    return gaussian_derivatives(e1, e2, e3)[0]


def laplacian(e1, e2, e3):
    """Physical Laplacian of the exact solution, in logical coordinates (Cuboid: x_i = l_i + L_i * eta_i)."""
    g, d, dd = gaussian_derivatives(e1, e2, e3)
    return dd[0] / Lx**2 + dd[1] / Ly**2 + dd[2] / Lz**2


def rhs_fun(e1, e2, e3):
    return -laplacian(e1, e2, e3)


rhs_perturbation = GenericPerturbation(rhs_fun, given_in_basis="0")

model.em_fields.source.add_perturbation(rhs_perturbation)


if __name__ == "__main__":
    # post-processing and plots: pproc_gaussian_cube.py
    sim.run(profiling_activated=True, one_time_step=args.steps == 1)
