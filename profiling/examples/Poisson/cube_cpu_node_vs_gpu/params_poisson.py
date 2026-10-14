# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Poisson on 3D cube, no preconditioner: 28 CPU ranks vs 1 GPU"
description = """
Poisson equation on a 3D cube (Cuboid), the problem of cube_strong_scaling/params_poisson.py, run once on
28 CPU MPI ranks with one thread each (NumPy) and once on 1 MPI rank with one thread and 1 GPU (CuPy).
The manufactured solution is a Gaussian 0-form on the logical domain, centered at eta = (0.5, 0.5, 0.5),
exciting the full spectrum.
Homogeneous Dirichlet boundary conditions are set in direction eta1.
The linear system is solved with unpreconditioned CG.

The backend comes from the environment variable CUNUMPY_BACKEND (the profiling driver exports it in the job
script). All timing comes from scope-profiler; on the CuPy backend it records CUDA-event device time per region.
"""

# One thread per MPI rank on both backends (CPU: 28 ranks x 1 thread, GPU: 1 rank x 1 thread): OMP_NUM_THREADS=1, and
# the same for OpenBLAS/MKL, which would otherwise start one thread per core in every rank. Set unconditionally, before
# NumPy is imported (OpenBLAS reads it when it is loaded).
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

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
model.em_fields.phi.save_data = True
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
parser.add_argument("--cells", type=int, default=64, help="Number of cells in each direction.")
parser.add_argument(
    "--steps",
    type=int,
    default=1,
    help="Number of time steps, i.e. Poisson solves with the same matrices (assembled once in the setup).",
)
args, _ = parser.parse_known_args()

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    restart=False,
    save_step=args.steps,  # save the first and the last snapshot only
)

# Time stepping
# Time stepping: every step solves the Poisson problem again (from x0 = 0, recycle=False) with the matrices
# assembled once in the setup, as in a time-dependent simulation; the solution is the same at every step.
dt = 0.01
time_opts = Time(dt=dt, Tend=round(args.steps * dt, 14))

# Geometry
Lx = 2.0
Ly = 3.0
Lz = 4.0
domain = domains.Cuboid(r1=Lx, l2=-Ly / 2, r2=Ly / 2, r3=Lz)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(args.cells,) * 3, mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions(degree=(1, 2, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profiling options: CUDA-event device times per region on the CuPy backend (not on cunumpy's fake CuPy, whose
# CUDA events have no elapsed time). The line profiler of the strong-scaling case is off: it would distort the
# CPU/GPU comparison.
import cunumpy as xp
from cunumpy.kernel_testing import fake_cupy_active

gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()

# scope-profiler run label, the name of the run in the reports and comparison plots, e.g.
# "CPU (numpy): 28 MPI ranks x 1 thread" and "GPU (cupy): 1 MPI rank x 1 thread, 1 GPU"
from maybempi import MPI, SerialMPI

num_ranks = 1 if isinstance(MPI, SerialMPI) else MPI.COMM_WORLD.Get_size()
num_threads = int(os.environ["OMP_NUM_THREADS"])
profiling_label = (
    f"{'GPU' if xp.get_backend() == 'cupy' else 'CPU'}: "
    f"{num_ranks} MPI rank{'s' if num_ranks != 1 else ''}"
    + (f", {num_ranks} GPU{'s' if num_ranks != 1 else ''}" if xp.get_backend() == "cupy" else "")
)

profiling_opts = ProfilingOptions(
    label=profiling_label,
    use_gpu_timing=gpu_timing,
    gpu_timing_backend="cupy",
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
    solver="pcg",
    precond=None,
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
    # switches for debugging and testing
    estimate_mem = False
    run = True
    pproc = True
    save_figs = True

    if estimate_mem:
        sim.estimate_mem(print_report=True)
        exit()

    if run:
        out = sim.run(profiling_activated=True)
    else:
        # use this for working with existing sim data from a previous run
        out = sim.output

    if pproc:
        out.pproc(create_vtk=True, parallel=True)

    from matplotlib import pyplot as plt

    def slice_axes(k, X, Y, Z, E1, E2, E3):
        """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5 in physical coordinates."""
        if k == 0:
            return Y, Z, "y", "z"
        elif k == 1:
            return X, Z, "x", "z"
        else:
            return X, Y, "x", "y"

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error on the slices eta_k = 0.5 (k = 1, 2, 3)."""
        fig = plt.figure(figsize=(16, 12))
        for k in range(3):
            idx = tuple(num.shape[k] // 2 if i == k else slice(None) for i in range(3))
            h, v, hlabel, vlabel = slice_axes(k, X[idx], Y[idx], Z[idx], E1[idx], E2[idx], E3[idx])
            eta_k = (E1, E2, E3)[k][idx].flat[0]
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 3, 3 * row + k + 1)
                plt.pcolormesh(h, v, data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(hlabel)
                plt.ylabel(vlabel)
                plt.title(f"{name} {what}, slice at eta{k + 1} = {eta_k:.2f}")
        fig.tight_layout()
        return fig

    if sim.rank == 0:
        # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
        # cell centres of the simulation grid (serial Derham, safe on rank 0 only).
        rhs = out.evaluate("em_fields/source", t=0).values

        phi_data = out.evaluate("em_fields/phi", t=-1)
        phi = phi_data.values

        # logical coordinates of the evaluation points and their images x = F(eta)
        E1, E2, E3 = np.meshgrid(
            *(phi_data.coords[c].values for c in ("eta1", "eta2", "eta3")),
            indexing="ij",
        )
        X, Y, Z = (phi_data.coords[c].values for c in ("X", "Y", "Z"))

        # pushed-forward exact solutions: the 0-forms at x = F(eta) are their values at eta
        rhs_exact = rhs_fun(E1, E2, E3)
        phi_exact = exact_solution(E1, E2, E3)

        fig_rhs = plot_slices(rhs, rhs_exact, "RHS")
        fig_phi = plot_slices(phi, phi_exact, "Phi")

        rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
        rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))

        print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
        print(f"Max relative error in Phi: {rel_err_phi:.2e}")

        assert rel_err_rhs < 1.7e-2, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_phi < 1.2e-2, (
            f"The computed solution does not match the exact solution, max rel error = {rel_err_phi}."
        )

        import os

        # `out.path_out` is the out's (absolute) output folder; `sim_folder` alone is a bare
        # name resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-out<id>`.
        results_dir = os.path.join(out.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
        np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
        np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
        np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

        if save_figs:
            fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
            fig_phi.savefig(os.path.join(results_dir, "phi_slices.png"))
        else:
            plt.show()
