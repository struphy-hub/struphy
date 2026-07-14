"""
Benchmark: compare gmres, uzawa, and schur solvers across problem sizes.

NOTE: "schur" requires SchurComplementSolver to be registered in
      LiteralOptions.OptsSaddlePointSolver and the allocate() dispatch.

Saves:
  runs/bench_plots/iterations.png
  runs/bench_plots/walltime.png
"""

import time
import os
import cunumpy as xp
import matplotlib.pyplot as plt
from cunumpy import pi, zeros_like
from mpi4py import MPI

from struphy.io.options import EnvironmentOptions, Time, DerhamOptions
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.linear_algebra.solver import SolverParameters
from struphy import Simulation
from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy

# ---------------------------------------------------------------
# fixed physical parameters
# ---------------------------------------------------------------
B0      = 1.0
nu      = 10.0
nu_e    = 1.0
epsilon = 1.0
sigma   = 0.0
dt      = 1.0
Tend    = 1.0          # single time step per run
tol     = 1e-5
degree  = (2, 2, 1)
BC      = "dirichlet_hom"

# ---------------------------------------------------------------
# sweep config
# ---------------------------------------------------------------
NEL_SIZES = [4, 8, 12, 16, 18]
SOLVERS   = ["gmres", "schur"]

# ---------------------------------------------------------------
# source terms (dirichlet_hom MMS)
# ---------------------------------------------------------------
def source_function_u(x, y, z):
    fx = (
        -2*pi*xp.sin(2*pi*x)
        - B0/epsilon * xp.cos(2*pi*x)*xp.sin(2*pi*y)
        - nu*8*pi**2 * xp.sin(2*pi*x)*xp.cos(2*pi*y)
    )
    fy = (
        2*pi*xp.cos(2*pi*y)
        - B0/epsilon * xp.sin(2*pi*x)*xp.cos(2*pi*y)
        + nu*8*pi**2 * xp.cos(2*pi*x)*xp.sin(2*pi*y)
    )
    return fx, fy, zeros_like(x)

def source_function_ue(x, y, z):
    fx = (
        2*pi*xp.sin(2*pi*x)
        + B0/epsilon * xp.cos(4*pi*x)*xp.sin(4*pi*y)
        - nu_e*32*pi**2 * xp.sin(4*pi*x)*xp.cos(4*pi*y)
        + sigma * xp.sin(4*pi*x)*xp.cos(4*pi*y)
    )
    fy = (
        -2*pi*xp.cos(2*pi*y)
        + B0/epsilon * xp.sin(4*pi*x)*xp.cos(4*pi*y)
        + nu_e*32*pi**2 * xp.cos(4*pi*x)*xp.sin(4*pi*y)
        - sigma * xp.cos(4*pi*x)*xp.sin(4*pi*y)
    )
    return fx, fy, zeros_like(x)

# ---------------------------------------------------------------
# single benchmark run
# ---------------------------------------------------------------
def run_benchmark(nel, solver_name):
    """Build, run one time step, return (elapsed_s, niter, success)."""
    Nel  = (nel, nel, 1)
    name = f"runs/bench_{solver_name}_nel{nel}"

    env         = EnvironmentOptions(sim_folder=name)
    time_opts   = Time(dt=dt, Tend=Tend)
    domain      = domains.Cuboid()
    equil       = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
    grid        = grids.TensorProductGrid(num_elements=Nel)
    derham_opts = DerhamOptions(
        degree=degree,
        bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None),
    )

    model = TwoFluidQuasiNeutralToy()
    model.propagators.qn_full.options = model.propagators.qn_full.Options(
        nu=nu,
        nu_e=nu_e,
        eps_norm=epsilon,
        stab_sigma=sigma,
        source_u=source_function_u,
        source_ue=source_function_ue,
        solver=solver_name,
        solver_params=SolverParameters(tol=tol, info=True),
    )

    sim = Simulation(
        model=model,
        params_path=__file__,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )

    # time the single step (Tend == dt, so sim.run() does exactly one step)
    t0      = time.perf_counter()
    sim.run()
    elapsed = time.perf_counter() - t0

    info    = sim.model.propagators.qn_full._Minv.get_info()
    niter   = info.get("niter",   -1)
    success = info.get("success", False)

    return elapsed, niter, success

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
if __name__ == "__main__":

    rank = MPI.COMM_WORLD.Get_rank()

    results = {
        s: {"nel": [], "time": [], "niter": [], "success": []}
        for s in SOLVERS
    }

    for solver_name in SOLVERS:
        if rank == 0:
            print(f"\n{'='*52}")
            print(f"  Solver: {solver_name}")
            print(f"{'='*52}")

        for nel in NEL_SIZES:
            if rank == 0:
                print(f"  Nel=({nel},{nel},1) ...", end=" ", flush=True)

            try:
                elapsed, niter, success = run_benchmark(nel, solver_name)

                if rank == 0:
                    print(f"time={elapsed:.2f}s  niter={niter}  ok={success}")
                    results[solver_name]["nel"].append(nel)
                    results[solver_name]["time"].append(elapsed)
                    results[solver_name]["niter"].append(niter)
                    results[solver_name]["success"].append(success)

            except Exception as exc:
                if rank == 0:
                    print(f"FAILED: {exc}")

    # ---------------------------------------------------------------
    # plots (rank 0 only)
    # ---------------------------------------------------------------
    if rank == 0:
        os.makedirs("runs/bench_plots", exist_ok=True)

        markers = {"gmres": "o-",  "uzawa": "s--", "schur": "^:"}
        colors  = {"gmres": "C0",  "uzawa": "C1",  "schur": "C2"}

        # approximate DOF: (nel + degree)^2 for 2D splines
        def approx_dof(nel):
            return (nel + degree[0]) * (nel + degree[1])

        # ---- iteration count ----
        fig, ax = plt.subplots(figsize=(7, 5))
        for s in SOLVERS:
            r = results[s]
            if not r["nel"]:
                continue
            dofs = [approx_dof(n) for n in r["nel"]]
            ax.plot(dofs, r["niter"], markers[s], color=colors[s], label=s, markersize=7)

        ax.set_xlabel("approximate DOF")
        ax.set_ylabel("outer iteration count")
        ax.set_title("Iterations vs problem size")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("runs/bench_plots/iterations.png", dpi=150)
        plt.close()
        print("\nSaved: runs/bench_plots/iterations.png")

        # ---- wall-clock time ----
        fig, ax = plt.subplots(figsize=(7, 5))
        for s in SOLVERS:
            r = results[s]
            if not r["nel"]:
                continue
            dofs = [approx_dof(n) for n in r["nel"]]
            ax.plot(dofs, r["time"], markers[s], color=colors[s], label=s, markersize=7)

        ax.set_xlabel("approximate DOF")
        ax.set_ylabel("wall-clock time (s)")
        ax.set_title("Wall-clock time vs problem size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("runs/bench_plots/walltime.png", dpi=150)
        plt.close()
        print("Saved: runs/bench_plots/walltime.png")

        # ---- summary table ----
        print(f"\n{'solver':<8} {'nel':>5} {'dof':>8} {'niter':>7} {'time(s)':>9} {'ok':>5}")
        print("-" * 46)
        for s in SOLVERS:
            r = results[s]
            for nel, niter, t, ok in zip(r["nel"], r["niter"], r["time"], r["success"]):
                print(f"{s:<8} {nel:>5} {approx_dof(nel):>8} {niter:>7} {t:>9.2f} {str(ok):>5}")