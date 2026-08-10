"""ToyDrift periodic slab (32^3): PETSc vs. pcg, the largest gap in this benchmark suite.

See ``profiling/examples/ToyDrift/periodic_slab_hires/params_periodic_slab_hires.py`` for the
full story: a 32^3 grid (32768 dofs) with PETSc's preconditioner explicitly set to ``"gamg"``
(algebraic multigrid, via ``SolverParameters.pc_type`` -- see
``struphy.linear_algebra.solver.SolverParameters``) instead of the default ``"jacobi"``.

Measured via ``struphy.linear_algebra.petsc_examples_benchmark``'s repeated-solve methodology
(isolates just the solve, amortizing gamg's one-time multigrid setup across several calls at
fixed dt -- see that module's docstring): pcg needs ~190 CG iterations per solve (~10.9s) at this
size, against PETSc+gamg's ~2 iterations (~0.29s) -- a ~38x difference. This near-singular system
is ill-conditioned mainly through its weakly constrained DC mode, not primarily through
resolution, so the ratio does not grow much further with grid size -- but the *absolute* cost
does: pcg's ~11 seconds per solve here is the practically relevant "huge difference" once
multiplied over the many timesteps of a real simulation. See
``submit_strong_landau_damping_petsc.py`` and ``submit_vlasov_maxwell_petsc.py`` for the
smaller-scale (3-10x) comparisons.

Each launch runs one 32^3-grid, one-time-step simulation (``params_periodic_slab_hires.py``'s
``__main__`` calls ``sim.run(one_time_step=True)``); note that a *single* one-time-step run still
pays gamg's one-time setup cost up front and will not by itself reproduce the ~38x figure above --
that requires several solves at the same dt to amortize the setup, exactly what
``petsc_examples_benchmark.py`` does. A pcg launch alone takes roughly a minute here (mostly the
single Poisson solve), so a local (non-SLURM) full sweep of this script takes several minutes.
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "ToyDrift" / "periodic_slab_hires"

    profiling_case = ProfilingCase(
        label="toydrift_periodic_slab_hires_petsc",
        name="ToyDrift periodic slab (32^3, hires): pcg vs. PETSc+gamg at a larger problem size",
        description=(
            "Higher-resolution (32^3 grid, 32768 dofs, degree-3 splines) periodic-slab ToyDrift "
            "setup, solving the per-step guiding-center Poisson problem with either feectools' "
            "native preconditioned CG (--solver pcg) or PETScSolver with an algebraic multigrid "
            "preconditioner (KSP=cg, PC=gamg, --solver petsc). At this size pcg needs roughly 190 "
            "iterations per solve (about 11 seconds) while PETSc+gamg needs about 2 (a third of a "
            "second); the ratio (~38x) is similar to the smaller periodic_slab case, but the "
            "absolute per-solve cost -- and hence the real wall-clock impact over a full run -- "
            "is far larger here."
        ),
        physics_problem="Electrostatic E x B drift of a single ion species in a periodic slab.",
        struphy_model_used="ToyDrift",
        params_source=params_dir / "params_periodic_slab_hires.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )
    profiling_case.use_slurm = False

    # Launch one run per (rank count, solver) combination, same rank counts for both solvers so
    # they are directly comparable. Each launch gets its own `--id` (hence its own `sim_<id>`
    # output folder), so the two solvers never collide even at the same rank count.
    for num_tasks in (1,):  # 2, 4):
        for solver in ("pcg", "petsc"):
            profiling_case.launch(num_tasks, param_flags=["--solver", solver])

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
