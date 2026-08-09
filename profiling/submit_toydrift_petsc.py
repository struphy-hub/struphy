"""ToyDrift periodic-slab: PETSc vs. pcg for a real per-step Poisson solve.

This file defines two profiling cases (the `ProfilingCase`s) built from the same periodic-slab
ToyDrift setup (see `profiling/examples/ToyDrift/periodic_slab/params_periodic_slab_{pcg,petsc}.py`
-- there is no periodic ToyDrift example under `examples/`: the real one,
`examples/ToyGyrokinetic/diocotron_instability`, uses a physically non-periodic HollowCylinder
domain, since radial confinement is inherent to the diocotron instability; this case swaps in a
periodic Cuboid domain instead, which works because ToyDrift's field solve is a plain
`PoissonSolve` with no geometry-coupled averaging -- unlike `PoissonAdiabaticGyrokinetic`, used by
`DriftKineticElectrostaticAdiabatic`, which was tried first and diverges outright on a periodic
domain regardless of options), differing only in the solver used for the field solve
(`model.propagators.gc_poisson.options.solver`, `"pcg"` vs. `"petsc"`).

Unlike VlasovAmpereOneSpecies (whose Poisson solve only runs once, as an initial condition),
ToyDrift's `gc_poisson` runs as a *regular per-step propagator* -- exactly the repeated-solve
pattern where PETSc's algebraic multigrid preconditioner shows a genuine win, without needing the
initial-Poisson-only benchmark's workaround of re-invoking the propagator by hand after
`sim.run()`. Each generated script runs the simulation itself by invoking the corresponding
`params_periodic_slab_{pcg,petsc}.py` directly (its `__main__` block calls
`sim.run(one_time_step=True)`, a single-step timing snapshot).

See `submit_strong_landau_damping_petsc.py` for the same comparison on VlasovAmpereOneSpecies, and
its module docstring / `struphy.linear_algebra.petsc_examples_benchmark`'s for the general PETSc
vs. pcg story (including the known MPI correctness caveat for this near-singular stab_eps regime).
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
    params_dir = script_dir / "examples" / "ToyDrift" / "periodic_slab"

    profiling_cases = {
        "pcg": ProfilingCase(
            label="toydrift_periodic_slab_pcg",
            name="ToyDrift periodic slab, per-step Poisson solve with pcg",
            description=(
                "Periodic-slab ToyDrift setup, solving the per-step guiding-center Poisson "
                "problem with feectools' native preconditioned CG."
            ),
            physics_problem="Electrostatic E x B drift of a single ion species in a periodic slab.",
            struphy_model_used="ToyDrift",
            params_source=params_dir / "params_periodic_slab_pcg.py",
            language="fortran",
            compiler="GNU",
            upload=args.upload,
        ),
        "petsc": ProfilingCase(
            label="toydrift_periodic_slab_petsc",
            name="ToyDrift periodic slab, per-step Poisson solve with PETSc",
            description=(
                "Same setup as 'toydrift_periodic_slab_pcg', but solving the per-step "
                "guiding-center Poisson problem with PETScSolver (KSP=cg, PC=gamg) instead."
            ),
            physics_problem="Electrostatic E x B drift of a single ion species in a periodic slab.",
            struphy_model_used="ToyDrift",
            params_source=params_dir / "params_periodic_slab_petsc.py",
            language="fortran",
            compiler="GNU",
            upload=args.upload,
        ),
    }

    for profiling_case in profiling_cases.values():
        profiling_case.use_slurm = False

        # Launch one run per rank count, same counts for both solvers so they are directly
        # comparable.
        for num_tasks in (1, 2, 4):
            profiling_case.launch(num_tasks)

    # Package and push each run as its own job finishes.
    for profiling_case in profiling_cases.values():
        profiling_case.finalize_run()


if __name__ == "__main__":
    main()
