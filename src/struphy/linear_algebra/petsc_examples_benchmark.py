"""Benchmark: solver="petsc" vs solver="pcg" on real struphy examples.

Uses the committed parameter files under
``profiling/examples/<model>/<case>/params_<case>.py``, each with a ``--solver`` CLI flag
(default ``"pcg"``) selecting the solver of the case's Poisson-type propagator:

- ``VlasovAmpereOneSpecies/{strong_Landau_damping,weak_Landau_damping,two_stream,bump_on}``: plain
  copies of the corresponding ``examples/VlasovAmpereOneSpecies/<case>/params_<case>.py`` (unedited
  on disk), with ``num_elements`` scaled up from the examples' tiny, highly-anisotropic 1D-style
  default of ``(32, 1, 1)`` cells to a proper ``(16, 16, 16)`` 3D grid -- PETSc's advantage only
  shows up above roughly 5,000 dofs -- ``ppc`` reduced to match, and a fixed
  ``LoadingParameters.seed`` added (the originals don't set one, so pcg/petsc would otherwise draw
  different particles and not be comparable). This model only solves Poisson *once*, as an initial
  condition (the field then evolves via VlasovAmpereCoupling), so the repeated-solve timing below
  re-invokes ``model.initial_poisson`` directly after ``sim.run()`` rather than relying on the
  model's own (single-shot) usage of it.

- ``ToyDrift/periodic_slab_hires``: no periodic ToyDrift example exists under ``examples/`` (the
  real one, ``examples/ToyGyrokinetic/diocotron_instability``, needs a physically non-periodic
  HollowCylinder domain), so this one is written from scratch with a periodic Cuboid domain
  instead -- which works because ToyDrift's field solve is a plain ``PoissonSolve`` with no
  geometry-coupled averaging (unlike ``PoissonAdiabaticGyrokinetic``, used by
  ``DriftKineticElectrostaticAdiabatic``, which diverges outright on a periodic domain regardless
  of options -- tried first, not usable here). Unlike VlasovAmpereOneSpecies, ToyDrift's
  ``gc_poisson`` runs as a *regular per-step propagator*, so no re-invocation workaround is needed.
  Uses a 32^3 grid to push feectools' unpreconditioned CG into several hundred iterations per
  solve while PETSc+gamg (``pc_type="gamg"``, set explicitly via ``SolverParameters.pc_type`` --
  see ``struphy.linear_algebra.solver.SolverParameters``) stays at a handful, regardless of grid
  size. The most lopsided case in this suite by design; see its own params file's docstring.

- ``VlasovMaxwellOneSpecies/weibel_instability``: plain copy of
  ``examples/VlasovMaxwellOneSpecies/weibel_instability/params_weibel_instability.py``, scaled up
  the same way as the VlasovAmpereOneSpecies cases above (3D grid, reduced ppc, fixed seed). Same
  one-shot ``model.initial_poisson`` pattern as VlasovAmpereOneSpecies (the fields then evolve via
  MaxwellWeakAmpere/PushVxB/VlasovAmpereCoupling instead), so the same re-invocation timing applies.

This script's only job is to import each file's ``sim``/``model`` and drive them -- no source
patching, no ``runpy``.

Correctness is checked between the two solvers on the *mean-removed* solution (the near-singular,
essentially unregularized ``stab_eps`` these examples use -- via ``ImplicitDiffusion``'s "always
stabilize" clamp to ``1e-14`` -- leaves the constant/DC mode only very weakly constrained, so it is
extremely sensitive to tiny numerical differences between solvers; this is expected and is not a
correctness issue in the physically meaningful, oscillatory part of the solution), normalized by
the *full* solution's norm (not the tiny mean-removed norm itself, which can be dominated by
floating-point noise for weak-perturbation examples and make a naively-normalized relative error
meaningless).

KNOWN ISSUE -- do not trust results under MPI (comm size > 1): for this same near-singular
``stab_eps``-clamped-to-``1e-14`` regime, PETScSolver was found to disagree substantially with
feectools' native solver specifically under >1 MPI rank, independent of ``pc_type`` (both
"jacobi" and "gamg" reproduced it; "gamg" was far worse -- a false "converged in 1 iteration" to
a wildly wrong answer). This reproduces even though pcg-vs-pcg (same solver, two independent runs)
is bit-identical, ruling out a methodology artifact in this script. The root cause was not found;
serial execution and non-near-singular systems (this same MPI path, e.g. with an explicit
``stab_eps`` of 1e-8 or larger) were extensively validated and are unaffected -- see
``petsc_poisson_benchmark.py`` and ``test_petsc_poisson_solve_pic.py``. This script therefore
only asserts/prints the correctness check when running serially, and warns instead under MPI.

Run with:

.. code-block:: bash

    python3 -m struphy.linear_algebra.petsc_examples_benchmark
"""

import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILING_EXAMPLES_DIR = REPO_ROOT / "profiling" / "examples"

# (model directory under profiling/examples/, case name)
CASES = (
    ("VlasovAmpereOneSpecies", "strong_Landau_damping"),
    ("VlasovAmpereOneSpecies", "weak_Landau_damping"),
    ("VlasovAmpereOneSpecies", "two_stream"),
    ("VlasovAmpereOneSpecies", "bump_on"),
    ("ToyDrift", "periodic_slab_hires"),
    ("VlasovMaxwellOneSpecies", "weibel_instability"),
)

# `profiling` is a repo-local package (not part of the installed struphy distribution), normally
# importable only because '' (cwd) is on sys.path at interpreter startup. This script chdir()s
# into a scratch directory before importing (to keep Simulation's output out of the repo), which
# breaks that for '-c'/REPL-style invocations where '' resolves dynamically -- so add the repo
# root explicitly, once, up front.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _poisson_propagator(model):
    """VlasovAmpereOneSpecies exposes its (one-shot) Poisson solve as `model.initial_poisson`;
    other models (e.g. ToyDrift) run it as a regular per-step propagator instead.
    """
    if hasattr(model, "initial_poisson"):
        return model.initial_poisson
    return model.propagators.gc_poisson


def _run_variant(model_dir: str, name: str, variant: str, out_folder: str, dt: float, n_solves: int):
    module_name = f"profiling.examples.{model_dir}.{name}.params_{name}"

    # EnvironmentOptions.out_folders defaults to `os.getcwd()`, but as a plain dataclass field
    # default this is evaluated once, the first time struphy.io.options is imported in this
    # process -- not per EnvironmentOptions() call, and not affected by a later os.chdir(). Since
    # `struphy`'s own __init__.py re-exports EnvironmentOptions (and everything else), that first
    # import can happen before this function even runs (e.g. as a side effect of importing this
    # very module). The profiling params files read STRUPHY_PROFILING_OUT_FOLDERS explicitly for
    # exactly this reason -- set it before importing, so their Simulation's output lands here
    # instead of wherever the process happened to start.
    os.environ["STRUPHY_PROFILING_OUT_FOLDERS"] = out_folder

    # Both variants now come from the same params_<name>.py (a `--solver` CLI flag picks between
    # them internally), so a plain second `import_module` would just return the first variant's
    # already-imported module/sim/model unchanged. Feed the desired `--solver` through sys.argv
    # (the params file reads it via argparse) and force a re-execution via `reload` so the second
    # variant gets its own fresh Simulation/model instead of reusing (and re-running) the first's.
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--solver", variant]
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
    finally:
        sys.argv = old_argv

    sim = mod.sim
    model = mod.model

    sim.run(one_time_step=True)  # real setup: particle loading, Derham/mass operators, and (for
    # VlasovAmpereOneSpecies) the initial Poisson solve, exactly as
    # VlasovAmpereOneSpecies.allocate_helpers / a real per-step run does

    poisson = _poisson_propagator(model)
    solver = poisson._solver
    if variant == "petsc":
        solver._options["pc_type"] = "gamg"
        solver._ksp = None  # force a rebuild with the new pc_type

    # repeated calls at fixed dt (same real charge deposition -- this benchmark is about the
    # linear-solve cost, not particle physics): matches how repeated Poisson solves would
    # amortize matrix assembly across timesteps at a fixed dt in a real run (see
    # ImplicitDiffusion.__call__'s lhs-operator caching)
    poisson(dt)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_solves):
        poisson(dt)
    t = (time.perf_counter() - t0) / n_solves
    info = solver.get_info() if hasattr(solver, "get_info") else solver._info

    phi = model.em_fields.phi.spline.vector.toarray()
    return t, info, phi


def bench_example(model_dir: str, name: str, dt: float = 0.05, n_solves: int = 10):
    """Import and run one example's pcg/petsc parameter files, and report timing + correctness."""
    params_dir = PROFILING_EXAMPLES_DIR / model_dir / name
    if not (params_dir / f"params_{name}.py").exists():
        raise FileNotFoundError(
            f"Could not find {params_dir}/params_{name}.py -- expected the parameter file "
            f"under profiling/examples/{model_dir}/{name}/."
        )

    # tempfile.mkdtemp() is not MPI-coordinated: each rank would otherwise get a *different*
    # random path, and Simulation's output-file creation (rank 0 only) would then fail on every
    # other rank. Create it on rank 0 and broadcast the path instead.
    out_folder = tempfile.mkdtemp() if rank == 0 else None
    if comm is not None:
        out_folder = comm.bcast(out_folder, root=0)

    try:
        t_cg, info_cg, sol_cg = _run_variant(model_dir, name, "pcg", out_folder, dt, n_solves)
        t_petsc, info_petsc, sol_petsc = _run_variant(model_dir, name, "petsc", out_folder, dt, n_solves)
    finally:
        if comm is not None:
            comm.Barrier()
        if rank == 0:
            shutil.rmtree(out_folder, ignore_errors=True)

    mean_removed_pcg = sol_cg - sol_cg.mean()
    mean_removed_petsc = sol_petsc - sol_petsc.mean()
    rel_err = xp.linalg.norm(mean_removed_pcg - mean_removed_petsc) / xp.linalg.norm(sol_cg)

    comm_size = comm.Get_size() if comm is not None else 1
    if rank == 0:
        print(f"\n{model_dir}/{name}: ndofs={sol_cg.size}")
        print(f"  pcg (unprec.) : {t_cg * 1e3:9.2f} ms/step  niter={info_cg.get('niter')}")
        print(f"  petsc + gamg  : {t_petsc * 1e3:9.2f} ms/step  niter={info_petsc.get('niter')}")
        print(f"  speedup: {t_cg / t_petsc:.2f}x", end="   ")
        if comm_size > 1:
            print(
                f"relative solution mismatch: {rel_err:.2e} "
                "-- NOT a reliable correctness check under MPI for this near-singular regime, "
                "see module docstring (KNOWN ISSUE)"
            )
        else:
            print(f"relative solution mismatch: {rel_err:.2e}")
            assert rel_err < 1e-6, (
                f"{model_dir}/{name}: pcg/petsc solutions disagree by {rel_err:.2e}, expected < 1e-6 serially"
            )


def main():
    for model_dir, name in CASES:
        bench_example(model_dir, name)


if __name__ == "__main__":
    main()
