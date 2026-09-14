"""Standalone, self-contained example: PETSc beats pcg on a real (non-toy) Poisson solve.

Run directly -- no profiling infrastructure, no submission, nothing to set up beyond an
environment with petsc4py installed (``pip install -e ".[petsc]"``):

.. code-block:: bash

    python src/struphy/linear_algebra/petsc_speedup_example.py

What it does
------------
Builds the same ToyDrift periodic-slab setup used by
``profiling/examples/ToyDrift/periodic_slab_hires`` (the largest PETSc-vs-pcg gap in struphy's
profiling suite) directly in this script, so you can read top to bottom exactly what is being
solved and how it is timed -- no need to trace through ``ProfilingCase``/submit-script machinery.

ToyDrift's ``gc_poisson`` is a *regular per-step propagator* (unlike e.g. VlasovAmpereOneSpecies,
which only solves Poisson once as an initial condition), so it is called once per timestep with
the left-hand-side operator reused across calls at fixed ``dt`` (see
``ImplicitDiffusion.__call__``'s lhs-operator caching). That means the *first* solve pays for
PETSc's one-time matrix assembly and, with ``pc_type="gamg"``, multigrid hierarchy construction --
this script does one untimed warm-up call for exactly that reason, matching how the cost would
amortize over the many timesteps of a real simulation, before timing several further calls.

What to expect
---------------
At this problem size (32768 dofs), feectools' unpreconditioned CG needs on the order of a few
hundred iterations per solve (several seconds), while PETSc with an algebraic multigrid
preconditioner (``pc_type="gamg"``) needs only a handful (well under a second) -- typically a
30-40x speedup. The two solutions are also checked against each other (mean-removed, since the
near-singular constant/DC mode is only weakly constrained and not physically meaningful here --
see ``PoissonSolve``'s stabilization) and should agree to within floating-point noise.
"""

import time

import cunumpy as xp

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import ToyDrift


def _build_and_run(solver: str, pc_type: str, n_solves: int):
    """Build a fresh ToyDrift periodic-slab simulation and time repeated Poisson solves."""
    model = ToyDrift(base_units=BaseUnits(kBT=1.0))

    env = EnvironmentOptions(sim_folder=f"sim_petsc_speedup_example_{solver}")
    time_opts = Time(dt=0.05, Tend=0.05, split_algo="LieTrotter")
    domain = domains.Cuboid()  # periodic: required for PETScSolver's DirectionalDerivativeOperator
    equil = equils.HomogenSlab(B0z=1.0, n0=1.0)
    grid = grids.TensorProductGrid(num_elements=(32, 32, 32))
    derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(None, None, None))  # fully periodic

    sim = Simulation(
        model=model,
        params_path=None,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=5, seed=42),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(4, 4, 4), do_sort=True),
        bufsize=0.4,
    )

    model.propagators.gc_poisson.options.solver = solver
    model.propagators.gc_poisson.options.solver_params = SolverParameters(
        tol=1e-10,
        maxiter=5_000,
        pc_type=pc_type,  # ignored by pcg, see SolverParameters.pc_type
    )
    model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
        algo="explicit",
        evaluate_e_field=True,
    )

    background = maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(1.0, None), vth_perp=(1.0, None), equil=equil)
    model.kinetic_ions.var.add_background(background)
    perturbation = perturbations.ModesCos(amps=(0.5,), ls=(1,))
    init = maxwellians.GyroMaxwellian2D(n=(1.0, perturbation), vth_para=(1.0, None), vth_perp=(1.0, None), equil=equil)
    model.kinetic_ions.var.add_initial_condition(init)

    # real setup: particle loading, Derham/mass operators, and one (untimed) Poisson solve
    sim.run(one_time_step=True)

    poisson = model.propagators.gc_poisson
    dt = time_opts.dt

    poisson(dt)  # warm-up: pays for one-time matrix assembly / gamg hierarchy construction
    t0 = time.perf_counter()
    for _ in range(n_solves):
        poisson(dt)
    elapsed = (time.perf_counter() - t0) / n_solves

    info = poisson._solver.get_info() if hasattr(poisson._solver, "get_info") else poisson._solver._info
    phi = model.em_fields.phi.spline.vector.toarray()
    return elapsed, info, phi


def main(n_solves: int = 5):
    print(f"Timing {n_solves} repeated Poisson solves per solver (after one warm-up call) ...\n")

    t_pcg, info_pcg, phi_pcg = _build_and_run("pcg", pc_type="jacobi", n_solves=n_solves)
    t_petsc, info_petsc, phi_petsc = _build_and_run("petsc", pc_type="gamg", n_solves=n_solves)

    mean_removed_pcg = phi_pcg - phi_pcg.mean()
    mean_removed_petsc = phi_petsc - phi_petsc.mean()
    rel_err = xp.linalg.norm(mean_removed_pcg - mean_removed_petsc) / xp.linalg.norm(phi_pcg)

    print(f"pcg (unpreconditioned) : {t_pcg * 1e3:9.2f} ms/solve   niter={info_pcg.get('niter')}")
    print(f"petsc + gamg           : {t_petsc * 1e3:9.2f} ms/solve   niter={info_petsc.get('niter')}")
    print(f"\nspeedup: {t_pcg / t_petsc:.2f}x")
    print(f"relative solution mismatch (mean-removed): {rel_err:.2e}")
    assert rel_err < 1e-6, f"pcg/petsc solutions disagree by {rel_err:.2e}, expected < 1e-6"
    print("\nSolutions agree -- the speedup above is not at the cost of correctness.")


if __name__ == "__main__":
    main()
