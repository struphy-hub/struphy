"""Benchmark: PoissonSolve(solver="petsc") vs PoissonSolve(solver="pcg") on a real Vlasov-Poisson testcase.

Builds actual :class:`~struphy.simulation.sim.Simulation` objects, using the same public API and
setup idiom as a real parameter file (compare
``examples/VlasovAmpereOneSpecies/strong_Landau_damping/params_strong_Landau_damping.py``,
generalized here to a 3D grid), and runs them with ``sim.run(one_time_step=True)`` -- this is the
one and only supported entry point for allocating and running a struphy simulation; it performs
real particle loading, real Derham/mass-operator setup, and (for this model) a real charge-density
deposition via ``ParticlesToGrid``/``AccumulatorVector``, solved via ``PoissonSolve`` exactly as
``VlasovAmpereOneSpecies.allocate_helpers`` does.

Two earlier, less realistic benchmarks are *not* wins for PETSc and are not repeated here:

- Mass-matrix solves (``L2Projector``): already well-conditioned, feectools' native
  preconditioned CG wins outright.
- ``PoissonSolve`` with a synthetic, non-mass-weighted random right-hand side: not representative
  of any real code path (every real source -- ``FEECVariable``, ``ParticlesToGrid``, ``Callable``
  -- is mass-matrix weighted when forming the weak-form right-hand side, which is inherently
  smoothing).

The genuine win requires: a small ``stab_eps`` (true elliptic Poisson, matching realistic
electrostatic PIC parameters -- ``stab_eps`` is a numerical regularization, not a dominant
physical diffusion), a broadband/noisy right-hand side (real PIC deposition, not a smooth
manufactured mode), and repeated solves at fixed ``dt`` (relies on
``ImplicitDiffusion.__call__`` caching its lhs operator when ``sig_1`` is unchanged, so
``PETScSolver`` can reuse its assembled matrix -- see git history for that fix). Since
``VlasovAmpereOneSpecies`` only calls its Poisson solve *once* (as an initial condition -- the
electric field then evolves via Ampere's law, not repeated Poisson solves), the repeated-solve
timing below re-invokes ``model.initial_poisson`` directly after ``sim.run()`` has performed the
real setup, rather than relying on the model's own (single-shot) usage of it.

Run with:

.. code-block:: bash

    python3 -m struphy.linear_algebra.petsc_poisson_benchmark
"""

import shutil
import tempfile
import time
import warnings

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import VlasovAmpereOneSpecies

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def build_and_run(num_elements, degree, ppc, solver_name, perturbation, pc_type=None, stab_eps=1e-8, out_folder=None):
    """Build a VlasovAmpereOneSpecies Simulation exactly as a params.py file would, and run one
    step of it via sim.run(one_time_step=True) -- the real, supported entry point. This performs
    real particle loading and the real (single-shot) initial Poisson solve.
    """
    # alpha=1.0, epsilon=-1.0 (matching the real strong_Landau_damping example) keeps the
    # right-hand side well-scaled (order 1); epsilon in particular can never be auto-derived as
    # negative (its formula is always positive), so overriding it is unavoidable here, and
    # struphy warns on every such override. The warning is expected and harmless -- silence it
    # rather than leaving alpha/epsilon at their auto-derived values, which was tried and
    # produces a poorly-scaled right-hand side (huge/tiny relative to 1), breaking the implicit
    # assumption -- shared by every SolverParameters.tol comparison in this benchmark -- that
    # "tol" means the same thing regardless of problem scale.
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message="Override equation parameter", category=UserWarning)
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

    env = EnvironmentOptions(out_folders=out_folder, sim_folder=f"bench_{solver_name}")
    time_opts = Time(dt=0.05, Tend=0.05, split_algo="LieTrotter")
    domain = domains.Cuboid()
    grid = grids.TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=(None, None, None))

    sim = Simulation(
        model=model,
        params_path=None,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=None,
        grid=grid,
        derham_opts=derham_opts,
    )

    loading_params = LoadingParameters(ppc=ppc, seed=1234)
    weights_params = WeightsParameters(control_variate=True)
    boundary_params = BoundaryParameters()
    model.kinetic_ions.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(
        stab_mat="M0",
        stab_eps=stab_eps,
        solver=solver_name,
        precond="MassMatrixPreconditioner",
        solver_params=SolverParameters(tol=1e-10, maxiter=20_000, info=False, recycle=False),
    )

    background = maxwellians.Maxwellian3D(n=(1.0, None))
    model.kinetic_ions.var.add_background(background)
    init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
    model.kinetic_ions.var.add_initial_condition(init)

    sim.run(one_time_step=True)

    if solver_name == "petsc" and pc_type is not None and model.initial_poisson._solver._options["pc_type"] != pc_type:
        model.initial_poisson._solver._options["pc_type"] = pc_type
        model.initial_poisson._solver._ksp = None

    return sim, model


def bench_case(name, num_elements, degree, ppc, perturbation, n_solves=10, dt=0.05, **kwargs):
    # tempfile.mkdtemp() is not MPI-coordinated: each rank would otherwise get a *different*
    # random path, and Simulation's output-file creation (rank 0 only) would then fail on every
    # other rank. Create it on rank 0 and broadcast the path instead.
    out_folder = tempfile.mkdtemp() if rank == 0 else None
    if comm is not None:
        out_folder = comm.bcast(out_folder, root=0)

    try:
        _, model_cg = build_and_run(num_elements, degree, ppc, "pcg", perturbation, out_folder=out_folder, **kwargs)
        _, model_petsc = build_and_run(
            num_elements, degree, ppc, "petsc", perturbation, pc_type="gamg", out_folder=out_folder, **kwargs
        )

        # both models' initial_poisson already ran once inside sim.run(); time repeated calls at
        # fixed dt, exactly as a real timestepping loop would (see module docstring)
        model_cg.initial_poisson(dt)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n_solves):
            model_cg.initial_poisson(dt)
        t_cg = (time.perf_counter() - t0) / n_solves
        info_cg = model_cg.initial_poisson._solver._info

        model_petsc.initial_poisson(dt)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n_solves):
            model_petsc.initial_poisson(dt)
        t_petsc = (time.perf_counter() - t0) / n_solves
        info_petsc = model_petsc.initial_poisson._solver.get_info()

        sol_cg = model_cg.em_fields.phi.spline.vector.toarray()
        sol_petsc = model_petsc.em_fields.phi.spline.vector.toarray()
        rel_err = xp.linalg.norm(sol_cg - sol_petsc) / xp.linalg.norm(sol_cg)
    finally:
        if comm is not None:
            comm.Barrier()
        if rank == 0:
            shutil.rmtree(out_folder, ignore_errors=True)

    ndofs = model_cg.em_fields.phi.spline.vector.space.dimension
    Np = model_cg.kinetic_ions.var.particles.markers.shape[0]

    if rank == 0:
        print(f"\n{name}: num_elements={num_elements}, degree={degree}, ndofs={ndofs}, Np~{Np}")
        print(f"  pcg (unprec.) : {t_cg * 1e3:9.2f} ms/step  niter={info_cg.get('niter')}")
        print(f"  petsc + gamg  : {t_petsc * 1e3:9.2f} ms/step  niter={info_petsc.get('niter')}")
        print(f"  speedup: {t_cg / t_petsc:.2f}x   relative solution mismatch: {rel_err:.2e}")


def main():
    # 1. grid-size scaling, matching examples/VlasovAmpereOneSpecies/strong_Landau_damping's ICs
    landau_damping = perturbations.ModesCos(amps=(0.5,), ls=(1,))
    for num_elements in [[8, 8, 8], [16, 16, 16], [24, 24, 24], [32, 32, 32]]:
        bench_case("Landau damping (grid scaling)", num_elements, [2, 2, 2], ppc=20, perturbation=landau_damping)

    # 2. weak Landau damping ICs (small-amplitude perturbation, closer to the linear regime)
    weak_landau_damping = perturbations.ModesCos(amps=(0.001,), ls=(1,))
    bench_case("weak Landau damping", [24, 24, 24], [2, 2, 2], ppc=20, perturbation=weak_landau_damping)

    # 3. higher spline degree (matching the real examples' degree=3 in the perturbed direction)
    bench_case("Landau damping, degree 3", [16, 16, 16], [3, 3, 3], ppc=20, perturbation=landau_damping)

    # 4. sparser sampling (fewer particles per cell -> noisier deposited density)
    bench_case("Landau damping, low ppc (noisier)", [24, 24, 24], [2, 2, 2], ppc=5, perturbation=landau_damping)

    # 5. genuinely 3D, multi-mode perturbation (unlike the 1D-in-x real examples), a closer
    # stand-in for 3D electrostatic turbulence
    multi_mode_3d = perturbations.ModesCos(amps=(0.5, 0.3, 0.2), ls=(1, 2, 0), ms=(0, 1, 2), ns=(0, 0, 1))
    bench_case("3D multi-mode perturbation", [24, 24, 24], [2, 2, 2], ppc=20, perturbation=multi_mode_3d)


if __name__ == "__main__":
    main()
