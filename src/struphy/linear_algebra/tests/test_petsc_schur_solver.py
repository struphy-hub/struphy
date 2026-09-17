"""Regression tests for SchurSolver's solver="petsc" support (struphy.linear_algebra.schur_solver).

SchurSolver previously always imported feectools' own `inverse`, which does not recognize the
name "petsc" -- so `solver="petsc"` would raise for every propagator built on it (MaxwellWeakAmpere,
VlasovAmpereCoupling, EfieldWeightsCoupling, CurlCurlSolve, ...), even though those propagators'
Options all declare `LiteralOptions.OptsSymmSolver` (which includes "petsc") as their solver type.

Wiring petsc in was not just an import swap: PETScSolver caches its assembled PETSc.Mat by the
*object identity* of the operator assigned to `.linop` (see PETScSolver._get_ksp), rebuilding only
when that identity changes. SchurSolver's non-petsc path mutates its `self._schur` buffer *in
place* every call (`self._schur *= 0.0; += ...`) -- the same Python object every time, which would
make PETScSolver silently reuse a stale matrix forever after the first call. Two call patterns
exist among current callers, and both need to be correct:

- MaxwellWeakAmpere never reassigns `.A`/`.BC` after construction (both are geometric, constant
  operators) -- caching by `dt` alone is correct and safe there.
- VlasovAmpereCoupling (and EfieldWeightsCoupling) reassign `.BC` to a fresh, particle-dependent
  operator via the property setter on *every* call -- caching there must be invalidated every
  time, tracked via a dirty flag set in the `A`/`BC` property setters (Python routes both
  `x.BC = y` and the augmented `x.BC *= y` through the setter).

These tests exercise both patterns directly (not synthetically) via the real propagators.
"""

import shutil
import tempfile

import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.mpi import mpi as MPI

from struphy import (
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
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.domains import Cuboid
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import VlasovAmpereOneSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.topology.grids import TensorProductGrid


def test_maxwell_weak_ampere_petsc_matches_pcg():
    """MaxwellWeakAmpere(solver="petsc") must match solver="pcg" over several implicit timesteps.

    Exercises SchurSolver's petsc dt-only cache-invalidation path (see module docstring):
    MaxwellWeakAmpere never reassigns `.A`/`.BC` after allocate(), so the same lhs operator must
    be correctly reused across all calls at fixed dt.
    """
    comm = MPI.COMM_WORLD

    domain = Cuboid()
    grid = TensorProductGrid(num_elements=[6, 6, 6])
    derham_opts = DerhamOptions(degree=[2, 2, 2], bcs=(None, None, None))
    derham = Derham(grid, derham_opts, comm=comm)
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    def run(solver_name):
        e_field = FEECVariable(space="Hcurl")
        e_field.add_perturbation(perturbations.ModesCos(amps=(0.1,), ls=(1,), comp=0))
        e_field.allocate(derham=derham, domain=domain)

        b_field = FEECVariable(space="Hdiv")
        b_field.add_perturbation(perturbations.ModesCos(amps=(0.05,), ls=(1,), comp=1))
        b_field.allocate(derham=derham, domain=domain)

        prop = MaxwellWeakAmpere()
        prop.variables.e = e_field
        prop.variables.b = b_field
        prop.options = prop.Options(
            algo="implicit",
            solver=solver_name,
            solver_params=SolverParameters(tol=1e-11, maxiter=3000),
        )
        prop.allocate()

        dt = 0.02
        for _ in range(3):
            prop(dt)

        return e_field.spline.vector.toarray(), b_field.spline.vector.toarray()

    e_pcg, b_pcg = run("pcg")
    e_petsc, b_petsc = run("petsc")

    rel_err_e = xp.linalg.norm(e_pcg - e_petsc) / xp.linalg.norm(e_pcg)
    rel_err_b = xp.linalg.norm(b_pcg - b_petsc) / xp.linalg.norm(b_pcg)
    assert rel_err_e < 1e-6, f"e-field mismatch: {rel_err_e:.2e}"
    assert rel_err_b < 1e-6, f"b-field mismatch: {rel_err_b:.2e}"


def test_vlasov_ampere_coupling_petsc_matches_pcg_with_real_pic_deposition():
    """VlasovAmpereCoupling(solver="petsc") must match solver="pcg" over several real timesteps,
    driven by real particle-in-cell deposition (not a synthetic source).

    Exercises SchurSolver's petsc dirty-flag cache-invalidation path (see module docstring):
    VlasovAmpereCoupling reassigns `.BC` to a fresh, particle-dependent operator every call, which
    the dt-only cache used by MaxwellWeakAmpere's test above would get wrong if applied here.
    Goes through the full model/Simulation machinery (unlike the other petsc regression tests in
    this directory, which build propagators directly) because VlasovAmpereCoupling requires a real
    PICVariable/ParticleSpecies (species.equation_params, weights_params) that is impractical to
    duck-type -- see test_petsc_poisson_solve_pic.py, whose ParticlesToGrid-based fake works
    because ParticlesToGrid does not check isinstance, unlike VlasovAmpereCoupling.Variables.ions.
    """
    comm = MPI.COMM_WORLD

    def run(solver_name, out_folder):
        model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

        env = EnvironmentOptions(out_folders=out_folder, sim_folder=f"sim_{solver_name}")
        time_opts = Time(dt=0.02, Tend=0.06, split_algo="LieTrotter")
        domain = domains.Cuboid(r1=12.56)
        grid = grids.TensorProductGrid(num_elements=(8, 8, 8))
        derham_opts = DerhamOptions(degree=(2, 2, 2), bcs=(None, None, None))

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

        model.kinetic_ions.set_markers(
            loading_params=LoadingParameters(Np=5_000, seed=1234),
            weights_params=WeightsParameters(control_variate=True),
        )

        model.propagators.push_eta.options = model.propagators.push_eta.Options()
        model.propagators.coupling_va.options = model.propagators.coupling_va.Options(
            solver=solver_name,
            solver_params=SolverParameters(tol=1e-10, maxiter=5000),
        )
        model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

        background = maxwellians.Maxwellian3D(n=(1.0, None))
        perturbation = perturbations.ModesCos(amps=(0.5,), ls=(1,))
        init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
        model.kinetic_ions.var.add_background(background)
        model.kinetic_ions.var.add_initial_condition(init)

        sim.run()  # several real steps, not one_time_step: coupling_va.BC changes every call

        return model.em_fields.e_field.spline.vector.toarray()

    out_folder = tempfile.mkdtemp() if comm.Get_rank() == 0 else None
    out_folder = comm.bcast(out_folder, root=0)

    try:
        e_pcg = run("pcg", out_folder)
        e_petsc = run("petsc", out_folder)
    finally:
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(out_folder, ignore_errors=True)

    rel_err = xp.linalg.norm(e_pcg - e_petsc) / xp.linalg.norm(e_pcg)
    assert rel_err < 1e-6, f"e-field mismatch: {rel_err:.2e}"


if __name__ == "__main__":
    test_maxwell_weak_ampere_petsc_matches_pcg()
    test_vlasov_ampere_coupling_petsc_matches_pcg_with_real_pic_deposition()
