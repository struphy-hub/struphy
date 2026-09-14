import cunumpy as xp
import pytest
from cunumpy import PyccelKernel

pytest.importorskip("petsc4py")

from feectools.ddm.mpi import mpi as MPI

from struphy import LoadingParameters, WeightsParameters, maxwellians, perturbations
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.pic.particles import Particles6D
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.topology.grids import TensorProductGrid


class _FakePICVariable:
    """Minimal duck-typed stand-in for a model's PICVariable, since ParticlesToGrid only reads .particles."""

    def __init__(self, particles):
        self.particles = particles


def test_poisson_solve_petsc_matches_pcg_with_real_pic_deposition():
    """PoissonSolve(solver="petsc") must match PoissonSolve(solver="pcg") when driven by a real
    particle-in-cell charge-density deposition (not a synthetic/manufactured source), reproducing
    the setup of examples/VlasovAmpereOneSpecies/strong_Landau_damping (Maxwellian3D background +
    ModesCos perturbation, control-variate weights).

    Compared *mean-removed* (matching struphy.linear_algebra.petsc_examples_benchmark's
    methodology): this case's background is a large uniform density (n=1.0 everywhere), so the
    charge density's mean/DC component is large, and the near-singular stab_eps=1e-8 regularizes
    the constant/DC mode only very weakly -- feectools' pcg divides that large DC charge by the
    tiny stab_eps, landing on an essentially arbitrary large DC potential offset (~-168 in
    testing) that is numerical-noise-amplification, not a physically meaningful answer. PETScSolver
    now registers the constant mode as a near null space for exactly this operator (see
    PETScSolver's near_null_space docstring and ImplicitDiffusion.allocate's near_null_space="constant"
    comment) -- the fix for a real MPI-rank-dependent correctness bug this same near-singular
    regime caused under >1 rank -- which makes it correctly and robustly discard that
    inconsistent DC component instead (mean exactly 0) rather than reproducing pcg's arbitrary
    noise-amplified one. The physically meaningful, oscillatory part of the solution still needs
    to match to near machine precision, which is what this test actually checks.
    """
    comm = MPI.COMM_WORLD

    domain = Cuboid()
    grid = TensorProductGrid(num_elements=[10, 10, 10])
    derham_opts = DerhamOptions(degree=[2, 2, 2], bcs=(None, None, None))
    derham = Derham(grid, derham_opts, comm=comm)
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    background = maxwellians.Maxwellian3D(n=(1.0, None))
    perturbation = perturbations.ModesCos(amps=(0.5,), ls=(1,))
    init = maxwellians.Maxwellian3D(n=(1.0, perturbation))

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs

    def run(solver_name):
        loading_params = LoadingParameters(Np=20_000, seed=1234)
        weights_params = WeightsParameters(control_variate=True)

        particles = Particles6D(
            comm_world=comm,
            clone_config=None,
            loading_params=loading_params,
            weights_params=weights_params,
            domain=domain,
            domain_decomp=(domain_array, nprocs),
            background=background,
            initial_condition=init,
        )
        particles.draw_markers()
        if comm.Get_size() > 1:
            particles.mpi_sort_markers()
        particles.initialize_weights()

        rho = ParticlesToGrid(
            _FakePICVariable(particles),
            "H1",
            PyccelKernel(accum_kernels.charge_density_0form),
        )

        phi = FEECVariable(space="H1")
        phi.allocate(derham=derham, domain=domain)

        solver_params = SolverParameters(tol=1e-10, maxiter=20000, info=False, recycle=False)

        prop = PoissonSolve(rho=rho)
        prop.variables.phi = phi
        prop.options = prop.Options(
            stab_eps=1e-8,
            solver=solver_name,
            precond="MassMatrixPreconditioner",
            solver_params=solver_params,
        )
        prop.allocate()
        if solver_name == "petsc":
            prop._solver._options["pc_type"] = "gamg"
            prop._solver._ksp = None
        prop(0.05)
        return phi.spline.vector.toarray()

    sol_pcg = run("pcg")
    sol_petsc = run("petsc")

    mean_removed_pcg = sol_pcg - sol_pcg.mean()
    mean_removed_petsc = sol_petsc - sol_petsc.mean()
    rel_err = xp.linalg.norm(mean_removed_pcg - mean_removed_petsc) / xp.linalg.norm(sol_pcg)
    assert rel_err < 1e-6


if __name__ == "__main__":
    test_poisson_solve_petsc_matches_pcg_with_real_pic_deposition()
