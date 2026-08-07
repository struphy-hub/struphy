import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.topology.grids import TensorProductGrid


def test_poisson_solve_petsc_matches_pcg():
    """PoissonSolve(solver="petsc") must match PoissonSolve(solver="pcg") on a fully periodic domain.

    PETScSolver can only assemble derham.grad on a *periodic* differentiation axis (see
    struphy.linear_algebra.petsc_solver._directional_derivative_to_stencil_matrix); this is why
    the domain here is fully periodic rather than using Dirichlet/Neumann boundaries.
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

    def sol_xyz(x, y, z):
        return xp.sin(2 * xp.pi * x) * xp.cos(2 * xp.pi * y)

    def rho_xyz(x, y, z):
        return sol_xyz(x, y, z) * ((2 * xp.pi) ** 2 + (2 * xp.pi) ** 2)

    def rho_pulled(e1, e2, e3):
        return domain.pull(rho_xyz, e1, e2, e3, kind="0", squeeze_out=False)

    def run(solver_name):
        solver_params = SolverParameters(tol=1e-11, maxiter=3000, info=False, recycle=False)

        phi = FEECVariable(space="H1")
        phi.allocate(derham=derham, domain=domain)

        prop = PoissonSolve(rho=rho_pulled)
        prop.variables.phi = phi
        prop.options = prop.Options(
            stab_eps=1e-12,
            solver=solver_name,
            precond="MassMatrixPreconditioner",
            solver_params=solver_params,
        )
        prop.allocate()
        prop(1.0)
        return phi

    phi_pcg = run("pcg")
    phi_petsc = run("petsc")

    e1 = xp.linspace(0.0, 1.0, 20)
    e2 = xp.linspace(0.0, 1.0, 20)
    e3 = xp.array([0.5])

    val_pcg = domain.push(phi_pcg.spline, e1, e2, e3, kind="0")
    val_petsc = domain.push(phi_petsc.spline, e1, e2, e3, kind="0")

    x, y, z = domain(e1, e2, e3)
    analytic = sol_xyz(x, y, z)

    assert xp.max(xp.abs(val_petsc - analytic)) < 1e-2
    assert xp.max(xp.abs(val_petsc - val_pcg)) < 1e-6


if __name__ == "__main__":
    test_poisson_solve_petsc_matches_pcg()
