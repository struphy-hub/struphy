import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace

from struphy.linear_algebra.petsc_solver import PETScSolver


def _define_tridiagonal_spd_system(n, p):
    """Banded, symmetric positive-definite StencilMatrix with 2p+1 diagonals, and a random exact solution."""
    domain_decomposition = DomainDecomposition([n - p], [False], comm=MPI.COMM_WORLD)
    cart = CartDecomposition(domain_decomposition, [n], [xp.array([0])], [xp.array([n - 1])], [p], [1])
    V = StencilVectorSpace(cart)
    s = V.starts[0]
    e = V.ends[0]

    A = StencilMatrix(V, V)
    A[:, -p:0] = -1.0
    A[:, 0:1] = 2 * p
    A[:, 1 : p + 1] = -1.0
    A.remove_spurious_entries()

    xe = StencilVector(V)
    xe[s : e + 1] = xp.random.random(e + 1 - s)

    return V, A, xe


@pytest.mark.parametrize("n", [8, 15])
@pytest.mark.parametrize("p", [1, 2])
def test_petsc_solver_matches_cg(n, p):
    """PETScSolver must solve Ax=b to the same accuracy as feectools' native CG solver."""
    xp.random.seed(n * p)

    _, A, xe = _define_tridiagonal_spd_system(n, p)

    b = A @ xe

    ref_solver = inverse(A, "cg", tol=1e-13, maxiter=2000, verbose=False, recycle=False)
    x_ref = ref_solver.solve(b)

    petsc_solver = PETScSolver(A, tol=1e-13, maxiter=2000, ksp_type="cg", pc_type="none")
    x_petsc = petsc_solver.solve(b)

    info = petsc_solver.get_info()
    assert info["success"]

    error_vs_exact = xp.linalg.norm((x_petsc - xe).toarray())
    assert error_vs_exact < 1e-8

    error_vs_ref = xp.linalg.norm((x_petsc - x_ref).toarray())
    assert error_vs_ref < 1e-6

    # re-solving with an unchanged operator (KSP/Mat cache reused) must still be correct
    b2 = A @ x_petsc
    x_petsc2 = petsc_solver.solve(b2)
    assert xp.linalg.norm((x_petsc2 - x_petsc).toarray()) < 1e-8


if __name__ == "__main__":
    test_petsc_solver_matches_cg(15, 2)
